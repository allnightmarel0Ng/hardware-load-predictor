"""
Server Group Manager
────────────────────────────────────────────────────────────────────────────
CRUD operations for ServerGroup and Server models.

A ServerGroup is a named cluster of servers that share the same business
metric formula.  Servers within the group each have their own Prometheus
endpoint and their own trained model, but they are all driven by the same
business signal.

Public API consumed by the request handler:
  create_group / get_group / list_groups / update_group / delete_group
  add_server / get_server / list_servers / update_server / remove_server
  provision_group_configs — creates a ForecastingConfig for every active
                            server in the group (convenience helper)
"""
from __future__ import annotations

import logging

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.db_models import ForecastingConfig, Server, ServerGroup
from app.schemas.schemas import (
    ServerCreate,
    ServerGroupCreate,
    ServerGroupUpdate,
    ServerUpdate,
)

logger = logging.getLogger(__name__)


# ── ServerGroup CRUD ──────────────────────────────────────────────────────────

def create_group(db: Session, data: ServerGroupCreate) -> ServerGroup:
    if db.query(ServerGroup).filter_by(name=data.name).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"ServerGroup '{data.name}' already exists.",
        )
    group = ServerGroup(**data.model_dump())
    db.add(group)
    db.commit()
    db.refresh(group)
    logger.info("Created ServerGroup %d (%s)", group.id, group.name)
    return group


def get_group(db: Session, group_id: int) -> ServerGroup:
    group = db.get(ServerGroup, group_id)
    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ServerGroup {group_id} not found.",
        )
    return group


def list_groups(db: Session, skip: int = 0, limit: int = 100) -> list[ServerGroup]:
    return db.query(ServerGroup).offset(skip).limit(limit).all()


def update_group(db: Session, group_id: int, data: ServerGroupUpdate) -> ServerGroup:
    group = get_group(db, group_id)
    for field, value in data.model_dump(exclude_none=True).items():
        setattr(group, field, value)
    db.commit()
    db.refresh(group)
    return group


def delete_group(db: Session, group_id: int) -> None:
    group = get_group(db, group_id)
    db.delete(group)
    db.commit()
    logger.info("Deleted ServerGroup %d", group_id)


# ── Server CRUD ───────────────────────────────────────────────────────────────

def add_server(db: Session, group_id: int, data: ServerCreate) -> Server:
    get_group(db, group_id)  # 404 guard

    # Name must be unique within the group
    existing = (
        db.query(Server)
        .filter_by(group_id=group_id, name=data.name)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Server '{data.name}' already exists in group {group_id}.",
        )

    server = Server(group_id=group_id, **data.model_dump())
    db.add(server)
    db.commit()
    db.refresh(server)
    logger.info("Added Server %d (%s → %s:%d)", server.id, server.name, server.host, server.port)
    return server


def get_server(db: Session, group_id: int, server_id: int) -> Server:
    server = db.query(Server).filter_by(id=server_id, group_id=group_id).first()
    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server {server_id} not found in group {group_id}.",
        )
    return server


def list_servers(db: Session, group_id: int, active_only: bool = False) -> list[Server]:
    get_group(db, group_id)  # 404 guard
    q = db.query(Server).filter_by(group_id=group_id)
    if active_only:
        q = q.filter_by(is_active=True)
    return q.all()


def update_server(
    db: Session, group_id: int, server_id: int, data: ServerUpdate
) -> Server:
    server = get_server(db, group_id, server_id)
    for field, value in data.model_dump(exclude_none=True).items():
        setattr(server, field, value)
    db.commit()
    db.refresh(server)
    return server


def remove_server(db: Session, group_id: int, server_id: int) -> None:
    server = get_server(db, group_id, server_id)
    db.delete(server)
    db.commit()
    logger.info("Removed Server %d from group %d", server_id, group_id)


# ── Convenience: provision configs ───────────────────────────────────────────

def provision_group_configs(db: Session, group_id: int) -> list[ForecastingConfig]:
    """
    Create a ForecastingConfig for every active server in the group that
    does not already have one.  The config inherits the group's business
    metric formula and the server's host/port.

    This is called after servers are added to a group so training can be
    triggered in bulk via POST /groups/{id}/train/.

    Returns the list of newly created configs (skips servers that already
    have a config).
    """
    group = get_group(db, group_id)
    servers = list_servers(db, group_id, active_only=True)
    created: list[ForecastingConfig] = []

    for server in servers:
        # Check if a config already exists for this server
        existing = db.query(ForecastingConfig).filter_by(server_id=server.id).first()
        if existing:
            logger.debug(
                "Server %d already has config %d — skipping provisioning.",
                server.id, existing.id,
            )
            continue

        config_name = f"{group.name}::{server.name}"
        # Name collision guard (e.g. if a legacy config already uses this name)
        if db.query(ForecastingConfig).filter_by(name=config_name).first():
            config_name = f"{group.name}::{server.name}::{server.id}"

        config = ForecastingConfig(
            name=config_name,
            server_id=server.id,
            host=server.host,
            port=server.port,
            business_metric_name=group.business_metric_name,
            business_metric_formula=group.business_metric_formula,
        )
        db.add(config)
        created.append(config)

    if created:
        db.commit()
        for c in created:
            db.refresh(c)
        logger.info(
            "Provisioned %d config(s) for group %d (%s)",
            len(created), group_id, group.name,
        )

    return created
