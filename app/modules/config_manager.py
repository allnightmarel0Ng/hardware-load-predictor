"""
Module 1 — Configuration Management
Handles CRUD for ForecastingConfig records (engineer-defined bindings
between a business metric and a monitored host).
"""
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.models.db_models import ForecastingConfig
from app.schemas.schemas import ForecastingConfigCreate, ForecastingConfigUpdate


def create_config(db: Session, data: ForecastingConfigCreate) -> ForecastingConfig:
    existing = db.query(ForecastingConfig).filter_by(name=data.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Config with name '{data.name}' already exists.",
        )
    config = ForecastingConfig(**data.model_dump())
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


def get_config(db: Session, config_id: int) -> ForecastingConfig:
    config = db.get(ForecastingConfig, config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Config {config_id} not found.",
        )
    return config


def list_configs(db: Session, skip: int = 0, limit: int = 100) -> list[ForecastingConfig]:
    return db.query(ForecastingConfig).offset(skip).limit(limit).all()


def update_config(
    db: Session, config_id: int, data: ForecastingConfigUpdate
) -> ForecastingConfig:
    config = get_config(db, config_id)
    for field, value in data.model_dump(exclude_none=True).items():
        setattr(config, field, value)
    db.commit()
    db.refresh(config)
    return config


def delete_config(db: Session, config_id: int) -> None:
    config = get_config(db, config_id)
    db.delete(config)
    db.commit()
