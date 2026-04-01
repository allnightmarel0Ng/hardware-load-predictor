"""add server_groups and servers tables

Revision ID: 0002_multi_server
Revises: 0001_initial
Create Date: 2026-03-30
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_multi_server"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── server_groups ─────────────────────────────────────────────────────────
    op.create_table(
        "server_groups",
        sa.Column("id",                       sa.Integer(),    nullable=False),
        sa.Column("name",                     sa.String(255),  nullable=False),
        sa.Column("description",              sa.String(1024), nullable=True),
        sa.Column("business_metric_name",     sa.String(255),  nullable=False),
        sa.Column("business_metric_formula",  sa.String(1024), nullable=False),
        sa.Column("metrics_host",             sa.String(255),  nullable=False),
        sa.Column("metrics_port",             sa.Integer(),    nullable=False),
        sa.Column("created_at",               sa.DateTime(),   nullable=True),
        sa.Column("updated_at",               sa.DateTime(),   nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_server_groups_id", "server_groups", ["id"])

    # ── servers ───────────────────────────────────────────────────────────────
    op.create_table(
        "servers",
        sa.Column("id",         sa.Integer(),    nullable=False),
        sa.Column("group_id",   sa.Integer(),    nullable=False),
        sa.Column("name",       sa.String(255),  nullable=False),
        sa.Column("host",       sa.String(255),  nullable=False),
        sa.Column("port",       sa.Integer(),    nullable=False),
        sa.Column("tags",       sa.JSON(),       nullable=True),
        sa.Column("is_active",  sa.Boolean(),    nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(),   nullable=True),
        sa.ForeignKeyConstraint(["group_id"], ["server_groups.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_servers_id",       "servers", ["id"])
    op.create_index("ix_servers_group_id", "servers", ["group_id"])

    # ── add server_id FK to forecasting_configs ───────────────────────────────
    op.add_column(
        "forecasting_configs",
        sa.Column("server_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_forecasting_configs_server_id",
        "forecasting_configs", "servers",
        ["server_id"], ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_forecasting_configs_server_id",
        "forecasting_configs", ["server_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_forecasting_configs_server_id", "forecasting_configs")
    op.drop_constraint("fk_forecasting_configs_server_id", "forecasting_configs", type_="foreignkey")
    op.drop_column("forecasting_configs", "server_id")
    op.drop_table("servers")
    op.drop_table("server_groups")
