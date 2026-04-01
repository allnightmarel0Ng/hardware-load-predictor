"""add ram_percent and disk_io_percent targets

Adds predicted_ram_percent, predicted_disk_io_percent (+ intervals + actuals)
to forecast_results and forecast_horizon_results, and adds the corresponding
mae/rmse/r2 columns to model_evaluations.

Also adds the new PromQL settings (no schema change — these are config values).

Revision ID: 0003_five_targets
Revises: 0002_multi_server
Create Date: 2026-04-01
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_five_targets"
down_revision: Union[str, None] = "0002_multi_server"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── forecast_results ──────────────────────────────────────────────────────
    for col in [
        "predicted_ram_percent",
        "predicted_disk_io_percent",
        "lower_ram_percent",
        "lower_disk_io_percent",
        "upper_ram_percent",
        "upper_disk_io_percent",
        "actual_ram_percent",
        "actual_disk_io_percent",
    ]:
        nullable = col not in ("predicted_ram_percent", "predicted_disk_io_percent")
        op.add_column(
            "forecast_results",
            sa.Column(col, sa.Float(), nullable=nullable, server_default="0.0"),
        )

    # ── forecast_horizon_results ──────────────────────────────────────────────
    for col in [
        "predicted_ram_percent",
        "predicted_disk_io_percent",
        "lower_ram_percent",
        "lower_disk_io_percent",
        "upper_ram_percent",
        "upper_disk_io_percent",
    ]:
        nullable = col not in ("predicted_ram_percent", "predicted_disk_io_percent")
        op.add_column(
            "forecast_horizon_results",
            sa.Column(col, sa.Float(), nullable=nullable, server_default="0.0"),
        )

    # ── model_evaluations ─────────────────────────────────────────────────────
    # rename old ram/net columns → ram_gb/net, add new ram_pct/disk columns
    # (SQLite doesn't support RENAME COLUMN in older versions; use ADD for new ones,
    #  keep old names as aliases in application code during migration window)
    for col in [
        "mae_ram_gb", "mae_ram_pct", "mae_disk",
        "rmse_ram_gb", "rmse_ram_pct", "rmse_disk",
        "r2_ram_gb", "r2_ram_pct", "r2_disk",
    ]:
        op.add_column(
            "model_evaluations",
            sa.Column(col, sa.Float(), nullable=True),
        )


def downgrade() -> None:
    # forecast_results
    for col in [
        "predicted_ram_percent", "predicted_disk_io_percent",
        "lower_ram_percent", "lower_disk_io_percent",
        "upper_ram_percent", "upper_disk_io_percent",
        "actual_ram_percent", "actual_disk_io_percent",
    ]:
        op.drop_column("forecast_results", col)

    # forecast_horizon_results
    for col in [
        "predicted_ram_percent", "predicted_disk_io_percent",
        "lower_ram_percent", "lower_disk_io_percent",
        "upper_ram_percent", "upper_disk_io_percent",
    ]:
        op.drop_column("forecast_horizon_results", col)

    # model_evaluations
    for col in [
        "mae_ram_gb", "mae_ram_pct", "mae_disk",
        "rmse_ram_gb", "rmse_ram_pct", "rmse_disk",
        "r2_ram_gb", "r2_ram_pct", "r2_disk",
    ]:
        op.drop_column("model_evaluations", col)
