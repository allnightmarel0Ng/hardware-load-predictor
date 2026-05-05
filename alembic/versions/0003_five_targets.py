from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_five_targets"
down_revision: Union[str, None] = "0002_multi_server"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
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
    for col in [
        "predicted_ram_percent", "predicted_disk_io_percent",
        "lower_ram_percent", "lower_disk_io_percent",
        "upper_ram_percent", "upper_disk_io_percent",
        "actual_ram_percent", "actual_disk_io_percent",
    ]:
        op.drop_column("forecast_results", col)

    for col in [
        "predicted_ram_percent", "predicted_disk_io_percent",
        "lower_ram_percent", "lower_disk_io_percent",
        "upper_ram_percent", "upper_disk_io_percent",
    ]:
        op.drop_column("forecast_horizon_results", col)

    for col in [
        "mae_ram_gb", "mae_ram_pct", "mae_disk",
        "rmse_ram_gb", "rmse_ram_pct", "rmse_disk",
        "r2_ram_gb", "r2_ram_pct", "r2_disk",
    ]:
        op.drop_column("model_evaluations", col)
