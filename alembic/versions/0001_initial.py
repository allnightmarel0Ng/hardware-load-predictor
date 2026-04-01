"""initial schema — all tables

Revision ID: 0001_initial
Revises:
Create Date: 2026-03-30
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── forecasting_configs ───────────────────────────────────────────────────
    op.create_table(
        "forecasting_configs",
        sa.Column("id",                     sa.Integer(),     nullable=False),
        sa.Column("name",                   sa.String(255),   nullable=False),
        sa.Column("host",                   sa.String(255),   nullable=False),
        sa.Column("port",                   sa.Integer(),     nullable=False),
        sa.Column("business_metric_name",   sa.String(255),   nullable=False),
        sa.Column("business_metric_formula",sa.String(1024),  nullable=False),
        sa.Column("created_at",             sa.DateTime(),    nullable=True),
        sa.Column("updated_at",             sa.DateTime(),    nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_forecasting_configs_id", "forecasting_configs", ["id"])

    # ── trained_models ────────────────────────────────────────────────────────
    op.create_table(
        "trained_models",
        sa.Column("id",           sa.Integer(),    nullable=False),
        sa.Column("config_id",    sa.Integer(),    nullable=False),
        sa.Column("version",      sa.Integer(),    nullable=False),
        sa.Column("algorithm",    sa.String(128),  nullable=False),
        sa.Column("status",       sa.Enum("pending","training","ready","failed",
                                         name="modelstatus"), nullable=True),
        sa.Column("parameters",   sa.JSON(),       nullable=True),
        sa.Column("metrics",      sa.JSON(),       nullable=True),
        sa.Column("artifact_path",sa.String(512),  nullable=True),
        sa.Column("lag_minutes",  sa.Integer(),    nullable=True),
        sa.Column("trained_at",   sa.DateTime(),   nullable=True),
        sa.Column("created_at",   sa.DateTime(),   nullable=True),
        sa.ForeignKeyConstraint(["config_id"], ["forecasting_configs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trained_models_id",        "trained_models", ["id"])
    op.create_index("ix_trained_models_config_id", "trained_models", ["config_id"])

    # ── forecast_results ──────────────────────────────────────────────────────
    op.create_table(
        "forecast_results",
        sa.Column("id",                    sa.Integer(), nullable=False),
        sa.Column("config_id",             sa.Integer(), nullable=False),
        sa.Column("model_id",              sa.Integer(), nullable=False),
        sa.Column("business_metric_value", sa.Float(),   nullable=False),
        sa.Column("predicted_cpu_percent", sa.Float(),   nullable=False),
        sa.Column("predicted_ram_gb",      sa.Float(),   nullable=False),
        sa.Column("predicted_network_mbps",sa.Float(),   nullable=False),
        sa.Column("actual_cpu_percent",    sa.Float(),   nullable=True),
        sa.Column("actual_ram_gb",         sa.Float(),   nullable=True),
        sa.Column("actual_network_mbps",   sa.Float(),   nullable=True),
        sa.Column("actuals_fetched_at",    sa.DateTime(),nullable=True),
        sa.Column("created_at",            sa.DateTime(),nullable=True),
        sa.ForeignKeyConstraint(["config_id"], ["forecasting_configs.id"]),
        sa.ForeignKeyConstraint(["model_id"],  ["trained_models.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_forecast_results_id",        "forecast_results", ["id"])
    op.create_index("ix_forecast_results_config_id", "forecast_results", ["config_id"])
    op.create_index("ix_forecast_results_model_id",  "forecast_results", ["model_id"])

    # ── model_evaluations ─────────────────────────────────────────────────────
    op.create_table(
        "model_evaluations",
        sa.Column("id",               sa.Integer(), nullable=False),
        sa.Column("model_id",         sa.Integer(), nullable=False),
        sa.Column("config_id",        sa.Integer(), nullable=False),
        sa.Column("n_samples",        sa.Integer(), nullable=False),
        sa.Column("mae_cpu",          sa.Float(),   nullable=True),
        sa.Column("mae_ram",          sa.Float(),   nullable=True),
        sa.Column("mae_net",          sa.Float(),   nullable=True),
        sa.Column("rmse_cpu",         sa.Float(),   nullable=True),
        sa.Column("rmse_ram",         sa.Float(),   nullable=True),
        sa.Column("rmse_net",         sa.Float(),   nullable=True),
        sa.Column("mape_overall",     sa.Float(),   nullable=True),
        sa.Column("r2_cpu",           sa.Float(),   nullable=True),
        sa.Column("r2_ram",           sa.Float(),   nullable=True),
        sa.Column("r2_net",           sa.Float(),   nullable=True),
        sa.Column("triggered_retrain",sa.Boolean(), nullable=True),
        sa.Column("psi_value",        sa.Float(),   nullable=True),
        sa.Column("psi_level",        sa.String(32),nullable=True),
        sa.Column("evaluated_at",     sa.DateTime(),nullable=True),
        sa.ForeignKeyConstraint(["config_id"], ["forecasting_configs.id"]),
        sa.ForeignKeyConstraint(["model_id"],  ["trained_models.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_evaluations_id",        "model_evaluations", ["id"])
    op.create_index("ix_model_evaluations_model_id",  "model_evaluations", ["model_id"])
    op.create_index("ix_model_evaluations_config_id", "model_evaluations", ["config_id"])

    # ── training_jobs ─────────────────────────────────────────────────────────
    op.create_table(
        "training_jobs",
        sa.Column("id",            sa.Integer(),  nullable=False),
        sa.Column("config_id",     sa.Integer(),  nullable=False),
        sa.Column("model_id",      sa.Integer(),  nullable=True),
        sa.Column("status",        sa.Enum("queued","running","done","failed",
                                          name="jobstatus"), nullable=False),
        sa.Column("lookback_days", sa.Integer(),  nullable=True),
        sa.Column("error_message", sa.Text(),     nullable=True),
        sa.Column("created_at",    sa.DateTime(), nullable=True),
        sa.Column("started_at",    sa.DateTime(), nullable=True),
        sa.Column("finished_at",   sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["config_id"], ["forecasting_configs.id"]),
        sa.ForeignKeyConstraint(["model_id"],  ["trained_models.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_jobs_id",        "training_jobs", ["id"])
    op.create_index("ix_training_jobs_config_id", "training_jobs", ["config_id"])


def downgrade() -> None:
    op.drop_table("training_jobs")
    op.drop_table("model_evaluations")
    op.drop_table("forecast_results")
    op.drop_table("trained_models")
    op.drop_table("forecasting_configs")
    op.execute("DROP TYPE IF EXISTS jobstatus")
    op.execute("DROP TYPE IF EXISTS modelstatus")
