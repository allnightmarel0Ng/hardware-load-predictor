"""add quality_metric_overrides to forecasting_configs

Revision ID: 0005_quality_metric_overrides
Revises: 0004_instance_label
Create Date: 2026-04-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '0005_quality_metric_overrides'
down_revision = '0004_instance_label'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'forecasting_configs',
        sa.Column('quality_metric_overrides', JSONB, nullable=True)
    )


def downgrade() -> None:
    op.drop_column('forecasting_configs', 'quality_metric_overrides')
