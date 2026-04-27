"""add instance_label to forecasting_configs

Revision ID: 0004_instance_label
Revises: 0003_five_targets
Create Date: 2026-04-13
"""
from alembic import op
import sqlalchemy as sa

revision = '0004_instance_label'
down_revision = '0003_five_targets'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'forecasting_configs',
        sa.Column('instance_label', sa.String(255), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('forecasting_configs', 'instance_label')
