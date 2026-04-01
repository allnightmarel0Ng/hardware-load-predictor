"""
Alembic environment — uses the app's SQLAlchemy Base metadata so that
`alembic revision --autogenerate` detects all model changes automatically.

Usage:
    # Generate a new migration after changing app/models/db_models.py:
    alembic revision --autogenerate -m "describe your change"

    # Apply all pending migrations:
    alembic upgrade head

    # Roll back one migration:
    alembic downgrade -1
"""
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import the app's metadata so autogenerate can diff against it
from app.core.database import Base  # noqa: F401
import app.models.db_models  # noqa: F401 — registers all ORM models with Base.metadata

# Alembic Config object (provides access to alembic.ini)
config = context.config

# Override sqlalchemy.url from DATABASE_URL env var if present
db_url = os.environ.get("DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL without a live connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (requires a live DB connection)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
