import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

os.environ["DATABASE_URL"]        = "sqlite:///./test.db"
os.environ["MODEL_STORAGE_PATH"]  = "/tmp/test_models"
os.environ["USE_PROMETHEUS_STUB"] = "true"

from app.core.database import Base, get_db
from app.core.config import settings
from app.main import app

settings.use_prometheus_stub = True
settings.model_storage_path  = "/tmp/test_models"

TEST_DB_URL = "sqlite:///./test.db"

engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
)

@event.listens_for(engine, "connect")
def set_wal_mode(dbapi_conn, _):
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA busy_timeout=10000")

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def create_tables():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def clean_tables():
    yield
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture()
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def client(db, monkeypatch):
    from app.modules import job_runner

    def override_get_db():
        try:
            yield db
        finally:
            pass

    monkeypatch.setattr(job_runner, "_executor", job_runner._executor or __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(job_runner._executor, "submit", lambda *a, **kw: None)

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
