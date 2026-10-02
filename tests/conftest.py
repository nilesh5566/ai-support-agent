import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

CLIENT = {"X-API-Key": "dev-client-key"}
ADMIN = {"X-API-Key": "dev-admin-key"}


def make_settings(tmp_path, **overrides) -> Settings:
    # CI runs the suite twice: SQLite/in-process, and Postgres+Redis via TEST_DATABASE_URL/TEST_REDIS_URL.
    db_url = os.getenv("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/test.db"
    base = dict(database_url=db_url, redis_url=os.getenv("TEST_REDIS_URL") or None,
                log_level="WARNING", rate_limit_per_minute=1000, seed_demo_data=True)
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _reset_external_db(url: str) -> None:
    if url.startswith("sqlite"):
        return
    from app.db import Base, Database

    db = Database(url)
    db.create_all()
    Base.metadata.drop_all(db.engine)
    db.engine.dispose()


@pytest.fixture
def app_factory(tmp_path):
    clients = []

    def _make(**overrides):
        settings = make_settings(tmp_path, **overrides)
        _reset_external_db(settings.database_url)
        if settings.redis_url:
            import redis

            redis.Redis.from_url(settings.redis_url).flushdb()
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(app_factory):
    return app_factory()


def chat(client, message, email=None, headers=CLIENT, **extra):
    r = client.post("/v1/chat", headers=headers, json={"message": message, "customer_email": email, **extra})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def unique():
    return uuid.uuid4().hex[:8]
