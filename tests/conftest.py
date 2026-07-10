"""Shared fixtures. Integration tests auto-skip when the compose stack is down."""

import psycopg
import pytest

from common.settings import get_settings


def _db_available() -> bool:
    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


@pytest.fixture(scope="session")
def db_conn():
    if not _db_available():
        pytest.skip("Postgres is not reachable (run `make up` first)")
    conn = psycopg.connect(get_settings().database_url)
    yield conn
    conn.close()


def pytest_collection_modifyitems(config, items):
    """Give every test using the db_conn fixture the integration marker."""
    for item in items:
        if "db_conn" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.integration)
