"""Create the local sample database before the app opens read-only queries."""

from pathlib import Path
from threading import Lock

from db.queries import DB, ROOT
from scripts.build_db import build_database

_BUILD_LOCK = Lock()


def ensure_sample_database(db_path=DB, sample_path=ROOT / "data/sample"):
    """Reuse the CLI builder on first launch; leave existing databases alone.

    The lock serializes startup requests from Streamlit sessions in this worker.
    The builder publishes atomically, so readers never see a partial database.
    No downloads or full-dataset processing occur during application startup.
    """
    db_path = Path(db_path)
    with _BUILD_LOCK:
        if not db_path.is_file():
            build_database(sample_path, db_path)
    return db_path
