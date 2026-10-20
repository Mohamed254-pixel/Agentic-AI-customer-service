"""Build a complete sample database on first launch, including hosted launches."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import duckdb

from db.queries import DB, ROOT

_BUILD_LOCK = Lock()
TABLES = ("aisles", "departments", "products", "orders", "order_products__prior", "order_products__train")


def ensure_sample_database(db_path=DB, sample_path=ROOT / "data/sample"):
    """Create atomically so concurrent sessions never see a half-built database."""
    db_path, sample_path = Path(db_path), Path(sample_path)
    with _BUILD_LOCK:
        if db_path.is_file():
            return db_path
        for table in TABLES:
            if not (sample_path / f"{table}.csv").is_file():
                raise FileNotFoundError(f"Missing sample table: {table}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=db_path.parent) as folder:
            temporary = Path(folder) / "sample.duckdb"
            with duckdb.connect(str(temporary)) as con:
                for table in TABLES:
                    con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?)", [str(sample_path / f"{table}.csv")])
            os.replace(temporary, db_path)
        return db_path
