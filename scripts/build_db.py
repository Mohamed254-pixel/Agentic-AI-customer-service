"""Build a DuckDB database from local CSVs, via CLI or application startup."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "aisles", "departments", "products", "orders",
    "order_products__prior", "order_products__train",
)


def build_database(data_path, db_path):
    """Build all six tables before publishing the database at its final path.

    Only local CSVs are read. If input or loading fails, an existing database
    is left untouched and no partial database is published.
    """
    data_path, db_path = Path(data_path), Path(db_path)
    for table in TABLES:
        csv_file = data_path / f"{table}.csv"
        if not csv_file.is_file():
            raise FileNotFoundError(csv_file)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="instacart-build-", dir=db_path.parent) as folder:
        temporary = Path(folder) / "instacart.duckdb"
        with duckdb.connect(str(temporary)) as con:
            for table in TABLES:
                print(f"Loading {table}...")
                con.execute(
                    f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?)",
                    [str(data_path / f"{table}.csv")],
                )
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"{table}: {count:,} rows")
        # Closing the connection flushes DuckDB before its atomic replacement.
        os.replace(temporary, db_path)
    print(f"\nDatabase ready: {db_path}")
    return db_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sample", "full"], default="sample")
    args = parser.parse_args()
    data = ROOT / "data/sample" if args.dataset == "sample" else ROOT / "data/raw/instacart"
    return build_database(data, ROOT / "data/processed" / f"instacart_{args.dataset}.duckdb")


if __name__ == "__main__":
    main()
