"""Build a DuckDB database from local CSVs, via CLI or application startup."""

import argparse
import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

ROOT = Path(__file__).resolve().parents[1]
ITEM_COLUMNS = {
    "order_id": "INTEGER", "product_id": "INTEGER",
    "add_to_cart_order": "INTEGER", "reordered": "INTEGER",
}
SCHEMAS = {
    "aisles": {"aisle_id": "INTEGER", "aisle": "VARCHAR"},
    "departments": {"department_id": "INTEGER", "department": "VARCHAR"},
    "products": {"product_id": "INTEGER", "product_name": "VARCHAR",
                 "aisle_id": "INTEGER", "department_id": "INTEGER"},
    "orders": {"order_id": "INTEGER", "user_id": "INTEGER", "eval_set": "VARCHAR",
               "order_number": "INTEGER", "order_dow": "INTEGER",
               "order_hour_of_day": "INTEGER", "days_since_prior_order": "DOUBLE"},
    "order_products__prior": ITEM_COLUMNS,
    "order_products__train": ITEM_COLUMNS,
}
TABLES = tuple(SCHEMAS)
KEYS = {
    "aisles": "aisle_id", "departments": "department_id", "products": "product_id",
    "orders": "order_id", "order_products__prior": "order_id, product_id",
    "order_products__train": "order_id, product_id",
}


def validate_database(con):
    """Validate batch integrity before publishing, without materializing rows in Python."""
    checks = {
        "duplicate purchases for one customer":
            "SELECT 1 FROM orders GROUP BY user_id, order_number HAVING COUNT(*) > 1",
        "invalid order fields": """
            SELECT 1 FROM orders WHERE user_id <= 0 OR order_id <= 0 OR order_number <= 0
                OR eval_set NOT IN ('prior', 'train', 'test') OR order_dow NOT BETWEEN 0 AND 6
                OR order_hour_of_day NOT BETWEEN 0 AND 23
                OR days_since_prior_order NOT BETWEEN 0 AND 30
                OR (days_since_prior_order IS NULL AND order_number > 1)
        """,
        "invalid product references": """
            SELECT 1 FROM products p LEFT JOIN aisles a USING (aisle_id)
                LEFT JOIN departments d USING (department_id)
            WHERE a.aisle_id IS NULL OR d.department_id IS NULL OR trim(p.product_name) = ''
        """,
    }
    for table, columns in SCHEMAS.items():
        required = (f"{column} IS NULL" for column in columns if column != "days_since_prior_order")
        checks[f"missing required values in {table}"] = f"SELECT 1 FROM {table} WHERE " + " OR ".join(required)
        checks[f"duplicate keys in {table}"] = f"SELECT 1 FROM {table} GROUP BY {KEYS[table]} HAVING COUNT(*) > 1"
    for split in ("prior", "train"):
        checks[f"invalid {split} basket rows"] = f"""
            SELECT 1 FROM order_products__{split} i LEFT JOIN orders o USING (order_id)
                LEFT JOIN products p USING (product_id)
            WHERE o.order_id IS NULL OR p.product_id IS NULL OR o.eval_set <> '{split}'
                OR i.add_to_cart_order <= 0 OR i.reordered NOT IN (0, 1)
        """
        checks[f"duplicate {split} basket positions"] = f"""
            SELECT 1 FROM order_products__{split} GROUP BY order_id, add_to_cart_order HAVING COUNT(*) > 1
        """
    for label, query in checks.items():
        if con.execute(query + " LIMIT 1").fetchone():
            raise ValueError(f"Invalid Instacart data: {label}")


def build_database(data_path, db_path):
    """Build all six tables before publishing the database at its final path.

    Only local CSVs are read. If input or loading fails, an existing database
    is left untouched and no partial database is published.
    """
    data_path, db_path = Path(data_path), Path(db_path)
    for table, columns in SCHEMAS.items():
        csv_file = data_path / f"{table}.csv"
        if not csv_file.is_file():
            raise FileNotFoundError(csv_file)
        with csv_file.open(newline="", encoding="utf-8") as source:
            if next(csv.reader(source), []) != list(columns):
                raise ValueError(f"Unexpected CSV header in {csv_file.name}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="instacart-build-", dir=db_path.parent) as folder:
        temporary = Path(folder) / "instacart.duckdb"
        with duckdb.connect(str(temporary), config={"threads": 2}) as con:
            for table, columns in SCHEMAS.items():
                print(f"Loading {table}...")
                con.execute(
                    f"CREATE TABLE {table} AS SELECT * FROM read_csv(?, columns = ?, header = true)",
                    [str(data_path / f"{table}.csv"), columns],
                )
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"{table}: {count:,} rows")
            validate_database(con)
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
