
import argparse
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["sample", "full"],
    default="sample"
)
args = parser.parse_args()

# Select the dataset
DATA = (
    ROOT / "data/sample"
    if args.dataset == "sample"
    else ROOT / "data/raw/instacart"
)

DB = ROOT / "data/processed" / f"instacart_{args.dataset}.duckdb"
DB.parent.mkdir(parents=True, exist_ok=True)

tables = [
    "aisles",
    "departments",
    "products",
    "orders",
    "order_products__prior",
    "order_products__train"
]

con = duckdb.connect(str(DB))

for table in tables:
    csv_file = DATA / f"{table}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    print(f"Loading {table}...")

    con.execute(
        f"CREATE OR REPLACE TABLE {table} AS "
        "SELECT * FROM read_csv_auto(?)",
        [str(csv_file)]
    )

    count = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print(f"{table}: {count:,} rows")

con.close()

print(f"\nDatabase ready: {DB}")
