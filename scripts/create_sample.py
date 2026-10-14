import duckdb
import shutil
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/instacart"
SAMPLE = ROOT / "data/sample"

SAMPLE.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

# 1. Select 10,000 orders
# Group by customer to preserve order histories
con.execute("""
    CREATE TEMP TABLE sample_orders AS
    SELECT *
    FROM read_csv_auto(?)
    WHERE eval_set IN ('prior', 'train')
    ORDER BY hash(user_id), order_number, order_id
    LIMIT 10000
""", [str(RAW / "orders.csv")])


def save_table(table, filename):
    output = str(SAMPLE / filename).replace("'", "''")
    con.execute(
        f"COPY {table} TO '{output}' "
        "(FORMAT CSV, HEADER TRUE)"
    )


save_table("sample_orders", "orders.csv")

# 2. Retrieve purchased products for sampled orders
for split in ["prior", "train"]:
    filename = f"order_products__{split}.csv"

    con.execute(f"""
        CREATE TEMP TABLE sample_{split} AS
        SELECT items.*
        FROM read_csv_auto(?) AS items
        JOIN sample_orders AS orders
            ON items.order_id = orders.order_id
        WHERE orders.eval_set = ?
    """, [str(RAW / filename), split])

    save_table(f"sample_{split}", filename)

# 3. Keep only products used by sampled orders
con.execute("""
    CREATE TEMP TABLE sample_products AS
    SELECT *
    FROM read_csv_auto(?)
    WHERE product_id IN (
        SELECT product_id FROM sample_prior
        UNION
        SELECT product_id FROM sample_train
    )
""", [str(RAW / "products.csv")])

save_table("sample_products", "products.csv")

# 4. Copy small reference tables
for filename in ["aisles.csv", "departments.csv"]:
    shutil.copy2(RAW / filename, SAMPLE / filename)

# 5. Print results
print("\nINSTACART SAMPLE CREATED")
print("-" * 40)

for table in [
    "sample_orders",
    "sample_prior",
    "sample_train",
    "sample_products"
]:
    count = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print(f"{table}: {count:,} rows")

customers = con.execute("""
    SELECT COUNT(DISTINCT user_id)
    FROM sample_orders
""").fetchone()[0]

print(f"Unique customers: {customers:,}")

con.close()
print("\nSample dataset created successfully!")