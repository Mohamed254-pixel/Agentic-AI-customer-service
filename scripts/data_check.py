import duckdb
from pathlib import Path

con = duckdb.connect()

data_path = Path("data/raw/instacart")

print("INSTACART DATASET VALIDATION")
print("-" * 40)

for file in sorted(data_path.glob("*.csv")):
    query = """
        SELECT COUNT(*)
        FROM read_csv_auto(?)
    """

    count = con.execute(query, [str(file)]).fetchone()[0]
    print(f"{file.name}: {count:,} rows")

con.close()