import duckdb

con = duckdb.connect()

path = "data/raw/instacart"

# Load CSVs as queryable views
con.execute(f"""
    CREATE VIEW orders AS
    SELECT * FROM read_csv_auto('{path}/orders.csv')
""")

con.execute(f"""
    CREATE VIEW products AS
    SELECT * FROM read_csv_auto('{path}/products.csv')
""")

# Check orders
print("ORDERS DATA QUALITY")

result = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT order_id) AS unique_orders,
        COUNT(*) FILTER (
            WHERE user_id IS NULL
        ) AS missing_customers,
        COUNT(*) FILTER (
            WHERE days_since_prior_order IS NULL
        ) AS missing_prior_days
    FROM orders
""").fetchdf()

print(result.to_string(index=False))

# Check products
print("\nPRODUCTS DATA QUALITY")

result = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT product_id) AS unique_products,
        COUNT(*) FILTER (
            WHERE product_name IS NULL
        ) AS missing_names
    FROM products
""").fetchdf()

print(result.to_string(index=False))

con.close()