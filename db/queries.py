
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/processed/instacart_sample.duckdb"


def get_customer_orders(user_id):
    """Retrieve a customer's historical orders."""

    with duckdb.connect(str(DB), read_only=True) as con:
        result = con.execute("""
            SELECT
                order_id,
                user_id,
                order_number,
                eval_set,
                days_since_prior_order
            FROM orders
            WHERE user_id = ?
            ORDER BY order_number DESC
        """, [user_id])

        columns = [col[0] for col in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def get_order_products(order_id):
    """Retrieve products belonging to a historical order."""

    with duckdb.connect(str(DB), read_only=True) as con:
        order = con.execute(
            "SELECT eval_set FROM orders WHERE order_id = ?",
            [order_id]
        ).fetchone()

        if order is None:
            return []

        tables = {
            "prior": "order_products__prior",
            "train": "order_products__train"
        }

        table = tables.get(order[0])

        if table is None:
            return []

        result = con.execute(f"""
            SELECT
                p.product_id,
                p.product_name,
                i.add_to_cart_order,
                i.reordered
            FROM {table} AS i
            JOIN products AS p
                ON i.product_id = p.product_id
            WHERE i.order_id = ?
            ORDER BY i.add_to_cart_order
        """, [order_id])

        columns = [col[0] for col in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]
