"""Format verified recommendation replies for the retail workspace."""

import re
from html import escape

import duckdb

from db.pricing import LABEL

PRICE = re.compile(r"\$[\d,]+\.\d{2}")


def recommendation_display(result, text, database):
    """Use structured product rows only when they match the current reply.

    Checkpoint state can retain the previous recommendation list after a failed
    turn. Matching the current reply prevents displaying those stale products.
    """
    products = result.get("recommendations") if result.get("route") == "PRODUCT_RECOMMENDATION" else None
    if not products:
        return text, []

    lines = text.splitlines()
    entries = [(index, line.removeprefix("- ")) for index, line in enumerate(lines) if line.startswith("- ")]
    if len(entries) != len(products):
        return text, []

    rows = []
    for (_, entry), product in zip(entries, products):
        name, separator, price = entry.rpartition(" — ")
        if not separator or name != product["product_name"] or not PRICE.fullmatch(price):
            return text, []
        rows.append({"name": name, "price": price, "id": product["product_id"]})

    try:
        with duckdb.connect(str(database), read_only=True) as con:
            categories = {
                (pid, name): department
                for pid, name, department in con.execute("""
                    SELECT p.product_id, p.product_name, d.department
                    FROM products p LEFT JOIN departments d USING (department_id)
                    WHERE p.product_id IN (SELECT unnest(?::INTEGER[]))
                """, [[row["id"] for row in rows]]).fetchall()
            }
    except (OSError, duckdb.Error):
        categories = {}

    for row in rows:
        row["category"] = categories.get((row["id"], row["name"]))
    start, end = entries[0][0], entries[-1][0]
    above = " ".join(line for line in lines[:start] if line.strip())
    below = " ".join(line for line in lines[end + 1:] if line.strip() and line != LABEL)
    summary = " ".join(part for part in (above, below) if part) + "\n\n" + LABEL
    return summary, rows


def recommendation_rows_html(rows):
    items = []
    for row in rows:
        category = f'<span class="cartside-product-category">{escape(row["category"]).capitalize()}</span>' if row["category"] else ""
        items.append(
            f'<li><div><strong>{escape(row["name"])}</strong>{category}</div>'
            f'<span class="cartside-price">{escape(row["price"])}</span></li>'
        )
    return '<ul class="cartside-products" aria-label="Recommended groceries">' + "".join(items) + "</ul>"
