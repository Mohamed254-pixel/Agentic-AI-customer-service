"""Separate, replaceable demo pricing store. Original sample tables stay read-only."""
from hashlib import sha256
from pathlib import Path
from threading import RLock

import duckdb

from db.queries import DB

_LOCK = RLock()
LABEL = 'Simulated prices · USD · Per catalog item · Tax and delivery excluded'


def _demo_cents(product_id):
    # Versioned SHA-256 is stable across restarts, machines and Python versions.
    digest = sha256(f'cartside-demo-v1:{product_id}'.encode()).digest()
    return 99 + int.from_bytes(digest[:8], 'big') % 1901


def _paths(source_db, pricing_db):
    source = Path(source_db)
    target = Path(pricing_db) if pricing_db else source.with_name(source.stem + '_pricing.duckdb')
    if source.resolve() == target.resolve():
        raise ValueError('Pricing must use a separate database')
    return source, target


def ensure_pricing_database(source_db=DB, pricing_db=None):
    """Create/backfill prices offline, without altering source CSVs or tables.

    One row per product, integer USD cents. Existing stored prices are never
    overwritten. DuckDB transactions roll back incomplete initialization.
    The lock serializes sessions inside the single Streamlit server process.
    """
    source, target = _paths(source_db, pricing_db)
    with _LOCK:
        with duckdb.connect(str(source), read_only=True) as con:
            ids = [row[0] for row in con.execute('SELECT product_id FROM products').fetchall()]
        target.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(target)) as con:
            con.execute('BEGIN TRANSACTION')
            try:
                con.execute('''CREATE TABLE IF NOT EXISTS product_prices (
                    product_id BIGINT PRIMARY KEY,
                    price_cents INTEGER NOT NULL CHECK (price_cents > 0),
                    currency VARCHAR NOT NULL CHECK (currency = 'USD'),
                    source VARCHAR NOT NULL CHECK (source = 'simulated-v1'))''')
                existing = {row[0] for row in con.execute('SELECT product_id FROM product_prices').fetchall()}
                missing = [(pid, _demo_cents(pid), 'USD', 'simulated-v1') for pid in ids if pid not in existing]
                if missing:
                    # Register an in-memory Arrow-free relation for one bulk insert.
                    con.execute('INSERT INTO product_prices SELECT unnest(?::BIGINT[]), unnest(?::INTEGER[]), \'USD\', \'simulated-v1\'',
                                [[row[0] for row in missing], [row[1] for row in missing]])
                con.execute('COMMIT')
            except Exception:
                con.execute('ROLLBACK')
                raise
    return target


def get_prices(product_ids, source_db=DB, pricing_db=None):
    """Provider boundary: only verified catalog IDs can receive a stored price."""
    ids = list(dict.fromkeys(product_ids))
    if any(type(pid) is not int or pid <= 0 for pid in ids):
        raise ValueError('Product IDs must be positive integers')
    if not ids:
        return {}
    with _LOCK:
        target = ensure_pricing_database(source_db, pricing_db)
        with duckdb.connect(str(source_db), read_only=True) as con:
            known = {r[0] for r in con.execute('SELECT product_id FROM products WHERE product_id IN (SELECT unnest(?::BIGINT[]))', [ids]).fetchall()}
        if known != set(ids):
            raise ValueError('Cannot price an unknown catalog product')
        with duckdb.connect(str(target)) as con:
            return dict(con.execute('SELECT product_id, price_cents FROM product_prices WHERE product_id IN (SELECT unnest(?::BIGINT[]))', [ids]).fetchall())


def quote_products(product_ids, source_db=DB, pricing_db=None):
    """One unit per selected product. All arithmetic uses stored integer cents."""
    prices = get_prices(product_ids, source_db, pricing_db)
    return {'prices': prices, 'total_cents': sum(prices.values()), 'currency': 'USD', 'simulated': True}


def money(cents):
    return f'${cents // 100:,}.{cents % 100:02d}'


def recommendation_lines(products, source_db=DB):
    """Render stored prices after model selection; the model never invents money."""
    prices = get_prices([p['product_id'] for p in products], source_db)
    return '\n'.join(f"- {p['product_name']} — {money(prices[p['product_id']])}" for p in products) + '\n\n' + LABEL
