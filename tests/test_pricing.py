"""Stable independent pricing, exact totals, and offline deployment behavior."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest

from db.pricing import ensure_pricing_database, get_prices, quote_products, money


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / 'source.duckdb'
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE products AS SELECT * FROM (VALUES (1, 'Apple'), (2, 'Yogurt')) p(product_id, product_name)")
    return path


def test_stable_prices_and_source_untouched(catalog, tmp_path):
    before = catalog.read_bytes()
    prices = get_prices([1, 2], catalog)
    assert prices == get_prices([1, 2], catalog)
    other = tmp_path / 'other_pricing.duckdb'
    assert prices == get_prices([1, 2], catalog, other)
    assert all(type(cents) is int and 99 <= cents <= 1999 for cents in prices.values())
    assert catalog.read_bytes() == before
    with duckdb.connect(str(catalog), read_only=True) as con:
        assert con.execute('SHOW TABLES').fetchall() == [('products',)]


def test_backfill_preserves_existing_stored_prices(catalog):
    target = ensure_pricing_database(catalog)
    with duckdb.connect(str(target)) as con:
        con.execute('UPDATE product_prices SET price_cents=123 WHERE product_id=1')
    with duckdb.connect(str(catalog)) as con:
        con.execute("INSERT INTO products VALUES (3, 'Banana')")
    prices = get_prices([1, 2, 3], catalog)
    assert prices[1] == 123
    assert len(prices) == 3


def test_quote_uses_database_cents_not_generation(catalog):
    target = ensure_pricing_database(catalog)
    with duckdb.connect(str(target)) as con:
        con.execute('UPDATE product_prices SET price_cents = CASE product_id WHEN 1 THEN 110 ELSE 225 END')
    quote = quote_products([1, 2, 1], catalog)
    assert quote['total_cents'] == 335  # Shopping lists contain one unit per unique product.
    assert quote['simulated'] and quote['currency'] == 'USD'
    assert quote_products([], catalog)['total_cents'] == 0
    assert money(335) == '$3.35'
    assert money(100000) == '$1,000.00'


@pytest.mark.parametrize('ids', [[999], [True], ['1'], [-1]])
def test_invalid_or_unknown_products_cannot_get_a_price(catalog, ids):
    with pytest.raises(ValueError):
        get_prices(ids, catalog)


def test_separate_store_required_and_missing_source_not_created(catalog, tmp_path):
    with pytest.raises(ValueError):
        ensure_pricing_database(catalog, catalog)
    missing = tmp_path / 'missing.duckdb'
    with pytest.raises(duckdb.Error):
        ensure_pricing_database(missing)
    assert not missing.exists()


def test_concurrent_first_use(catalog):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: get_prices([1, 2], catalog), range(4)))
    assert all(prices == results[0] for prices in results)
    with duckdb.connect(str(ensure_pricing_database(catalog))) as con:
        assert con.execute('SELECT count(*) FROM product_prices').fetchone()[0] == 2


def test_recommendations_and_price_followup_use_stored_values(monkeypatch):
    from agents.product_agent import product_turn
    from db.queries import DB
    first = product_turn('Recommend bananas', use_ai=False)
    products = first['recommendations']
    prices = get_prices([p['product_id'] for p in products], DB)
    assert 'Simulated prices' in first['text']
    for product in products:
        assert f"{product['product_name']} — {money(prices[product['product_id']])}" in first['text']
    followup = product_turn('How much do those cost?', first, use_ai=False)
    assert 'Simulated prices' in followup['text']
    assert followup['suggested_product_ids'] == first['suggested_product_ids']


def test_shopping_ui_displays_database_total(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from db.queries import DB
    monkeypatch.setenv('DEMO_MODE', '1')
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'main.py').run(timeout=30)
    app.radio[0].set_value('Shopping list').run(timeout=30)
    with duckdb.connect(str(DB), read_only=True) as con:
        ids = [r[0] for r in con.execute('SELECT product_id FROM products ORDER BY product_name LIMIT 2').fetchall()]
    app.multiselect[0].set_value(ids).run(timeout=30)
    assert not app.exception
    assert any('Simulated prices' in caption.value for caption in app.caption)
    expected = money(quote_products(ids)['total_cents'])
    assert any('Simulated total' in item.value and expected in item.value for item in app.markdown)
    before = expected
    app.checkbox[0].check().run(timeout=30)
    assert any('Simulated total' in item.value and before in item.value for item in app.markdown)


def test_named_price_request_routes_to_catalog_without_recommending(monkeypatch):
    from agents.router import build_router
    from langchain_core.messages import HumanMessage
    monkeypatch.setenv('DEMO_MODE', '1')
    graph = build_router()
    result = graph.invoke({'messages': [HumanMessage(content='What is the price of bananas?')], 'user_id': 5117},
                          config={'configurable': {'thread_id': 'named-price'}})
    assert result['route'] == 'PRODUCT_RECOMMENDATION'
    assert 'Bananas — $' in result['messages'][-1].content
    assert 'Simulated prices' in result['messages'][-1].content
    assert result['suggested_product_ids'] == []
