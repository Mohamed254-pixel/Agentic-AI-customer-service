"""Regression coverage for the grocery workspace presentation."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from db.queries import DB
from ui.recommendations import recommendation_display

ROOT = Path(__file__).resolve().parents[1]


def test_overview_links_to_customer_baskets(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    app = AppTest.from_file(ROOT / "main.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Your groceries, all in one place."
    assert any("Order #1405149" in item.value for item in app.markdown)

    app.button(key="overview_orders").click().run(timeout=30)
    assert app.title[0].value == "Order history"
    assert app.expander[0].label == "Order #1405149 · Purchase 10 · 3 items"
    assert any("Lemon Yogurt" in item.value for item in app.expander[0].markdown)


def test_recommendation_rows_show_catalog_categories(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    app = AppTest.from_file(ROOT / "main.py").run(timeout=30)
    app.chat_input[0].set_value("Recommend products based on my purchase history.").run(timeout=30)
    assert not app.exception
    reply = app.session_state["messages"][-1]
    assert "purchase frequency and recency" in reply["summary"]
    assert "Simulated prices" in reply["summary"]
    assert reply["products"] and all(row["category"] for row in reply["products"])
    assert all(row["price"].startswith("$") for row in reply["products"])


def test_stale_products_do_not_replace_a_recovery_message():
    checkpoint = {
        "route": "PRODUCT_RECOMMENDATION",
        "recommendations": [{"product_name": "Bananas", "product_id": 24852}],
    }
    recovery = "I couldn't load grocery suggestions right now. Please try again."
    assert recommendation_display(checkpoint, recovery, DB) == (recovery, [])
