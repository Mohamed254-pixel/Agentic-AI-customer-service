"""Regression checks for the public, offline portfolio workflow."""

import shutil
from pathlib import Path

import duckdb
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from streamlit.testing.v1 import AppTest

from agents import order_agent, router
from db.queries import DB, get_order_products
from scripts.build_db import build_database

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PROMPT = "Recommend products based on my purchase history."


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "")


def ask(graph, text, user=5117):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)], "user_id": user},
        {"configurable": {"thread_id": "shared-test-thread"}},
    )


def test_customer_5117_documented_baskets():
    first = get_order_products(1662937)
    latest = get_order_products(1405149)
    assert len(first) == 4
    assert [row["product_name"] for row in latest] == [
        "Lemon Yogurt",
        "Blueberry Yoghurt",
        "Original No Pulp 100% Florida Orange Juice",
    ]
    assert order_agent.handle_order_question(5117, "What products were in my first order?")["rows"] == first


@pytest.mark.parametrize("user", [0, -1, True, "5117 OR 1=1", 2**80])
def test_invalid_customer_id(user):
    result = order_agent.handle_order_question(user, "What was in my latest order?")
    assert not result["rows"]
    assert "valid demo customer ID" in result["text"]


def test_empty_order_history():
    result = ask(router.build_router(), "What was in my latest order?", 2_147_483_647)
    assert "couldn't find any orders" in result["messages"][-1].content


def test_personalized_ranking_matches_independent_sql():
    with duckdb.connect(str(DB), read_only=True) as con:
        expected = con.execute("""
            SELECT p.product_id, COUNT(*) AS frequency
            FROM orders o
            JOIN (SELECT order_id, product_id FROM order_products__prior
                  UNION ALL SELECT order_id, product_id FROM order_products__train) b USING (order_id)
            JOIN products p USING (product_id)
            WHERE o.user_id = 5117
            GROUP BY p.product_id, p.product_name
            ORDER BY frequency DESC, MAX(o.order_number) DESC, p.product_name, p.product_id
            LIMIT 5
        """).fetchall()
    graph = router.build_router()
    result = ask(graph, HISTORY_PROMPT)
    assert result["recommendation_method"] == "customer_history"
    assert [(p["product_id"], p["shared_orders"]) for p in result["recommendations"]] == expected
    assert "purchase frequency and recency" in result["messages"][-1].content
    followup = ask(graph, "Why those?")
    assert "your sample purchase history" in followup["messages"][-1].content


def test_customer_without_history_gets_labeled_popularity():
    result = ask(router.build_router(), HISTORY_PROMPT, 2_147_483_647)
    assert result["recommendation_method"] == "popular_in_sample"
    assert result["recommendations"]
    assert "popular items across the sample" in result["messages"][-1].content


def test_same_thread_customer_change_clears_order_and_products():
    graph = router.build_router()
    assert ask(graph, "What was in my latest order?")["order_id"] == 1405149
    assert ask(graph, HISTORY_PROMPT)["suggested_product_ids"]
    result = ask(graph, "Show that again", user=89)
    assert result["order_id"] is None
    assert result["suggested_product_ids"] == []
    assert len(result["messages"]) == 2
    assert "1405149" not in result["messages"][-1].content


@pytest.mark.parametrize("question, expectation", [
    ("How can I get help with an order?", "contact the service where you ordered"),
    ("Can you change my order?", "cannot change or cancel orders"),
    ("Track my order", "Live delivery tracking is unavailable"),
])
def test_support_explains_capabilities(question, expectation):
    result = ask(router.build_router(), question)
    assert result["route"] == "CUSTOMER_SUPPORT"
    assert expectation in result["messages"][-1].content


def test_model_cannot_author_order_or_refund_facts():
    class Model:
        def invoke(self, messages):
            return AIMessage(content="Sure, your refund is approved and your order arrived yesterday.")

    graph = router.build_router(model=Model())
    assert "arrived yesterday" not in ask(graph, "What was in my latest order?")["messages"][-1].content
    assert "refund is approved" not in ask(graph, "I want a refund")["messages"][-1].content


def test_order_database_failure_is_recoverable(monkeypatch):
    graph = router.build_router()
    with monkeypatch.context() as broken:
        def unavailable(_):
            raise duckdb.IOException("private server details")

        broken.setattr(order_agent, "get_customer_orders", unavailable)
        result = ask(graph, "What was in my latest order?")
        assert result["messages"][-1].content == order_agent.DATABASE_ERROR_TEXT
    assert ask(graph, "What was in my latest order?")["order_id"] == 1405149


@pytest.mark.parametrize("extra, failure", [
    ("303158,45504,1,1", "duplicate keys"),
    ("2147483647,45504,1,0", "invalid prior basket rows"),
])
def test_ingestion_rejects_invalid_data_before_publication(tmp_path, extra, failure):
    sample = tmp_path / "sample"
    shutil.copytree(ROOT / "data/sample", sample)
    with (sample / "order_products__prior.csv").open("a") as csv_file:
        csv_file.write(f"{extra}\n")
    target = tmp_path / "bad.duckdb"
    with pytest.raises(ValueError, match=failure):
        build_database(sample, target)
    assert not target.exists()


def test_hosted_demo_does_not_use_present_key(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-fake-test-key")

    def reject_client(**kwargs):
        pytest.fail("Hosted Streamlit should not initialize an OpenAI client")

    monkeypatch.setattr(router, "ChatOpenAI", reject_client)
    app = AppTest.from_file(ROOT / "main.py").run(timeout=30)
    assert not app.exception
    next(button for button in app.button if button.label == HISTORY_PROMPT).click().run(timeout=30)
    assert not app.exception
    assert "purchase frequency and recency" in app.chat_message[-1].markdown[0].value
