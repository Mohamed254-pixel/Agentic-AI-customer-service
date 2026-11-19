import pytest
from langchain_core.messages import HumanMessage

from agents.router import build_router


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    return build_router()


def ask(graph, text):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)], "user_id": 5117},
        config={"configurable": {"thread_id": "smalltalk-test"}},
    )


@pytest.mark.parametrize("message", [
    "wssup", "how ist goin", "yo", "what's good", "how you doing",
    "heyyy!", "hey, how's it going?", "good morning",
])
def test_casual_greetings_are_answered_offline(graph, message):
    result = ask(graph, message)
    assert result["route"] == "SMALLTALK"
    assert result["mode"] == "Local demo"
    assert "demo mode" not in result["messages"][-1].content.lower()


def test_social_replies_vary_and_preserve_order_context(graph):
    ask(graph, "What was in my latest order?")
    replies = [ask(graph, "yo") for _ in range(3)]
    assert len({reply["messages"][-1].content for reply in replies}) == 3
    assert all(reply["active_route"] == "ORDER_HISTORY" for reply in replies)
    assert ask(graph, "thx!")["messages"][-1].content == "You're welcome!"
    assert "1405149" in ask(graph, "Show that again")["messages"][-1].content


def test_social_prefix_keeps_service_intent_and_followups_work(graph):
    assert ask(graph, "yo, recommend bananas")["route"] == "PRODUCT_RECOMMENDATION"
    assert ask(graph, "what's good")["active_route"] == "PRODUCT_RECOMMENDATION"
    assert ask(graph, "why those?")["route"] == "PRODUCT_RECOMMENDATION"
    assert ask(graph, "hey, can I get a refund?")["route"] == "CUSTOMER_SUPPORT"
    assert ask(graph, "how you doing")["active_route"] == "CUSTOMER_SUPPORT"
    assert ask(graph, "it was unused")["route"] == "CUSTOMER_SUPPORT"


def test_offline_general_question_leaves_specialist_topic(graph):
    ask(graph, "Recommend bananas")
    result = ask(graph, "What is the weather today?")
    assert result["active_route"] == "SMALLTALK"
    assert "I stick to grocery help" in result["messages"][-1].content
    assert ask(graph, "yo, show my latest order")["route"] == "ORDER_HISTORY"
