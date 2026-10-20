"""LangGraph router for the Instacart customer-service demo.

This module connects the existing order-history and product-recommendation
functions to a small LangGraph workflow.  It keeps the database-backed agents
as the source of truth and uses the language model only to choose a route and
handle simple conversation text.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, TypedDict

import operator
from functools import lru_cache
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.order_agent import handle_order_question
from agents.product_agent import recommend_products
from agents.refund_agent import handle_refund_question


ROOT = Path(__file__).resolve().parents[1]


class RouterState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_id: int
    route: str
    order_id: int
    product_question: str
    effective_question: str
    mode: str


ROUTER_PROMPT = """You route requests for an Instacart customer-service demo.

Return exactly one label:
- ORDER_HISTORY: questions about past orders or items in an order
- PRODUCT_RECOMMENDATION: grocery product suggestions or products bought together
- CUSTOMER_SUPPORT: refunds, policies, complaints, or general support questions
- SMALLTALK: greetings, thanks, or goodbyes

The dataset does not contain live delivery tracking, prices, payment data,
quantities, customer addresses, or actual refund processing. Use CUSTOMER_SUPPORT
for those questions so the system can explain the limitation or policy.
Return only the label.
"""


def _message_text(message: AnyMessage | dict[str, Any]) -> str:
    """Extract text from either LangChain or dictionary-style messages."""

    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _last_user_message(state: RouterState) -> str:
    """Return the latest user message, ignoring prior assistant messages."""

    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return _message_text(message)
        if isinstance(message, dict) and message.get("role") == "user":
            return _message_text(message)
    return _message_text(state.get("messages", [])[-1]) if state.get("messages") else ""


def _normalise_route(value: str) -> str:
    """Convert model output into one safe graph route."""

    value = value.upper().strip()
    for route in (
        "ORDER_HISTORY",
        "PRODUCT_RECOMMENDATION",
        "CUSTOMER_SUPPORT",
        "SMALLTALK",
    ):
        if route in value:
            return route
    return "CUSTOMER_SUPPORT"


def _keyword_route(question: str) -> str:
    """Provide a local fallback when no OpenAI key is configured."""

    text = question.lower()
    # Order matters: real requests are checked before greetings, so
    # "Hello, I want a refund" is not treated as small talk.
    if re.search(
        r"\b(refund|refunded|return|returned|damaged|missing|wrong|cancel|"
        r"complaint|policy|charged)\b",
        text,
    ):
        return "CUSTOMER_SUPPORT"
    # An explicit ask for suggestions wins over incidental order wording
    # ("recommend something based on my last order").
    if re.search(r"\b(recommend|suggest|similar)\b", text):
        return "PRODUCT_RECOMMENDATION"
    # Past-purchase wording ("what did I buy last time?") is order history.
    if re.search(
        r"\b(order|orders|purchased|bought|history|latest|last time)\b"
        r"|\bdid i (buy|order|purchase)\b",
        text,
    ):
        return "ORDER_HISTORY"
    if re.search(
        r"\b(product|grocery|groceries)\b|\bwhat should i (buy|get)\b", text
    ):
        return "PRODUCT_RECOMMENDATION"
    if re.search(r"\b(hi|hello|hey|thanks|thank you|bye|goodbye)\b", text):
        return "SMALLTALK"
    return "CUSTOMER_SUPPORT"



def build_router(model=None, debug: bool = False):
    """Build and compile the project's LangGraph router.

    A model can be injected for tests. If omitted, ChatOpenAI uses the
    OPENAI_API_KEY and OPENAI_MODEL values from the project's .env file.
    """

    load_dotenv(ROOT / ".env")
    if model is None and os.getenv("OPENAI_API_KEY") and os.getenv("DEMO_MODE", "0") != "1":
        model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, timeout=20, max_retries=1)

    def choose_route(state: RouterState):
        question = _last_user_message(state)
        effective = question
        previous_route = state.get("route")
        if re.search(r"\b(that|those|them|it|else|again)\b", question.lower()):
            if previous_route == "ORDER_HISTORY" and state.get("order_id") and not re.search(r"\border\s*#?\s*\d+", question.lower()):
                effective += f" (order {state['order_id']})"
            elif previous_route == "PRODUCT_RECOMMENDATION" and state.get("product_question"):
                effective += " (previous product request: " + state["product_question"] + ")"
        mode = "Local demo" if model is None else "OpenAI"
        if model is None:
            route = _keyword_route(effective)
        else:
            try:
                result = model.invoke([
                    SystemMessage(content=ROUTER_PROMPT),
                    *state.get("messages", []),
                ])
                route = _normalise_route(_message_text(result))
            except Exception:
                mode = "Local fallback (OpenAI unavailable)"
                # Network, rate-limit, or key problems: use the local rules.
                route = _keyword_route(effective)
        if debug:
            print(f"Router selected: {route}")
        return {"route": route, "effective_question": effective, "mode": mode}

    def order_node(state: RouterState):
        question = state.get("effective_question", _last_user_message(state))
        user_id = state.get("user_id")
        if user_id is None:
            answer = "Please select a customer account before asking about order history."
        else:
            result = handle_order_question(user_id, question)
            answer = result["text"]
            if result.get("rows") and "product_name" in result["rows"][0]:
                match = re.search(r"Order (\d+) contained", answer)
                if match:
                    return {"messages": [AIMessage(content=answer)], "order_id": int(match[1])}
        return {"messages": [AIMessage(content=answer)]}

    def product_node(state: RouterState):
        try:
            answer = recommend_products(state.get("effective_question", _last_user_message(state)), use_ai=model is not None)
        except FileNotFoundError as exc:
            # The product agent raises this when the DuckDB file isn't built yet.
            answer = str(exc)
        except Exception:
            answer = "I couldn't load product recommendations right now. Please try again later."
        return {"messages": [AIMessage(content=answer)], "product_question": state.get("effective_question", _last_user_message(state)), "mode": "Local fallback (OpenAI unavailable)" if "AI explanation unavailable" in answer else state.get("mode", "Local demo")}

    def support_node(state: RouterState):
        answer = handle_refund_question(state.get("messages", []), model=model)
        return {"messages": [AIMessage(content=answer)], "mode": "Local fallback (OpenAI unavailable)" if "OpenAI is unavailable" in answer else state.get("mode", "Local demo")}

    def smalltalk_node(state: RouterState):
        no_model_answer = "Hello! I can help with order history, grocery recommendations, and simulated customer support questions."
        if model is None:
            answer = no_model_answer
        else:
            try:
                result = model.invoke([
                    SystemMessage(content=(
                        "Respond briefly and professionally to the user's small talk. "
                        "Mention that you can help with order history, grocery recommendations, "
                        "and simulated customer support questions."
                    )),
                    HumanMessage(content=_last_user_message(state)),
                ])
                answer = _message_text(result)
            except Exception:
                answer = no_model_answer
        return {"messages": [AIMessage(content=answer)]}

    graph = StateGraph(RouterState)
    graph.add_node("Router", choose_route)
    graph.add_node("Order_History_Agent", order_node)
    graph.add_node("Product_Recommendation_Agent", product_node)
    graph.add_node("Customer_Support_Agent", support_node)
    graph.add_node("Small_Talk", smalltalk_node)
    graph.set_entry_point("Router")
    graph.add_conditional_edges(
        "Router",
        lambda state: state["route"],
        {
            "ORDER_HISTORY": "Order_History_Agent",
            "PRODUCT_RECOMMENDATION": "Product_Recommendation_Agent",
            "CUSTOMER_SUPPORT": "Customer_Support_Agent",
            "SMALLTALK": "Small_Talk",
            "END": END,
        },
    )
    for node in (
        "Order_History_Agent",
        "Product_Recommendation_Agent",
        "Customer_Support_Agent",
        "Small_Talk",
    ):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=MemorySaver())


@lru_cache(maxsize=1)
def get_router():
    return build_router()


def invoke_router(question: str, user_id: int, thread_id: str):
    """Convenience function for Streamlit or a future API endpoint."""

    return get_router().invoke(
        {
            "messages": [HumanMessage(content=question)],
            "user_id": user_id,
        },
        config={"configurable": {"thread_id": f"{user_id}:{thread_id}"}},
    )
