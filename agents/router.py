"""LangGraph router for the Instacart customer-service demo.

This module connects the existing order-history and product-recommendation
functions to a small LangGraph workflow.  It keeps the database-backed agents
as the source of truth and uses the language model only to choose a route and
handle simple conversation text.
"""

from __future__ import annotations

import os
import re
import json
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
from agents.product_agent import is_product_reference, normalize_request, parse_exclusions, product_turn
from agents.refund_agent import handle_refund_question
from agents.voice import CHAT_STYLE


ROOT = Path(__file__).resolve().parents[1]


class RouterState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_id: int
    route: str
    order_id: int
    product_question: str
    effective_question: str
    mode: str
    active_route: str
    excluded_products: list[str]
    suggested_product_ids: list[int]
    last_suggested_product_ids: list[int]
    seed_product_id: int | None
    recommendations: list[dict[str, Any]]
    recommendation_method: str
    matched_product: str | None


ROUTER_PROMPT = """You route requests for an Instacart customer-service demo.

Return exactly one label:
- ORDER_HISTORY: questions about past orders or items in an order
- PRODUCT_RECOMMENDATION: grocery product suggestions or products bought together
- CUSTOMER_SUPPORT: refunds, policies, complaints, or general support questions
- SMALLTALK: greetings, thanks, goodbyes, introductory questions, or general knowledge/conversation outside the grocery service tasks

The dataset does not contain live delivery tracking, prices, payment data,
quantities, customer addresses, or actual refund processing. Use CUSTOMER_SUPPORT
for those questions so the system can explain the limitation or policy.
Product rejections such as "nah give me sum else", "no milk and bananas",
"not those", and "what about apples?" continue product recommendations. They
are not refund requests. Only switch to support for actual support intent.
Use the active topic for short follow-ups, even after greetings or thanks.
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
    return "SMALLTALK"


def _explicit_route(question: str):
    """Prioritize actual service requests, not generic words like 'wrong'."""
    text = normalize_request(question)
    if re.search(
        r"\b(refund(?:ed|s)?|returns?|returned|exchange|damaged|spoiled|cancel(?:led|ation)?|"
        r"complaint|policy|charged|billing|tracking|track|delivery|shipped|payment)\b"
        r"|\bmoney back\b|\b(wrong|missing)\s+(order|item|items|delivery|product)\b"
        r"|\b(order|item|delivery)\b.*\b(wrong|missing)\b"
        r"|\bmy\b.{1,60}\b(missing|never arrived)\b", text,
    ):
        return "CUSTOMER_SUPPORT"
    if (re.search(r"\b(prices?|costs?|cheapest|cheaper|expensive)\b|\bhow much\b", text)
            and is_product_reference(question)):
        return "PRODUCT_RECOMMENDATION"
    if re.search(r"\b(recommend(?:ation|ations|s)?|suggest(?:ion|ions|s)?|similar|grocery|groceries|products?)\b"
                 r"|\bwhat should i (buy|get)\b", text):
        return "PRODUCT_RECOMMENDATION"
    if re.search(r"\b(order|orders|purchased|bought)\b"
                 r"|\bdid i (buy|order|purchase)\b|\b(first|initial) (order|purchase)\b", text):
        return "ORDER_HISTORY"
    return None


def _smalltalk(question: str):
    return bool(re.fullmatch(
        r"(?:hi|hello|hey|thanks|thank you|thx|ty|bye|goodbye|ok|okay|cool|nice|alright|aight|ight|how are you)(?: yo| bro| fam)?[!.?\s]*",
        normalize_request(question).strip(),
    ))


def _general_request(question: str):
    text = normalize_request(question)
    # General questions must be able to leave a previous shopping conversation.
    return bool(re.search(
        r"\b(python|javascript|programming|photosynthesis|math|algebra|homework|movie|movies|music|song|songs|poem|joke|capital|weather|news)\b"
        r"|^(?:explain|write|solve|translate|summarize|tell me about)\b"
        r"|^what(?: is|'s| are) (?!my\b|in\b|your (?:refund|return|policy))", text,
    ))


def _product_followup(question: str):
    text = normalize_request(question)
    return bool(parse_exclusions(question)[1] or re.search(
        r"\b(nah|nope|no|else|different|another|more|other|those|these|them|that|it|one|ones|instead|gimme)\b"
        r"|\b(what|how) about\b|\bgive me\b|\b(don't|not) (?:like|want)\b", text,
    ))


def _context_route(question: str, state):
    route = _explicit_route(question)
    general_domain = re.search(
        r"\b(python|javascript|programming|photosynthesis|math|algebra|homework|movie|movies|music|song|songs|poem|joke|capital|weather|news)\b",
        normalize_request(question),
    )
    if (_general_request(question) and route not in {"CUSTOMER_SUPPORT", "ORDER_HISTORY"}
            and (route != "PRODUCT_RECOMMENDATION" or general_domain)):
        return "SMALLTALK"
    if route:
        return route
    if _smalltalk(question):
        return "SMALLTALK"
    topic = state.get("active_route", state.get("route"))
    # Substantive follow-ups stay with their specialist. A thanks/greeting
    # doesn't erase this topic or product exclusions/history.
    if topic == "PRODUCT_RECOMMENDATION" and _product_followup(question):
        return topic
    text = normalize_request(question)
    if (len(text.split()) <= 5 and not re.match(r"(?:what|why|who|when|where|how|can|could|explain|tell|write|solve)\b", text)
            and is_product_reference(question)):
        return "PRODUCT_RECOMMENDATION"
    if topic == "ORDER_HISTORY" and re.search(
        r"\b(that|those|them|it|again|first|last|latest|earliest|oldest|initial|one|ones)\b", normalize_request(question)
    ):
        return topic
    if topic == "SMALLTALK" and re.search(r"\b(more|that|it|those|explain|why|how)\b", text):
        return "SMALLTALK"
    if topic == "CUSTOMER_SUPPORT" and re.search(
        r"\b(it|that|unused|used|sale|days|ago|arrived|item)\b", normalize_request(question)
    ):
        return topic
    if parse_exclusions(question)[1]:
        return "PRODUCT_RECOMMENDATION"
    return None


def _keyword_route(question: str) -> str:
    return _context_route(question, {}) or "SMALLTALK"



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
        topic = state.get("active_route", state.get("route"))
        forced = _context_route(question, state)
        changes_order = re.search(r"\b(first|earliest|oldest|initial|latest|last|recent)\b", question.lower())
        if (forced == "ORDER_HISTORY" and topic == "ORDER_HISTORY" and not changes_order
                and state.get("order_id") and re.search(r"\b(that|those|them|it|again)\b", question.lower())
                and not re.search(r"\border\s*#?\s*\d+", question.lower())):
            effective += f" (order {state['order_id']})"
        mode = "Local demo" if model is None else "OpenAI"
        if model is None:
            route = forced or _keyword_route(effective)
        else:
            try:
                result = model.invoke([
                    SystemMessage(content=ROUTER_PROMPT + f"\nActive topic: {topic or 'none'}"),
                    *state.get("messages", []),
                ])
                route = forced or _normalise_route(_message_text(result))
            except Exception:
                mode = "Local fallback (OpenAI unavailable)"
                route = forced or _keyword_route(effective)
        if debug:
            print(f"Router selected: {route}")
        return {"route": route, "effective_question": effective, "mode": mode,
                "active_route": topic if route == "SMALLTALK" and _smalltalk(question) else route}

    def conversational_order(state, result):
        """Let the model explain retrieved facts, with a safe database fallback."""
        fallback = result["text"]
        if model is None or not result.get("rows"):
            return fallback
        facts = json.dumps(result, default=str)
        try:
            response = model.invoke([
                SystemMessage(content=CHAT_STYLE + "\nAnswer using only the verified order facts below. "
                    "Keep every product name and relevant order ID exactly as supplied. "
                    "Do not infer quantities, order dates, tracking, prices, or refunds. "
                    "For a repeated question, briefly acknowledge it and show the details again.\n"
                    + facts),
                *state.get("messages", []),
            ])
            answer = _message_text(response).strip()
            # Preserve the factual list and reject newly invented numeric details.
            names = [row["product_name"] for row in result["rows"] if "product_name" in row]
            allowed_numbers = set(re.findall(r"\d+", facts)) | {str(len(result["rows"]))}
            if (not answer or any(name not in answer for name in names)
                    or not set(re.findall(r"\d+", answer)).issubset(allowed_numbers)):
                return fallback
            return answer
        except Exception:
            return fallback

    def order_node(state: RouterState):
        question = state.get("effective_question", _last_user_message(state))
        user_id = state.get("user_id")
        if user_id is None:
            answer = "Please select a customer account before asking about order history."
        else:
            result = handle_order_question(user_id, question)
            answer = conversational_order(state, result)
            if result.get("rows") and "product_name" in result["rows"][0]:
                match = re.search(r"Order (\d+) contained", result["text"])
                if match:
                    return {"messages": [AIMessage(content=answer)], "order_id": int(match[1])}
        return {"messages": [AIMessage(content=answer)]}

    def product_node(state: RouterState):
        try:
            result = product_turn(_last_user_message(state), state,
                                  messages=state.get("messages", []), model=model,
                                  use_ai=model is not None)
        except Exception:
            return {"messages": [AIMessage(content="I couldn't load grocery suggestions right now. Please try again.")]}
        updates = {key: value for key, value in result.items() if key not in {"text", "ai_fallback"}}
        return {**updates, "messages": [AIMessage(content=result["text"])],
                "mode": "Local fallback (OpenAI unavailable)" if result["ai_fallback"] else state.get("mode", "Local demo")}

    def support_node(state: RouterState):
        answer = handle_refund_question(state.get("messages", []), model=model)
        return {"messages": [AIMessage(content=answer)], "mode": "Local fallback (OpenAI unavailable)" if "OpenAI is unavailable" in answer else state.get("mode", "Local demo")}

    def smalltalk_node(state: RouterState):
        text = normalize_request(_last_user_message(state))
        if re.fullmatch(r"(?:thanks|thank you|thx|ty)(?: yo| bro| fam)?[!.?\s]*", text):
            no_model_answer = "You're welcome!"
        elif _smalltalk(text):
            no_model_answer = "Hey! What can I help you with?"
        else:
            no_model_answer = "I'm in demo mode, so I can look up groceries and orders but can't answer general questions yet."
        response_mode = state.get("mode", "Local demo")
        if model is None:
            answer = no_model_answer
        else:
            try:
                result = model.invoke([
                    SystemMessage(content=CHAT_STYLE + "\nYou can help with historical grocery orders, "
                        "product suggestions, and simulated refund questions. Mention these only when asked. "
                        "You can also answer general questions, explain ideas, help with writing or coding, "
                        "and chat normally. Use simple English and keep it short unless asked for detail. "
                        "You have no live web search: say when you cannot verify current information. "
                        "Do not invent Cartside products, customer records, prices, or actions. "
                        "If a question needs a product lookup, ask which grocery item rather than inventing a store recommendation."),
                    *state.get("messages", []),
                ])
                answer = _message_text(result)
                response_mode = "OpenAI"
            except Exception:
                answer = "I couldn't reach AI chat just now. I can still help with the sample orders and groceries."
                response_mode = "Local fallback (OpenAI unavailable)"
        return {"messages": [AIMessage(content=answer)], "mode": response_mode}

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
