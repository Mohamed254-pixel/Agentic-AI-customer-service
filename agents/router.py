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

from functools import lru_cache
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from agents.order_agent import handle_order_question
from agents.product_agent import is_product_reference, normalize_request, parse_exclusions, product_turn
from agents.refund_agent import handle_refund_question
from agents.voice import CHAT_STYLE


ROOT = Path(__file__).resolve().parents[1]


class RouterState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: int
    context_customer: int | None
    route: str | None
    order_id: int | None
    product_question: str
    effective_question: str
    mode: str
    active_route: str | None
    excluded_products: list[str]
    suggested_product_ids: list[int]
    last_suggested_product_ids: list[int]
    seed_product_id: int | None
    recommendations: list[dict[str, Any]]
    recommendation_method: str | None
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
        r"|\bmy\b.{1,60}\b(missing|never arrived)\b"
        r"|\b(help|support|contact)\b.*\b(orders?|customer service)\b"
        r"|\b(change|modify|edit|update|place)\b.*\b(my|an|the|a)\s+order\b", text,
    ):
        return "CUSTOMER_SUPPORT"
    if (re.search(r"\b(prices?|costs?|cheapest|cheaper|expensive)\b|\bhow much\b", text)
            and is_product_reference(question)):
        return "PRODUCT_RECOMMENDATION"
    if re.search(r"\b(recommend(?:ation|ations|s)?|suggest(?:ion|ions|s)?|similar)\b"
                 r"|\bwhat should i (buy|get)\b", text):
        return "PRODUCT_RECOMMENDATION"
    if re.search(r"\b(order|orders|purchased|bought)\b"
                 r"|\bdid i (buy|order|purchase)\b|\b(first|initial) (order|purchase)\b", text):
        return "ORDER_HISTORY"
    if re.search(r"\b(grocery|groceries|products?)\b", text):
        return "PRODUCT_RECOMMENDATION"
    return None


GREETING = r"(?:h+i+|h+e+y+|h+e+l+o+|hiya|yo+|sup|wassup|wssup|whatsup|what'?s up|what'?s good|good (?:morning|afternoon|evening))"
CHECK_IN = r"(?:how (?:are (?:you|u)|(?:are )?(?:you|u) doin(?:g)?'?|(?:is it|ist|it'?s|s) goin(?:g)?'?)|how'?s it goin(?:g)?'?)"
SOCIAL_PATTERNS = {
    "check_in": re.compile(rf"(?:{GREETING} )?{CHECK_IN}"),
    "greeting": re.compile(GREETING + r"(?: there| cartside| yo| bro| fam)?"),
    "thanks": re.compile(r"(?:thanks(?: a lot| so much)?|thank (?:you|u)(?: so much)?|thx|ty|tysm|appreciate it)(?: yo| bro| fam| cartside)?"),
    "goodbye": re.compile(r"bye(?: bye)?|goodbye|see (?:you|ya)(?: later)?|later|take care"),
    "acknowledgment": re.compile(r"ok(?:ay)?|cool|nice|alright|aight|ight|got it|sounds good"),
    "capabilities": re.compile(r"what (?:can you (?:do|help (?:me )?with)|do you do)|who are you"),
}
SOCIAL_REPLIES = {
    "check_in": ("Hey! I'm here to help. What's on your mind?", "Ready to help with your groceries. What's up?"),
    "greeting": ("Hey! What can I help you with?", "Hi there! What's on your shopping list today?", "Hey! Need a hand with anything?"),
    "thanks": ("You're welcome!", "Anytime!"),
    "goodbye": ("Take care!", "See you next time!"),
    "acknowledgment": ("Got it.", "Sounds good."),
    "capabilities": ("I can look up past orders, suggest groceries, and explain Cartside's simulated support policies.",),
}
OFFLINE_LIMITATION = "I stick to grocery help in demo mode. I can look up past orders or suggest something for your next shop."


def _social_intent(question: str) -> str | None:
    """Match whole social messages so a greeting can't swallow a service request."""
    text = " ".join(re.sub(r"[^a-z0-9'\s]", " ", question.lower().replace("’", "'")).split())
    return next((intent for intent, pattern in SOCIAL_PATTERNS.items() if pattern.fullmatch(text)), None)


def _social_reply(intent, messages):
    """Rotate replies within a thread without random or shared state."""
    replies = SOCIAL_REPLIES[intent]
    previous = next((message.content for message in reversed(messages)
                     if isinstance(message, AIMessage) and message.content in replies), None)
    return replies[(replies.index(previous) + 1) % len(replies)] if previous else replies[0]


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
    if _social_intent(question):
        return "SMALLTALK"
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



def build_router(model=None, debug: bool = False, *, allow_openai: bool = True):
    """Build and compile the project's LangGraph router.

    A model can be injected for tests. If omitted, ChatOpenAI uses the
    OPENAI_API_KEY and OPENAI_MODEL values from the project's .env file.
    """

    load_dotenv(ROOT / ".env")
    if allow_openai and model is None and os.getenv("OPENAI_API_KEY") and os.getenv("DEMO_MODE") == "0":
        model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, timeout=15, max_retries=0, max_tokens=400)

    def isolate_customer(state: RouterState):
        user_id = state.get("user_id")
        previous_customer = state.get("context_customer")
        if type(previous_customer) is type(user_id) and previous_customer == user_id:
            return {}
        latest = next((m for m in reversed(state.get("messages", []))
                       if isinstance(m, HumanMessage)), None)
        return {
            "context_customer": user_id,
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *([latest] if latest else [])],
            "active_route": None, "route": None, "order_id": None,
            "excluded_products": [], "suggested_product_ids": [],
            "last_suggested_product_ids": [], "seed_product_id": None,
            "recommendations": [], "recommendation_method": None,
            "matched_product": None, "product_question": "",
        }

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
                    *state.get("messages", [])[-12:],
                ])
                route = forced or _normalise_route(_message_text(result))
            except Exception:
                mode = "Local fallback (OpenAI unavailable)"
                route = forced or _keyword_route(effective)
        if debug:
            print(f"Router selected: {route}")
        return {"route": route, "effective_question": effective, "mode": mode,
                "active_route": topic if route == "SMALLTALK" and _social_intent(question) else route}

    def conversational_order(state, result):
        """Optionally add a fixed acknowledgment to the rendered database facts."""
        fallback = result["text"]
        if model is None or not result.get("rows"):
            return fallback
        try:
            response = model.invoke([
                SystemMessage(content=CHAT_STYLE + "\nReply with Sure if a short acknowledgment fits. "
                              "Cartside will display the order records directly."),
                *state.get("messages", [])[-12:],
            ])
            if re.match(r"^sure\b", _message_text(response).strip(), re.I):
                return "Sure. " + fallback
            return fallback
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
            return {"messages": [AIMessage(content=answer)], "order_id": result.get("order_id")}
        return {"messages": [AIMessage(content=answer)]}

    def product_node(state: RouterState):
        try:
            result = product_turn(_last_user_message(state), state,
                                  messages=state.get("messages", [])[-12:], model=model,
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
        intent = _social_intent(_last_user_message(state))
        response_mode = state.get("mode", "Local demo")
        if intent:
            answer = _social_reply(intent, state.get("messages", []))
        elif model is None:
            answer = OFFLINE_LIMITATION
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
                    *state.get("messages", [])[-12:],
                ])
                answer = _message_text(result)
                response_mode = "OpenAI"
            except Exception:
                answer = "I couldn't reach AI chat just now. I can still help with the sample orders and groceries."
                response_mode = "Local fallback (OpenAI unavailable)"
        return {"messages": [AIMessage(content=answer)], "mode": response_mode}

    graph = StateGraph(RouterState)
    graph.add_node("Customer_Context", isolate_customer)
    graph.add_node("Router", choose_route)
    graph.add_node("Order_History_Agent", order_node)
    graph.add_node("Product_Recommendation_Agent", product_node)
    graph.add_node("Customer_Support_Agent", support_node)
    graph.add_node("Small_Talk", smalltalk_node)
    graph.set_entry_point("Customer_Context")
    graph.add_edge("Customer_Context", "Router")
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
