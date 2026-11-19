from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from agents.voice import CHAT_STYLE

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

Intent = Literal["order_status", "refund_policy", "product_suggestion", "general"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Intent
    answer: str


POLICY_PATH = Path(__file__).resolve().parents[1] / "data/policies/refund_policy.txt"
REFUND_POLICY = POLICY_PATH.read_text(encoding="utf-8").strip()


def handle_refund_question(messages, model=None):
    """Use fixed policy text and let the optional model select a support topic."""
    question = next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    topic = "refund"
    if re.search(r"\b(track|tracking|delivery status|where is my order|shipped|arriv)\w*\b", question, re.I):
        topic = "tracking"
    if re.search(r"\b(cancel|change|edit|modify|update|place|payment|billing|charged)\w*\b", question, re.I):
        topic = "changes"
    if re.search(r"\b(help|support|contact)\b", question, re.I):
        topic = "help"
    if re.search(r"\b(refund|return|exchange|damaged|spoiled|unused|sale)\w*\b", question, re.I):
        topic = "refund"

    fallback = ""
    if model is not None:
        try:
            response = model.invoke([
                SystemMessage(content=CHAT_STYLE + "\nReturn one support label: refund, tracking, changes, or help. "
                              "Cartside will display the simulated policy directly."),
                *messages[-12:],
            ])
            if topic == "refund" and not re.search(r"\b(refund|return|exchange)\w*\b", question, re.I):
                topic = {"tracking": "tracking", "changes": "changes", "help": "help"}.get(
                    str(response.content).strip().lower(), topic
                )
        except Exception:
            fallback = "\n\nOpenAI is unavailable; showing the policy directly."

    introduction = {
        "refund": "",
        "tracking": "Live delivery tracking is unavailable. Ask to see your latest historical order.\n\n",
        "changes": "Cartside cannot change or cancel orders, take payments, or issue refunds. "
                   "For a real order, use the support channel where you placed it.\n\n",
        "help": "For past items, ask about your latest order or open Orders. For a real delivery, "
                "order change, or refund, contact the service where you ordered. "
                "Cartside can explain the simulated policy below.\n\n",
    }[topic]
    return introduction + "Simulated Cartside policy (portfolio demo):\n\n" + REFUND_POLICY + fallback


def route_intent(text: str) -> Intent:
    """Transparent router that selects one of three specialist agents."""
    t = text.lower()
    if any(w in t for w in ["refund", "return", "money back", "exchange"]):
        return "refund_policy"
    if any(w in t for w in ["order", "track", "tracking", "shipped", "delivery", "where is"]):
        return "order_status"
    if any(w in t for w in ["refund", "return", "money back", "exchange", "sale item", "used", "days ago"]):
        return "refund_policy"
    if any(w in t for w in ["recommend", "suggest", "which", "best", "grocery", "groceries", "product"]):
        return "product_suggestion"
    return "general"


def latest_user_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def router_node(state: AgentState) -> AgentState:
    return {"intent": route_intent(latest_user_text(state))}


def mock_answer(intent: Intent, state: AgentState) -> str:
    if intent == "refund_policy":
        return handle_refund_question(state.get("messages", []))
    return "This module handles simulated refunds. Use the main Cartside router for order history and grocery recommendations."


def make_specialist(intent: Intent):
    def specialist_node(state: AgentState) -> AgentState:
        if intent == "refund_policy" and os.getenv("DEMO_MODE") == "0" and os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                               temperature=0, timeout=15, max_retries=0, max_tokens=400)
            answer = handle_refund_question(state.get("messages", []), model=model)
        else:
            answer = mock_answer(intent, state)
        return {"answer": answer, "messages": [AIMessage(content=answer)]}
    return specialist_node


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("order_status_agent", make_specialist("order_status"))
    graph.add_node("refund_policy_agent", make_specialist("refund_policy"))
    graph.add_node("product_suggestion_agent", make_specialist("product_suggestion"))
    graph.add_node("general_agent", make_specialist("general"))
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda s: s["intent"], {
        "order_status": "order_status_agent",
        "refund_policy": "refund_policy_agent",
        "product_suggestion": "product_suggestion_agent",
        "general": "general_agent",
    })
    for node in ["order_status_agent", "refund_policy_agent", "product_suggestion_agent", "general_agent"]:
        graph.add_edge(node, END)
    return graph.compile(checkpointer=MemorySaver())


def ask(graph, thread_id: str, text: str) -> str:
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, {"configurable": {"thread_id": thread_id}})
    return result["answer"]


def main():
    graph = build_graph()
    thread_id = "demo-customer-1"
    print("Cartside: Hi! I can help with orders, refunds, or product recommendations. Type 'quit' to exit.")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        print(f"Cartside: {ask(graph, thread_id, user)}")


if __name__ == "__main__":
    main()



