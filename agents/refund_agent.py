from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

Intent = Literal["order_status", "refund_policy", "product_suggestion", "general"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Intent
    answer: str


SYSTEM_PROMPT = """You are Cartside, a friendly customer-service and sales assistant for Cartside.
You must be concise, accurate, and helpful. Use only the store information supplied by the
specialist agent. If information is missing, say what you need and never invent an order,
refund approval, price, or delivery date. Always identify the policy as simulated; never claim a refund or order change was completed. Keep the conversation context in mind."""

REFUND_AGENT_PROMPT = """You are Cartside's refund-policy specialist.
Explain the policy clearly and politely. Do not guarantee that a refund will be approved.
Use only the policy facts provided. Ask for missing details such as the order number,
delivery date, whether the item is unused, and whether it was a sale item. If the case is
damaged, unusual, outside the policy, or unclear, recommend human review. Never ask for a
full credit-card number or other unnecessary sensitive information."""

POLICY_PATH = Path(__file__).resolve().parents[1] / "data/policies/refund_policy.txt"
REFUND_POLICY = POLICY_PATH.read_text(encoding="utf-8").strip()


def handle_refund_question(messages, model=None):
    """Explain a clearly simulated policy using the saved conversation."""
    fallback = "Simulated Cartside policy (class demo):\n\n" + REFUND_POLICY
    if model is None:
        return fallback
    try:
        result = model.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            SystemMessage(content=REFUND_AGENT_PROMPT + "\nSIMULATED POLICY:\n" + REFUND_POLICY),
            *messages,
        ])
        return str(result.content)
    except Exception:
        return fallback + "\n\nOpenAI is unavailable; showing the policy directly."


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
        if os.getenv("DEMO_MODE", "0") == "1" or not os.getenv("OPENAI_API_KEY"):
            answer = mock_answer(intent, state)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, timeout=20, max_retries=1)
            specialist_prompt = REFUND_AGENT_PROMPT if intent == "refund_policy" else f"You are the {intent.replace('_', ' ')} specialist."
            facts = REFUND_POLICY if intent == "refund_policy" else "Use the main Cartside router for historical orders and grocery recommendations. No live tracking or prices are available."
            prompt = [
                SystemMessage(content=SYSTEM_PROMPT),
                SystemMessage(content=f"{specialist_prompt}\nStore facts:\n{facts}"),
                *state.get("messages", []),
            ]
            answer = str(llm.invoke(prompt).content)
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



