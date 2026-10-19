from __future__ import annotations
REFUND_AGENT_PROMPT = """You are BrightCart's refund-policy specialist.
Explain the policy clearly and politely. Do not guarantee that a refund will be approved.
Use only the policy facts provided. Ask for missing details such as the order number,
delivery date, whether the item is unused, and whether it was a sale item. If the case is
damaged, unusual, outside the policy, or unclear, recommend human review. Never ask for a
full credit-card number or other unnecessary sensitive information."""

STORE_CONTEXT = {
    "order_status": "Orders usually ship in 1-2 business days. Order BC-1042 shipped yesterday and is expected Friday. Other orders require an order number.",
    "refund_policy": "BrightCart accepts unused items within 30 days of delivery. Sale items are exchange-only. Refunds go back to the original payment method in 5-7 business days after inspection.",
    "product_suggestion": "The TrailPro Backpack is $79, water-resistant, and best for commuting. The CityLite Tote is $49, lighter, and best for short daily trips. The Weekender Duffel is $89 and best for overnight travel.",
}

REFUND_POLICY = """BrightCart accepts unused items within 30 days of delivery.
Sale items are exchange-only. Refunds go back to the original payment method.
Refunds are processed in 5–7 business days after inspection.
"""


def route_intent(text: str) -> Intent:
    """Transparent router that selects one of three specialist agents."""
    t = text.lower()
    if any(w in t for w in ["order", "track", "tracking", "shipped", "delivery", "where is"]):
        return "order_status"
    if any(w in t for w in ["refund", "return", "money back", "exchange", "sale item", "used", "days ago"]):
        return "refund_policy"
    if any(w in t for w in ["recommend", "suggest", "which", "best", "backpack", "tote", "duffel"]):
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
    text = latest_user_text(state).lower()
    # Read only customer messages for policy facts. This prevents the agent's own
    # previous answer (which mentions "sale items") from changing later turns.
    customer_text = " ".join(
        str(m.content) for m in state.get("messages", []) if isinstance(m, HumanMessage)
    ).lower()
    if intent == "order_status":
        if "bc-1042" in text or "bc-1042" in customer_text:
            return "Your BrightCart order BC-1042 shipped yesterday and is expected Friday."
        return "I can check that for you. Please send your BrightCart order number."
    if intent == "refund_policy":
        if "sale" in text and "not on sale" not in text and "not a sale" not in text:
            return "Sale items are exchange-only under BrightCart's policy, so they are not eligible for a standard refund. A human representative can review unusual circumstances."
        if "45 days" in customer_text or "60 days" in customer_text or ("used" in text and "unused" not in text):
            return "The standard policy requires the item to be unused and returned within 30 days of delivery. Because this situation may be outside the standard policy, a human representative should review it."
        return "Unused items can be returned within 30 days of delivery. Sale items are exchange-only. After inspection, refunds return to the original payment method in 5–7 business days."
    if intent == "product_suggestion":
        return "For commuting, I recommend the $79 TrailPro Backpack because it is water-resistant. For a lighter short-trip option, the $49 CityLite Tote is a good choice."
    return "I can help with order status, refunds and returns, or product recommendations. What would you like to know?"


def make_specialist(intent: Intent):
    def specialist_node(state: AgentState) -> AgentState:
        if os.getenv("DEMO_MODE", "0") == "1" or not os.getenv("OPENAI_API_KEY"):
            answer = mock_answer(intent, state)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
            specialist_prompt = REFUND_AGENT_PROMPT if intent == "refund_policy" else f"You are the {intent.replace('_', ' ')} specialist."
            facts = REFUND_POLICY if intent == "refund_policy" else STORE_CONTEXT.get(intent, "")
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
        "general": "general_agent",  })
    for node in ["order_status_agent", "refund_policy_agent", "product_suggestion_agent", "general_agent"]:
        graph.add_edge(node, END)
    return graph.compile(checkpointer=MemorySaver())


def ask(graph, thread_id: str, text: str) -> str:
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, {"configurable": {"thread_id": thread_id}})
    return result["answer"]


def main():
    graph = build_graph()
    thread_id = "demo-customer-1"
    print("Nova: Hi! I can help with orders, refunds, or product recommendations. Type 'quit' to exit.")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        print(f"Nova: {ask(graph, thread_id, user)}")


if __name__ == "__main__":
    main()
