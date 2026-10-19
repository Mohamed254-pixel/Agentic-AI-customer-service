
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def product_agent_test(state: ChatState):
    preferences = []

    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            for fruit in ["bananas", "strawberries", "apples"]:
                if fruit in message.content.lower():
                    if fruit not in preferences:
                        preferences.append(fruit)

    question = state["messages"][-1].content.lower()

    if "what fruits" in question:
        answer = "You mentioned: " + ", ".join(preferences)
    else:
        answer = "I will remember your preferences."

    return {"messages": [AIMessage(content=answer)]}


builder = StateGraph(ChatState)
builder.add_node("product_agent", product_agent_test)
builder.add_edge(START, "product_agent")
builder.add_edge("product_agent", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "test-123"}}

questions = [
    "I like bananas.",
    "I also like strawberries.",
    "What fruits do I like?"
]

for question in questions:
    print("\nCustomer:", question)

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config
    )

    print("Product Agent:", result["messages"][-1].content)

print("\nMemorySaver test completed successfully!")
