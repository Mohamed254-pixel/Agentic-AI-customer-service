"""Instacart Product Recommendation Agent.

Uses the team's existing DuckDB tables and an OpenAI API key stored in .env.
No changes to the order, refund, or router agents are required.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import duckdb
from dotenv import load_dotenv
from openai import OpenAI
from agents.voice import CHAT_STYLE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "processed" / "instacart_sample.duckdb"
SYSTEM_PROMPT = """You are the Product Recommendation Agent for a student-built
Instacart customer service demo. Help customers discover grocery products.
Use ONLY the provided database results as product recommendations. Do not
invent products, prices, availability, nutrition claims, or purchase histories.
The Instacart data is historical and not live inventory. Give concise, friendly
reasons for suggestions. If no matches are found, say so and ask for another
product. Do not handle order tracking or refund requests; those belong to
other agents. Never claim a purchase or refund has been completed."""


def _database_path() -> Path:
    configured = os.getenv("INSTACART_DB_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_DB


def _find_product(con: duckdb.DuckDBPyConnection, question: str) -> dict[str, Any] | None:
    """Match an exact product name or the longest product-name phrase in the question."""
    rows = con.execute("SELECT product_id, product_name, aisle_id FROM products").fetchall()
    normalized = re.sub(r"[^a-z0-9 ]", " ", question.lower())
    normalized = " " + " ".join(normalized.split()) + " "
    matches = []
    for product_id, name, aisle_id in rows:
        term = " ".join(re.sub(r"[^a-z0-9 ]", " ", name.lower()).split())
        if len(term) >= 3 and f" {term} " in normalized:
            matches.append((len(term), product_id, name, aisle_id))
    if not matches:
        # Fallback: single-word product queries, e.g. "recommend bananas".
        tokens = set(normalized.split()) - {"recommend", "products", "product", "grocery", "groceries", "similar", "suggest", "some", "what", "with", "for", "the", "and", "can", "you", "me", "buy", "other", "like", "please", "based", "on", "my", "to"}
        for product_id, name, aisle_id in rows:
            words = set(re.sub(r"[^a-z0-9 ]", " ", name.lower()).split())
            if tokens & words:
                matches.append((len(tokens & words), product_id, name, aisle_id))
    if not matches:
        return None
    _, product_id, name, aisle_id = max(matches, key=lambda x: (x[0], -len(x[2])))
    return {"product_id": product_id, "product_name": name, "aisle_id": aisle_id}


def find_recommendations(question: str, limit: int = 5) -> dict[str, Any]:
    """Find products co-purchased with the named item; fall back to its aisle."""
    db_path = _database_path()
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Database not found: {db_path}. Run: python scripts/build_db.py --dataset sample"
        )
    with duckdb.connect(str(db_path), read_only=True) as con:
        seed = _find_product(con, question)
        if seed is None:
            return {"matched_product": None, "recommendations": [], "method": "no_match"}
        # The prior/train tables contain historical order baskets, not live stock.
        results = con.execute("""
            WITH baskets AS (
                SELECT order_id, product_id FROM order_products__prior
                UNION ALL
                SELECT order_id, product_id FROM order_products__train
            ), seed_orders AS (
                SELECT DISTINCT order_id FROM baskets WHERE product_id = ?
            )
            SELECT p.product_id, p.product_name, COUNT(DISTINCT b.order_id) AS shared_orders
            FROM seed_orders s
            JOIN baskets b ON s.order_id = b.order_id
            JOIN products p ON p.product_id = b.product_id
            WHERE b.product_id <> ?
            GROUP BY p.product_id, p.product_name
            ORDER BY shared_orders DESC, p.product_name
            LIMIT ?
        """, [seed["product_id"], seed["product_id"], limit]).fetchall()
        method = "frequently_bought_together"
        if not results:
            results = con.execute("""
                SELECT product_id, product_name, 0 AS shared_orders
                FROM products WHERE aisle_id = ? AND product_id <> ?
                ORDER BY product_name LIMIT ?
            """, [seed["aisle_id"], seed["product_id"], limit]).fetchall()
            method = "same_aisle"
    return {
        "matched_product": seed["product_name"],
        "recommendations": [
            {"product_id": pid, "product_name": name, "shared_orders": count}
            for pid, name, count in results
        ],
        "method": method,
    }


def recommend_products(question: str, use_ai: bool = True) -> str:
    """Public entry point for local testing and future router integration."""
    result = find_recommendations(question)
    if not result["matched_product"]:
        return "I couldn't identify a matching product in our sample data. What grocery item would you like recommendations for?"
    if not result["recommendations"]:
        return f"I found {result['matched_product']}, but no related products in our sample data. Try another grocery item."
    items = result["recommendations"]
    lines = [f"- {p['product_name']}" for p in items]
    evidence = "Frequently purchased in the same historical orders" if result["method"] == "frequently_bought_together" else "Other products in the same grocery aisle"
    fallback = f"For {result['matched_product']}, here are suggestions ({evidence.lower()}):\n" + "\n".join(lines)
    if not use_ai:
        return fallback
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key or os.getenv("DEMO_MODE", "0") == "1":
        return fallback + "\n\n(AI explanation unavailable: set OPENAI_API_KEY in your .env file.)"
    try:
        client = OpenAI(api_key=key, timeout=20, max_retries=1)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": CHAT_STYLE + "\n" + SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Customer question: {question}\nMatched product: {result['matched_product']}\n"
                    f"Evidence type: {evidence}\nAllowed suggestions:\n" + "\n".join(lines)
                    + "\nGive a short natural introduction and a compact list. Explain that the evidence "
                    "is the exact evidence type supplied (co-purchases OR shared aisle); never claim both. Don't invent pairings, "
                    "taste descriptions, nutrition, or reasons unsupported by this evidence."
                )},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or fallback
    except Exception as exc:
        # Preserve useful recommendations when the network or API fails.
        return fallback + f"\n\n(AI explanation unavailable: {type(exc).__name__}.)"


def product_agent(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph-friendly node: reads the latest user message and returns a reply.

    The group's router should merge this output into its own state schema.
    """
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        question = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
    else:
        question = state.get("question", "")
    answer = recommend_products(str(question))
    return {"messages": [{"role": "assistant", "content": answer}]}


if __name__ == "__main__":
    print("Instacart Product Recommendation Agent (type 'exit' to quit)")
    while True:
        user_question = input("You: ").strip()
        if user_question.lower() in {"exit", "quit"}:
            break
        if user_question:
            print("Product Agent:", recommend_products(user_question), "\n")
