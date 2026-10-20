"""Cartside web app. Run with: streamlit run main.py."""

import os
from html import escape
from pathlib import Path
from uuid import uuid4

import duckdb
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agents.router import build_router
from db.pricing import ensure_pricing_database, quote_products, money, LABEL
from db.bootstrap import ensure_sample_database
from db.queries import DB, get_customer_orders, get_order_products

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Cartside · Your grocery assistant", page_icon=":material/shopping_bag:", layout="wide")
load_dotenv(ROOT / ".env")
# Hosted root-level secrets are loaded into the server environment, never the page.
try:
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "DEMO_MODE"):
        value = st.secrets.get(name)
        if value is not None:
            os.environ[name] = str(value)
except FileNotFoundError:
    pass

st.markdown(f'<style>{(ROOT / "ui/styles.css").read_text()}</style>', unsafe_allow_html=True)

try:
    ensure_sample_database()
    ensure_pricing_database()
except (OSError, duckdb.Error):
    st.error("The sample data couldn't be loaded. Please restart the app or contact the project team.")
    st.stop()

with duckdb.connect(str(DB), read_only=True) as con:
    customer_ids = [row[0] for row in con.execute("SELECT DISTINCT user_id FROM orders ORDER BY user_id").fetchall()]
if not customer_ids:
    st.info("No demo customers are available yet.")
    st.stop()


def reset_chat():
    st.session_state.messages = []
    for key in ("chat_graph", "submitted_question", "reply_mode", "thread_id"):
        st.session_state.pop(key, None)


def start_new_chat():
    reset_chat()
    st.session_state.workspace_page = "Chat"


def submit_question():
    st.session_state.submitted_question = st.session_state.chat_composer


def render_composer():
    st.chat_input("Message Cartside…", key="chat_composer", on_submit=submit_question, max_chars=2000)


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-icon">c</span>Cartside</div>', unsafe_allow_html=True)
    st.button("New chat", icon=":material/add:", width="stretch", on_click=start_new_chat)
    page = st.radio("Workspace", ["Chat", "Orders", "Shopping list"], label_visibility="collapsed", key="workspace_page")
    st.markdown('<div class="sidebar-label">CUSTOMER</div>', unsafe_allow_html=True)
    user_id = st.selectbox("Demo customer", customer_ids,
        index=customer_ids.index(5117) if 5117 in customer_ids else 0,
        format_func=lambda value: f"Customer #{value}")

if st.session_state.get("active_customer") != user_id:
    reset_chat()
    st.session_state.active_customer = user_id
if "chat_graph" not in st.session_state:
    st.session_state.chat_graph = build_router()
    st.session_state.thread_id = str(uuid4())

orders = get_customer_orders(user_id)
with st.sidebar:
    st.markdown(f'<div class="sidebar-stat"><span>Historical orders</span><strong>{len(orders)}</strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-note">Sample workspace<br>Historical data · Simulated refunds</div>', unsafe_allow_html=True)


has_ai = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("DEMO_MODE", "0") != "1"
mode = st.session_state.get("reply_mode", "OpenAI" if has_ai else "Local demo")
# Render only known status strings; never expose exception text or credentials.
status = "Local fallback" if "fallback" in mode else "AI enabled" if mode == "OpenAI" else "Demo mode"
st.markdown(f'<div class="topline"><span class="topline-title">{escape(page)}</span><span class="status"><span class="status-dot"></span>{status}</span></div>', unsafe_allow_html=True)

if page == "Orders":
    st.markdown('<div class="page-heading"><span class="eyebrow">YOUR WORKSPACE</span><h1>Order history</h1><p>A closer look at your past grocery orders.</p></div>', unsafe_allow_html=True)
    st.caption(f"{len(orders)} historical orders · Customer #{user_id}")
    for order in orders:
        with st.expander(f"Order #{order['order_id']} · Purchase {order['order_number']}"):
            products = get_order_products(order["order_id"])
            if products:
                for product in products:
                    st.markdown(f'<div class="order-item"><span>{escape(product["product_name"])}</span><span class="item-meta">{product["add_to_cart_order"]:02d}</span></div>', unsafe_allow_html=True)
                st.caption(f"{len(products)} products recorded")
            else:
                st.caption("Item details aren't included for this order in the sample data.")
    st.stop()

if page == "Shopping list":
    st.markdown('<div class="page-heading"><span class="eyebrow">PLAN YOUR NEXT SHOP</span><h1>Shopping list</h1><p>Keep the essentials in one place.</p></div>', unsafe_allow_html=True)
    # This view is session-only; catalog, agents, and order records stay unchanged.
    with duckdb.connect(str(DB), read_only=True) as con:
        catalog = dict(con.execute("SELECT product_id, product_name FROM products ORDER BY product_name").fetchall())
    lists = st.session_state.setdefault("shopping_lists", {})
    shopping = lists.setdefault(user_id, {"products": [], "checked": {}})
    def save_products():
        shopping["products"] = st.session_state[f"shopping_products_{user_id}"]
    def save_checked(product_id):
        shopping["checked"][product_id] = st.session_state[f"shopping_done_{user_id}_{product_id}"]
    st.session_state[f"shopping_products_{user_id}"] = shopping["products"]
    selected = st.multiselect("Add products", list(catalog), format_func=catalog.get,
                              key=f"shopping_products_{user_id}", on_change=save_products,
                              placeholder="Search the product catalog…")
    quote = quote_products(selected)
    st.caption(LABEL)
    if not selected:
        st.markdown('<div class="list-empty"><h3>Room for your essentials.</h3><p>Search above to add your first product.</p></div>', unsafe_allow_html=True)
    else:
        checked = 0
        with st.container(key="shopping_items"):
            for product_id in selected:
                st.session_state[f"shopping_done_{user_id}_{product_id}"] = shopping["checked"].get(product_id, False)
                checked += st.checkbox(f"{catalog[product_id]} — {money(quote['prices'][product_id])}", key=f"shopping_done_{user_id}_{product_id}",
                                       on_change=save_checked, args=(product_id,))
        st.markdown(f'<div class="list-total"><span>Simulated total · {len(selected)} items</span><strong>{money(quote["total_cents"])}</strong></div>', unsafe_allow_html=True)
        st.caption(f"{checked} of {len(selected)} checked · Remove products using the search field above")
    st.caption("Saved for this browser session. This list doesn't place an order.")
    st.stop()

pending = None
if not st.session_state.messages:
    # Keep the first question beside its context; active chats use the native
    # bottom-pinned Streamlit composer below instead. Never render both.
    with st.container(key="welcome_panel"):
        st.markdown('<div class="welcome"><h1>How can I help?</h1><p>Groceries, orders, and everyday questions.</p></div>', unsafe_allow_html=True)
        render_composer()
        with st.container(key="starters"):
            columns = st.columns(3, gap="small")
            prompts = [
                ("My latest order", ":material/receipt_long:", "What was in my latest order?"),
                ("Find something new", ":material/eco:", "Recommend bananas"),
                ("Returns & refunds", ":material/undo:", "What is your refund policy?"),
            ]
            for column, (label, icon, question) in zip(columns, prompts):
                if column.button(label, icon=icon, width="stretch"):
                    pending = question

question = st.session_state.pop("submitted_question", None) or pending
if question:
    st.session_state.messages.append({"role": "user", "text": question})
    try:
        with st.spinner("Thinking…"):
            result = st.session_state.chat_graph.invoke(
                {"messages": [HumanMessage(content=question)], "user_id": user_id},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
        response = {"text": str(result["messages"][-1].content), "mode": result.get("mode", "Local demo")}
        st.session_state.reply_mode = response["mode"]
    except Exception:
        response = {"text": "I couldn't complete that request. Try again, or start a new chat."}
    st.session_state.messages.append({"role": "assistant", **response})
    # Redraw from state so the empty-state heading disappears immediately.
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=":material/eco:" if message["role"] == "assistant" else ":material/person:"):
        st.write(message["text"])
        if "fallback" in message.get("mode", "").lower():
            st.caption("AI unavailable · using local data")
        if message.get("rows"):
            st.dataframe(message["rows"], hide_index=True, width="stretch")


if st.session_state.messages:
    render_composer()
