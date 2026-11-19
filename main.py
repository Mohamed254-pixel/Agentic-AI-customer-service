"""Cartside grocery workspace. Run with: streamlit run main.py."""

import os
from html import escape
from pathlib import Path
from uuid import uuid4

import duckdb
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agents.router import build_router
from db.bootstrap import ensure_sample_database
from db.pricing import LABEL, ensure_pricing_database, money, quote_products
from db.queries import DB, get_customer_order_items, get_customer_orders
from ui.recommendations import recommendation_display, recommendation_rows_html

ROOT = Path(__file__).resolve().parent
PROMPTS = (
    ("My latest order", "What was in my latest order?"),
    ("Recommend products based on my purchase history.", "Recommend products based on my purchase history."),
    ("How can I get help with an order?", "How can I get help with an order?"),
)
PAGE_NAMES = {"Chat": "Overview", "Orders": "Orders", "Shopping list": "Shopping list"}

st.set_page_config(page_title="Cartside · Grocery customer care", page_icon=":material/shopping_basket:", layout="wide")
load_dotenv(ROOT / ".env")
# Live mode requires an explicit local-only listener and an explicit opt-in.
has_ai = (st.get_option("server.address") in {"127.0.0.1", "::1", "localhost"}
          and os.getenv("DEMO_MODE") == "0" and bool(os.getenv("OPENAI_API_KEY")))
st.html(ROOT / "ui/styles.css")


@st.cache_data
def initialize_workspace():
    ensure_sample_database()
    ensure_pricing_database()
    with duckdb.connect(str(DB), read_only=True) as con:
        return [row[0] for row in con.execute(
            "SELECT DISTINCT user_id FROM orders ORDER BY user_id"
        ).fetchall()]


def reset_chat():
    st.session_state.messages = []
    for key in ("chat_graph", "submitted_question", "reply_mode", "thread_id"):
        st.session_state.pop(key, None)


def open_page(page):
    st.session_state.workspace_page = page


def start_new_chat():
    reset_chat()
    open_page("Chat")


def submit_question():
    st.session_state.submitted_question = st.session_state.chat_composer


def use_starter(question):
    st.session_state.submitted_question = question


def render_header(customer_id, status):
    st.markdown(
        '<div class="cartside-header">'
        '<div class="cartside-wordmark"><strong>Cartside</strong><span>Grocery customer care</span></div>'
        f'<div class="cartside-account"><span>Customer #{customer_id}</span>'
        f'<span class="cartside-status">{escape(status)}</span></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Portfolio demo · Historical Instacart data · Simulated support policies and prices")


def page_heading(title, description):
    st.title(title, anchor=False)
    st.markdown(f'<p class="cartside-intro">{escape(description)}</p>', unsafe_allow_html=True)


def render_overview(customer_id, orders):
    with st.container(key="overview_panel", gap="small"):
        st.subheader("Customer overview", anchor=False)
        st.markdown(
            '<dl class="cartside-facts">'
            f'<div><dt>Sample customer</dt><dd>#{customer_id}</dd></div>'
            f'<div><dt>Historical orders</dt><dd>{len(orders):,}</dd></div>'
            '</dl>',
            unsafe_allow_html=True,
        )
        if orders:
            latest = orders[0]
            st.markdown(
                '<div class="cartside-latest">'
                '<span>Latest recorded order</span>'
                f'<strong>Order #{latest["order_id"]}</strong>'
                f'<small>Purchase {latest["order_number"]} in the order sequence</small></div>',
                unsafe_allow_html=True,
            )
            st.button("View order history", key="overview_orders", width="stretch",
                      on_click=open_page, args=("Orders",))
        else:
            st.info("This customer has no orders in the sample.")


def render_orders(customer_id, orders):
    page_heading("Order history", f"{len(orders):,} historical orders for customer #{customer_id}. Expand an order to see its recorded products.")
    if not orders:
        st.info("This customer has no orders in the sample.")
        return
    try:
        items_by_order = get_customer_order_items(customer_id)
    except duckdb.Error:
        st.error("Order items are temporarily unavailable. Reload the page to try again.")
        return
    with st.container(key="order_history", gap="small"):
        for order in orders:
            products = items_by_order.get(order["order_id"], [])
            count = f"{len(products)} {'item' if len(products) == 1 else 'items'}" if products else "Basket unavailable"
            label = f"Order #{order['order_id']} · Purchase {order['order_number']} · {count}"
            with st.expander(label):
                if products:
                    names = "".join(f"<li>{escape(product['product_name'])}</li>" for product in products)
                    st.markdown(f'<ol class="cartside-basket">{names}</ol>', unsafe_allow_html=True)
                else:
                    st.caption("Item details aren't included for this order in the sample data.")


def render_shopping_list(customer_id):
    page_heading("Shopping list", "Save groceries for your next shop and see a simulated total.")
    try:
        with duckdb.connect(str(DB), read_only=True) as con:
            catalog = dict(con.execute(
                "SELECT product_id, product_name FROM products ORDER BY product_name, product_id"
            ).fetchall())
    except duckdb.Error:
        st.error("The shopping catalog is temporarily unavailable. Reload the page to try again.")
        return

    lists = st.session_state.setdefault("shopping_lists", {})
    shopping = lists.setdefault(customer_id, {"products": [], "checked": {}})
    products_key = f"shopping_products_{customer_id}"

    def save_products():
        shopping["products"] = st.session_state[products_key]

    def save_checked(product_id, key):
        shopping["checked"][product_id] = st.session_state[key]

    with st.container(key="shopping_layout"):
        list_column, summary_column = st.columns([2, 1], gap="large")
        with list_column, st.container(key="shopping_panel", gap="medium"):
            st.session_state[products_key] = shopping["products"]
            selected = st.multiselect(
                "Add or remove products", list(catalog), format_func=catalog.get,
                key=products_key, on_change=save_products, placeholder="Search the grocery catalog…",
            )
            try:
                quote = quote_products(selected)
            except (ValueError, duckdb.Error):
                st.error("Simulated pricing is temporarily unavailable. Reload the page to try again.")
                return
            checked = 0
            if selected:
                st.subheader("Your items", anchor=False)
                with st.container(key="shopping_items", gap="small"):
                    for product_id in selected:
                        key = f"shopping_done_{customer_id}_{product_id}"
                        st.session_state[key] = shopping["checked"].get(product_id, False)
                        price = money(quote["prices"][product_id])
                        checked += st.checkbox(f"{catalog[product_id]} — {price}", key=key,
                                               on_change=save_checked, args=(product_id, key))
            else:
                st.info("Your list is empty. Search the catalog above to add a grocery item.")
        with summary_column, st.container(key="shopping_summary", gap="medium"):
            st.subheader("List summary", anchor=False)
            st.markdown(
                '<dl class="cartside-facts">'
                f'<div><dt>Items checked</dt><dd>{checked} of {len(selected)}</dd></div>'
                f'<div><dt>Simulated total</dt><dd>{money(quote["total_cents"])}</dd></div></dl>',
                unsafe_allow_html=True,
            )
            st.caption(LABEL)
            st.caption("Saved for this browser session. This list doesn't place an order.")


def render_assistant(customer_id):
    with st.container(key="assistant_panel", gap="small"):
        st.subheader("Shopping assistant", anchor=False)
        if not st.session_state.messages:
            st.caption("Try one of these questions or type your own.")
            with st.container(key="starters", gap="small"):
                for label, question in PROMPTS:
                    st.button(label, width="stretch", help=question,
                              on_click=use_starter, args=(question,))

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                speaker = "You" if message["role"] == "user" else "Cartside"
                st.html(f'<span class="cartside-speaker">{speaker}</span>')
                st.write(message.get("summary", message["text"]))
                if message.get("products"):
                    st.html(recommendation_rows_html(message["products"]))
                if "fallback" in message.get("mode", "").lower():
                    st.caption("AI unavailable · using local data")

        question = st.session_state.pop("submitted_question", "")
        if question.strip():
            st.session_state.messages.append({"role": "user", "text": question})
            try:
                with st.spinner("Finding an answer…", show_time=False):
                    result = st.session_state.chat_graph.invoke(
                        {"messages": [HumanMessage(content=question)], "user_id": customer_id},
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                    )
                text = str(result["messages"][-1].content)
                response = {"text": text, "mode": result.get("mode", "Local demo")}
                summary, products = recommendation_display(result, text, DB)
                if products:
                    response.update(summary=summary, products=products)
                st.session_state.reply_mode = response["mode"]
            except Exception:
                response = {"text": "I couldn't complete that request. Try again, or start a new chat."}
            st.session_state.messages.append({"role": "assistant", **response})
            st.rerun()

        if len(st.session_state.messages) < 50:
            st.chat_input("Ask about your groceries…", key="chat_composer", on_submit=submit_question, max_chars=2000)
        else:
            st.info("Start a new chat to continue. This conversation has reached 25 questions.")


try:
    customer_ids = initialize_workspace()
except (OSError, ValueError, duckdb.Error):
    st.error("The sample data couldn't be loaded. The app owner can check the deployment logs and restart the app.")
    st.stop()

if not customer_ids:
    st.info("No demo customers are available yet.")
    st.stop()

with st.sidebar:
    st.markdown('<p class="cartside-sidebar-brand">Cartside</p>', unsafe_allow_html=True)
    page = st.radio("Workspace", PAGE_NAMES, format_func=PAGE_NAMES.__getitem__, key="workspace_page")
    st.divider()
    user_id = st.selectbox("Sample customer", customer_ids,
                           index=customer_ids.index(5117) if 5117 in customer_ids else 0,
                           format_func=lambda value: f"Customer #{value}")
    st.button("New chat", width="stretch", on_click=start_new_chat)

if st.session_state.get("active_customer") != user_id:
    reset_chat()
    st.session_state.active_customer = user_id
if "chat_graph" not in st.session_state:
    st.session_state.chat_graph = build_router(allow_openai=has_ai)
    st.session_state.thread_id = str(uuid4())

mode = st.session_state.get("reply_mode", "OpenAI" if has_ai else "Local demo")
status = "Local fallback" if "fallback" in mode else "AI enabled" if mode == "OpenAI" else "Demo mode"
render_header(user_id, status)

try:
    orders = get_customer_orders(user_id)
except duckdb.Error:
    st.error("The order history is temporarily unavailable. Reload the page to try again.")
    st.stop()

with st.sidebar:
    st.caption(f"{len(orders):,} historical orders")

if page == "Orders":
    render_orders(user_id, orders)
elif page == "Shopping list":
    render_shopping_list(user_id)
else:
    page_heading("Your groceries, all in one place.",
                 "Review past orders, discover products, and get help with your shopping.")
    with st.container(key="customer_workspace"):
        overview, assistant = st.columns([1, 2], gap="large")
        with overview:
            render_overview(user_id, orders)
        with assistant:
            render_assistant(user_id)
