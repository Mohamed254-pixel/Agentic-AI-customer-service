"""Cartside web app. Run with: streamlit run main.py."""

import os
from pathlib import Path
from uuid import uuid4

import duckdb
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agents.router import build_router
from db.bootstrap import ensure_sample_database
from db.queries import DB, get_customer_orders

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Cartside · Your grocery assistant", page_icon="✳", layout="wide")
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


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-icon">c</span>Cartside</div>', unsafe_allow_html=True)
    st.button("New chat", icon=":material/add:", width="stretch", on_click=reset_chat)
    st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
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
    with st.expander("Order history", icon=":material/history:"):
        st.dataframe([{"Order": o["order_id"], "Sequence": o["order_number"]} for o in orders], hide_index=True, width="stretch")
    st.markdown('<div class="sidebar-note">Instacart sample data<br>Refund policies are simulated.</div>', unsafe_allow_html=True)

has_ai = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("DEMO_MODE", "0") != "1"
mode = st.session_state.get("reply_mode", "OpenAI" if has_ai else "Local demo")
# Render only known status strings; never expose exception text or credentials.
status = "Local fallback" if "fallback" in mode else "AI enabled" if mode == "OpenAI" else "Demo mode"
st.markdown(f'<div class="topline"><span class="topline-title">Your grocery assistant</span><span class="status"><span class="status-dot"></span>{status}</span></div>', unsafe_allow_html=True)

pending = None
if not st.session_state.messages:
    st.markdown('<div class="welcome"><div class="welcome-mark">✳</div><h1>A little help with<br>your everyday groceries.</h1><p>Explore your orders. Find your next favorite.</p></div>', unsafe_allow_html=True)
    with st.container(key="starters"):
        columns = st.columns(3)
        prompts = [
            ("My latest order", ":material/receipt_long:", "What was in my latest order?"),
            ("Find something new", ":material/eco:", "Recommend bananas"),
            ("Returns & refunds", ":material/undo:", "What is your refund policy?"),
        ]
        for column, (label, icon, question) in zip(columns, prompts):
            if column.button(label, icon=icon, width="stretch"):
                pending = question
    # One short welcome preserves the same accessible chat structure as replies.
    with st.chat_message("assistant", avatar=":material/eco:"):
        st.write("What can I help you with?")

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


def submit_question():
    st.session_state.submitted_question = st.session_state.chat_composer


st.chat_input("Ask Cartside anything about your groceries…", key="chat_composer", on_submit=submit_question, max_chars=2000)
