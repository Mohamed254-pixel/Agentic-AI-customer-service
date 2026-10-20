"""Run with: streamlit run main.py"""

from uuid import uuid4

import duckdb
import streamlit as st

from db.queries import DB, get_customer_orders, get_order_products
from agents.router import build_router
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Cartside · Customer care", page_icon="🛒", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"] {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(ellipse at 75% 0%,#292929 0%,#141414 52%);color:#f1f1ef;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#191919;border-right:1px solid #303030;}
.block-container {max-width:1040px;padding-top:3.8rem;padding-bottom:3rem;}
h1,h2,h3,h4 {color:#f1f1ef;letter-spacing:-.035em;}
.hero {font-size:clamp(30px,4vw,46px);font-weight:600;line-height:1.2;letter-spacing:-1.8px;margin:0 0 12px;}
.badge {display:inline-block;border:1px solid #465448;background:#252e27;border-radius:30px;padding:6px 13px;font-size:12px;color:#b8cfae;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:#aaaaaa;}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label {color:#dededb;}
[data-testid="stChatMessage"] {background:#222222;border:1px solid #393939;border-radius:16px;padding:20px;margin-top:12px;}
.stButton>button {border-radius:12px;border:1px solid #414141;background:#252525;color:#e0e0dd;min-height:44px;transition:background .15s,border-color .15s;}
.stButton>button:hover {border-color:#9eb695;background:#293029;color:#e3eddf;}
.stButton>button:focus:not(:active) {border-color:#9eb695;color:#d7e5d1;}
[data-testid="stMetric"] {background:linear-gradient(135deg,#2b2b2b,#212121);border:1px solid #3b3b3b;border-radius:16px;padding:20px 22px;}
[data-testid="stMetricLabel"] {color:#b0b0ad;font-size:13px;}
[data-testid="stMetricValue"] {color:#f1f1ed;font-size:30px;font-weight:500;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#393939!important;border-radius:16px!important;}
[data-testid="stChatInput"] {border:1px solid #494949;border-radius:14px;background:#272727;}
[data-testid="stChatInput"]:focus-within {border-color:#9eb695;}
[data-testid="stChatMessageAvatarAssistant"] {background:#344036;color:#c2d7b8;}
[data-testid="stChatMessageAvatarUser"] {background:#383838;}
@media(max-width:640px) {.block-container {padding-top:2rem;} [data-testid="stMetric"] {padding:12px;} [data-testid="stMetricValue"] {font-size:24px;}}
</style>
""", unsafe_allow_html=True)

if not DB.exists():
    st.title("Let's get your demo ready")
    st.info("Build the sample database once, then reload this page.")
    st.code("python scripts/build_db.py --dataset sample", language="bash")
    st.stop()

with duckdb.connect(str(DB), read_only=True) as con:
    customer_ids = [row[0] for row in con.execute("SELECT DISTINCT user_id FROM orders ORDER BY user_id").fetchall()]

with st.sidebar:
    st.markdown("## 🛒 Cartside")
    st.divider()
    user_id = st.selectbox("Customer", customer_ids, index=customer_ids.index(5117) if 5117 in customer_ids else 0, format_func=lambda value: f"Customer #{value}")
    st.divider()
    if st.button("＋ New chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.pop("chat_graph", None)
        st.session_state.pop("submitted_question", None)
        st.session_state.pop("reply_mode", None)
    st.caption("Instacart sample · Demo accounts")

if st.session_state.get("active_customer") != user_id:
    st.session_state.active_customer = user_id
    st.session_state.messages = []
    st.session_state.pop("chat_graph", None)
    st.session_state.pop("submitted_question", None)
    st.session_state.pop("reply_mode", None)

if "chat_graph" not in st.session_state:
    st.session_state.chat_graph = build_router()
    st.session_state.thread_id = str(uuid4())

title, badge = st.columns([4, 1])
with title:
    st.markdown('<div class="hero">How can we help?</div>', unsafe_allow_html=True)
with badge:
    st.markdown('<span class="badge">● Preview</span>', unsafe_allow_html=True)
st.caption("Historical sample · " + st.session_state.get("reply_mode", "Local demo / OpenAI when configured"))
st.write("")

orders = get_customer_orders(user_id)
latest_items = get_order_products(orders[0]["order_id"]) if orders else []
left, middle, right = st.columns(3)
left.metric("Sample orders", len(orders))
middle.metric("Latest order", str(orders[0]["order_id"]) if orders else "—")
right.metric("Products", len(latest_items))
st.write("")

st.subheader("Chat")
suggestions = ["Show my order history", "What was in my latest order?"]
pending = None
buttons = st.columns(2)
for column, suggestion in zip(buttons, suggestions):
    if column.button(suggestion, width="stretch"):
        pending = suggestion
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🛒"):
        st.write("Ask about your orders.")
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
        st.session_state.reply_mode = result.get("mode", "Local demo")
    except Exception:
        response = {"text": "I couldn't complete that request. Please try again or start a new chat."}
    st.session_state.messages.append({"role": "assistant", **response})
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🛒" if message["role"] == "assistant" else "👤"):
        st.write(message["text"])
        if message.get("mode"):
            st.caption(message["mode"])
        if message.get("rows"):
            st.dataframe(message["rows"], hide_index=True, width="stretch")


def submit_question():
    st.session_state.submitted_question = st.session_state.chat_composer


st.chat_input("Ask about your orders…", key="chat_composer", on_submit=submit_question)

with st.sidebar:
    with st.expander("Order history"):
        st.dataframe(
            [{"Order": o["order_id"], "Sequence": o["order_number"]} for o in orders],
            hide_index=True,
            width="stretch",
        )
