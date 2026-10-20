"""Offline integration checks against the included sample database."""
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from agents.order_agent import handle_order_question
from agents.product_agent import find_recommendations
from agents.router import build_router


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('DEMO_MODE', '1')


def ask(graph, question, thread='test', user=5117):
    return graph.invoke({'messages': [HumanMessage(content=question)], 'user_id': user},
                        config={'configurable': {'thread_id': thread}})


def test_order_ownership_and_known_products():
    result = handle_order_question(5117, 'Show order 1662937')
    assert len(result['rows']) == 4
    assert handle_order_question(89, 'Show order 1662937')['rows'] == []
    assert handle_order_question(5117, 'Show order 999999999')['rows'] == []


def test_three_turn_memory_and_new_thread_isolation():
    graph = build_router()
    for question in ['What was in my latest order?', 'Show that again', 'What was in it?']:
        result = ask(graph, question)
        assert '1405149' in result['messages'][-1].content
        assert 'Lemon Yogurt' in result['messages'][-1].content
    assert len(result['messages']) == 6
    fresh = ask(graph, 'What was in it?', thread='new')
    assert '1405149' not in fresh['messages'][-1].content


@pytest.mark.parametrize('question,route', [
    ('Show my order history', 'ORDER_HISTORY'),
    ('Recommend bananas', 'PRODUCT_RECOMMENDATION'),
    ('Refund my last order', 'CUSTOMER_SUPPORT'),
    ('Hello, I want a refund for order 1662937', 'CUSTOMER_SUPPORT'),
    ('hello', 'SMALLTALK'),
])
def test_routes(question, route):
    assert ask(build_router(), question)['route'] == route


def test_support_is_simulated_and_uses_shared_policy():
    answer = ask(build_router(), 'What is the refund policy?')['messages'][-1].content
    assert 'SIMULATED POLICY' in answer and '30 days' in answer
    assert 'BrightCart' not in answer and 'BC-1042' not in answer


def test_recommendations_and_product_followup():
    evidence = find_recommendations('Recommend bananas')
    assert evidence['matched_product'] == 'Bananas'
    assert evidence['recommendations']
    graph = build_router()
    ask(graph, 'Recommend bananas')
    result = ask(graph, 'Suggest something else with those')
    assert 'Bananas' in result['messages'][-1].content


class FakeModel:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def invoke(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError('unavailable')
        return AIMessage(content='CUSTOMER_SUPPORT' if len(self.calls) % 2 else 'Simulated refund policy response')


def test_model_receives_saved_conversation():
    model = FakeModel()
    graph = build_router(model=model)
    ask(graph, 'I want a refund')
    ask(graph, 'It arrived 10 days ago')
    ask(graph, 'It was unused')
    assert any(isinstance(m, HumanMessage) and m.content == 'I want a refund' for m in model.calls[-1])
    assert any(isinstance(m, HumanMessage) and m.content == 'It arrived 10 days ago' for m in model.calls[-1])


def test_api_failure_has_visible_fallback():
    result = ask(build_router(model=FakeModel(fail=True)), 'What is your refund policy?')
    assert 'fallback' in result['mode'].lower()
    assert 'OpenAI is unavailable' in result['messages'][-1].content


def test_streamlit_chat_and_customer_reset():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'main.py').run(timeout=30)
    assert not app.exception
    for question in ['What was in my latest order?', 'Show that again', 'What was in it?']:
        app.chat_input[0].set_value(question).run(timeout=30)
        assert not app.exception
        assert '1405149' in app.chat_message[-1].markdown[0].value
    old_thread = app.session_state['thread_id']
    next(b for b in app.button if 'New chat' in b.label).click().run(timeout=30)
    assert app.session_state['thread_id'] != old_thread
    assert len(app.chat_message) == 1
    app.chat_input[0].set_value('Show order 1662937').run(timeout=30)
    app.selectbox[0].set_value(89).run(timeout=30)
    assert not app.exception
    assert len(app.chat_message) == 1
    app.chat_input[0].set_value('What was in it?').run(timeout=30)
    assert '1662937' not in app.chat_message[-1].markdown[0].value


@pytest.mark.parametrize('invent_number', [False, True])
def test_order_voice_uses_history_and_rejects_invented_details(invent_number):
    class OrderVoice:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            if messages[0].content.startswith('You route'):
                return AIMessage(content='ORDER_HISTORY')
            text = "Sure — order 1405149 had Lemon Yogurt, Blueberry Yoghurt, and Original No Pulp 100% Florida Orange Juice."
            return AIMessage(content=text + (' It arrives in 999 days.' if invent_number else ''))

    model = OrderVoice()
    graph = build_router(model=model)
    ask(graph, 'What was in my latest order?')
    result = ask(graph, 'Show that again')
    text = result['messages'][-1].content
    assert '999' not in text
    assert 'Lemon Yogurt' in text
    assert any(isinstance(m, HumanMessage) and m.content == 'What was in my latest order?' for m in model.calls[-1])
    if not invent_number:
        assert text.startswith('Sure')


def test_explicit_recommendation_overrides_wrong_model_route(monkeypatch):
    from agents import router
    monkeypatch.setattr(router, 'product_turn', lambda *args, **kwargs: {'text': 'Database suggestions for bananas', 'ai_fallback': False})
    result = ask(build_router(model=FakeModel()), 'Recommend bananas')
    assert result['route'] == 'PRODUCT_RECOMMENDATION'
    assert result['messages'][-1].content == 'Database suggestions for bananas'


@pytest.mark.parametrize('question', ['how bout my first order', 'my earliest order', 'the oldest one', 'first-ever order'])
def test_first_order_uses_chronological_sequence(question):
    from db.queries import get_customer_orders, get_order_products
    first = min(get_customer_orders(5117), key=lambda row: row['order_number'])
    result = handle_order_question(5117, question)
    assert str(first['order_id']) in result['text']
    assert result['rows'] == get_order_products(first['order_id'])


def test_switch_latest_to_first_then_follow_up():
    from db.queries import get_customer_orders
    first = min(get_customer_orders(5117), key=lambda row: row['order_number'])['order_id']
    graph = build_router()
    ask(graph, 'What was in my latest order?')
    for question in ['How bout my first order?', 'What was in that first one?', 'Show that again']:
        result = ask(graph, question)
        assert str(first) in result['messages'][-1].content
        assert '1405149' not in result['messages'][-1].content


def test_hosted_first_launch_builds_database_atomically(tmp_path):
    import duckdb
    from db.bootstrap import ensure_sample_database
    database = tmp_path / 'processed/sample.duckdb'
    sample = Path(__file__).resolve().parents[1] / 'data/sample'
    ensure_sample_database(database, sample)
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute('SELECT COUNT(*) FROM orders').fetchone()[0] == 10000
        assert con.execute('SELECT COUNT(*) FROM products').fetchone()[0] == 12207
    modified = database.stat().st_mtime_ns
    ensure_sample_database(database, sample)
    assert database.stat().st_mtime_ns == modified


def test_missing_source_leaves_no_partial_database(tmp_path):
    from db.bootstrap import ensure_sample_database
    database = tmp_path / 'processed/sample.duckdb'
    with pytest.raises(FileNotFoundError):
        ensure_sample_database(database, tmp_path / 'missing')
    assert not database.exists()
