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
