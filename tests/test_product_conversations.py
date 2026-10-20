"""Multi-turn regressions for routing, exclusions, provenance and isolation."""
import json
from pathlib import Path

import duckdb
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.product_agent import find_recommendations, parse_exclusions, product_turn
from agents.router import build_router


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('DEMO_MODE', '1')


class MisleadingRouter:
    """Always misroutes to support; product rules must still win."""
    def invoke(self, messages):
        if messages[0].content.startswith('You route'):
            return AIMessage(content='CUSTOMER_SUPPORT')
        try:
            payload = json.loads(messages[-1].content)
            return AIMessage(content=json.dumps({'product_ids': [p['product_id'] for p in payload['candidates'][:3]]}))
        except (ValueError, KeyError):
            return AIMessage(content='This is a simulated refund policy; actual refunds require human review.')


def ask(graph, question, thread='product-test'):
    return graph.invoke({'messages': [HumanMessage(content=question)], 'user_id': 5117},
                        config={'configurable': {'thread_id': thread}})


def assert_verified(result, seen=()):
    items = result['recommendations']
    assert items
    ids = {p['product_id'] for p in items}
    assert ids.isdisjoint(seen)
    from agents.product_agent import _database_path
    with duckdb.connect(str(_database_path()), read_only=True) as con:
        actual = dict(con.execute('SELECT product_id, product_name FROM products').fetchall())
    for product in items:
        assert actual[product['product_id']] == product['product_name']
        assert '- ' + product['product_name'] in result['messages'][-1].content
    return ids


@pytest.mark.parametrize('with_model', [False, True])
def test_slang_exclusions_and_topic_changes(with_model):
    graph = build_router(model=MisleadingRouter() if with_model else None)
    seen = set()
    for question in ['Recommend bananas', 'nah give me sum else', 'no milk and bananas',
                     'gimme smth else', 'not those', 'those are the wrong ones']:
        result = ask(graph, question)
        assert result['route'] == 'PRODUCT_RECOMMENDATION'
        assert 'refund' not in result['messages'][-1].content.lower()
        ids = assert_verified(result, seen)
        seen |= ids
        assert set(result['suggested_product_ids']) == seen
        if question in ['no milk and bananas', 'gimme smth else', 'not those', 'those are the wrong ones']:
            assert set(result['excluded_products']) == {'milk', 'bananas'}
            for p in result['recommendations']:
                assert 'milk' not in p['product_name'].lower()
                assert 'banana' not in p['product_name'].lower()
    thanks = ask(graph, 'thanks')
    assert thanks['route'] == 'SMALLTALK'
    assert thanks['active_route'] == 'PRODUCT_RECOMMENDATION'
    result = ask(graph, 'what about apples?')
    assert result['route'] == 'PRODUCT_RECOMMENDATION'
    seen |= assert_verified(result, seen)
    assert result['excluded_products'] == ['milk', 'bananas']
    refund = ask(graph, 'I want a refund for my last order')
    assert refund['route'] == 'CUSTOMER_SUPPORT'
    result = ask(graph, 'Suggest something else')
    assert result['route'] == 'PRODUCT_RECOMMENDATION'
    assert_verified(result, seen)
    assert result['excluded_products'] == ['milk', 'bananas']


@pytest.mark.parametrize('text', ["no milk and bananas", "without milk or bananas", "don't want milk, bananas", "skip milk and bananas please"])
def test_negative_phrases_are_not_positive_seeds(text):
    positive, exclusions = parse_exclusions(text)
    assert set(exclusions) == {'milk', 'bananas'}
    assert 'milk' not in positive and 'bananas' not in positive


def test_explicit_new_product_does_not_replay_original_seed():
    graph = build_router()
    ask(graph, 'Recommend bananas')
    result = ask(graph, 'no milk and bananas but recommend apples')
    assert result['route'] == 'PRODUCT_RECOMMENDATION'
    assert result['seed_product_id'] is not None
    assert result['excluded_products'] == ['milk', 'bananas']
    assert 'Apples' in result['messages'][-1].content
    for product in result['recommendations']:
        assert not any(term in product['product_name'].lower() for term in ('milk', 'banana'))


def test_unknown_new_product_asks_instead_of_reusing_old_seed():
    graph = build_router()
    ask(graph, 'Recommend bananas')
    result = ask(graph, 'Recommend xyzunfindable987')
    assert result['recommendations'] == []
    assert 'match' in result['messages'][-1].content


def test_fresh_thread_has_no_exclusions_or_seen_products():
    graph = build_router()
    first = ask(graph, 'Recommend bananas', thread='first')
    ask(graph, 'no milk and bananas', thread='first')
    second = ask(graph, 'Recommend bananas', thread='second')
    assert second['excluded_products'] == []
    assert first['last_suggested_product_ids'] == second['last_suggested_product_ids']


@pytest.fixture
def tiny_catalog(tmp_path, monkeypatch):
    path = tmp_path / 'products.duckdb'
    with duckdb.connect(str(path)) as con:
        con.execute('CREATE TABLE products(product_id INTEGER, product_name VARCHAR, aisle_id INTEGER)')
        con.executemany('INSERT INTO products VALUES (?,?,?)', [
            (1, 'Banana', 1), (2, 'Whole Milk', 2), (3, 'Almondmilk', 2), (4, 'Apple', 1),
            (5, 'Cookies', 3), (6, 'Strawberry Milk', 2), (7, 'Bananas Snack', 1), (8, 'Pear', 1), (9, 'Oats', 3)])
        con.execute('CREATE TABLE order_products__prior(order_id INTEGER, product_id INTEGER)')
        con.executemany('INSERT INTO order_products__prior VALUES (?,?)',
                        [(1, i) for i in range(1, 10)] + [(2, 1), (2, 2), (2, 3), (2, 6), (2, 7)])
        con.execute('CREATE TABLE order_products__train AS SELECT * FROM order_products__prior WHERE FALSE')
    monkeypatch.setenv('INSTACART_DB_PATH', str(path))
    return path


def test_filter_before_limit_and_excluded_seed(tiny_catalog):
    result = find_recommendations('Recommend Banana', limit=3, excluded_products=['milk', 'bananas'],
                                  previously_suggested_ids=[5])
    assert result['seed_product_id'] is None
    assert {p['product_id'] for p in result['recommendations']} == {4, 8, 9}


def test_exhaustion_does_not_repeat_or_clear_constraints(tiny_catalog):
    result = product_turn('nah something else', {'suggested_product_ids': list(range(1, 10)),
        'excluded_products': ['milk'], 'seed_product_id': 1}, use_ai=False)
    assert result['recommendations'] == []
    assert 'run out' in result['text']
    assert result['excluded_products'] == ['milk']
    assert result['suggested_product_ids'] == list(range(1, 10))


@pytest.mark.parametrize('invalid', ['invented', 'excluded', 'duplicate', 'prose', 'empty', 'boolean'])
def test_unverified_model_output_never_reaches_recommendations(tiny_catalog, invalid):
    class UnsafeModel:
        def invoke(self, messages):
            candidates = json.loads(messages[-1].content)['candidates']
            valid = candidates[0]['product_id']
            choices = {'invented': [999999], 'excluded': [2], 'duplicate': [valid, valid],
                       'empty': [], 'boolean': [True]}
            content = 'Buy Martian Milk for $1' if invalid == 'prose' else json.dumps(
                {'product_ids': choices[invalid], 'opening': 'Buy Martian Milk for $1'})
            return AIMessage(content=content)
    result = product_turn('Recommend groceries without milk', model=UnsafeModel())
    assert result['ai_fallback']
    assert result['recommendations']
    assert all(p['product_id'] not in {2, 3, 6} for p in result['recommendations'])
    assert 'Martian' not in result['text'] and '$1' not in result['text']


def test_even_valid_model_ids_cannot_inject_extra_products(tiny_catalog):
    class ExtraProse:
        def invoke(self, messages):
            candidate = json.loads(messages[-1].content)['candidates'][0]
            return AIMessage(content=json.dumps({'product_ids': [candidate['product_id']],
                                                'text': 'Buy an invented product for $10'}))
    result = product_turn('Recommend groceries', model=ExtraProse())
    assert not result['ai_fallback']
    assert len(result['recommendations']) == 1
    assert 'invented' not in result['text'] and '$10' not in result['text']


def test_exact_product_name_has_priority_over_plural_alias():
    assert find_recommendations('Recommend bananas')['matched_product'] == 'Bananas'


@pytest.mark.parametrize('question', ['anything else', 'give me different ones', "nah that ain't it", 'nope', 'more options please', 'nah gimme sumthin else fam pls'])
def test_more_casual_rejections_keep_product_topic(question):
    graph = build_router(model=MisleadingRouter())
    first = ask(graph, 'Recommend bananas')
    result = ask(graph, question)
    assert result['route'] == 'PRODUCT_RECOMMENDATION'
    assert_verified(result, first['suggested_product_ids'])


def test_bare_product_name_after_clarification():
    graph = build_router()
    ask(graph, 'Recommend xyzunknown')
    result = ask(graph, 'apples please')
    assert result['route'] == 'PRODUCT_RECOMMENDATION'
    assert_verified(result)


def test_price_question_does_not_invent_prices_or_consume_suggestions():
    graph = build_router()
    first = ask(graph, 'Recommend bananas')
    result = ask(graph, 'How much do those cost?')
    assert result['route'] == 'PRODUCT_RECOMMENDATION'
    assert "don't have live prices" in result['messages'][-1].content
    assert result['suggested_product_ids'] == first['suggested_product_ids']
    assert result['last_suggested_product_ids'] == first['last_suggested_product_ids']


def test_general_conversation_can_leave_and_return_to_product_topic():
    class ChatModel(MisleadingRouter):
        def invoke(self, messages):
            if messages[0].content.startswith('Speak as Cartside') and 'answer general questions' in messages[0].content:
                return AIMessage(content='Python is a programming language. You can use it to build apps and work with data.')
            return super().invoke(messages)
    graph = build_router(model=ChatModel())
    first = ask(graph, 'Recommend bananas')
    ask(graph, 'no milk and bananas')
    general = ask(graph, 'What is Python?')
    assert general['route'] == 'SMALLTALK'
    assert general['active_route'] == 'SMALLTALK'
    assert 'programming language' in general['messages'][-1].content
    assert 'refund' not in general['messages'][-1].content
    followup = ask(graph, 'Tell me more')
    assert followup['route'] == 'SMALLTALK'
    assert 'programming language' in followup['messages'][-1].content
    result = ask(graph, 'Recommend apples')
    assert result['route'] == 'PRODUCT_RECOMMENDATION'
    assert result['excluded_products'] == ['milk', 'bananas']
    assert_verified(result, first['suggested_product_ids'])


def test_demo_mode_is_honest_about_general_questions():
    result = ask(build_router(), 'Explain photosynthesis')
    assert result['route'] == 'SMALLTALK'
    assert 'demo mode' in result['messages'][-1].content
    assert 'policy' not in result['messages'][-1].content.lower()


def test_product_memory_resets_with_ui_new_chat_and_customer_switch():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'main.py').run(timeout=30)
    for text in ['Recommend bananas', 'no milk and bananas']:
        app.chat_input[0].set_value(text).run(timeout=30)
        assert not app.exception
    def state():
        graph = app.session_state['chat_graph']
        return graph.get_state({'configurable': {'thread_id': app.session_state['thread_id']}}).values
    assert state()['excluded_products'] == ['milk', 'bananas']
    old_ids = state()['suggested_product_ids']
    next(b for b in app.button if 'New chat' in b.label).click().run(timeout=30)
    assert not state()
    app.chat_input[0].set_value('Recommend bananas').run(timeout=30)
    assert state()['excluded_products'] == []
    assert set(state()['suggested_product_ids']) & set(old_ids)
    app.chat_input[0].set_value('no milk and bananas').run(timeout=30)
    app.selectbox[0].set_value(89).run(timeout=30)
    assert not state()
    app.chat_input[0].set_value('Recommend bananas').run(timeout=30)
    assert not app.exception
    assert state()['excluded_products'] == []


def test_real_order_request_leaves_recommendations():
    graph = build_router(model=MisleadingRouter())
    ask(graph, 'Recommend bananas')
    result = ask(graph, 'What was in my latest order?')
    assert result['route'] == 'ORDER_HISTORY'
    assert '1405149' in result['messages'][-1].content
    refund = ask(graph, 'My order arrived damaged and I want my money back')
    assert refund['route'] == 'CUSTOMER_SUPPORT'


def test_general_chat_api_failure_is_honest():
    class Unavailable:
        def invoke(self, messages):
            raise RuntimeError('offline')
    result = ask(build_router(model=Unavailable()), 'What is Python?')
    assert result['route'] == 'SMALLTALK'
    assert 'fallback' in result['mode']
    assert "couldn't reach AI" in result['messages'][-1].content
    assert 'refund policy' not in result['messages'][-1].content.lower()


@pytest.mark.parametrize('question', ['Why those?', 'Show them again', 'Which one is best?'])
def test_explaining_previous_suggestions_does_not_consume_new_items(question):
    graph = build_router()
    first = ask(graph, 'Recommend bananas')
    result = ask(graph, question)
    assert result['route'] == 'PRODUCT_RECOMMENDATION'
    assert result['suggested_product_ids'] == first['suggested_product_ids']
    assert [p['product_id'] for p in result['recommendations']] == first['last_suggested_product_ids']
    assert 'refund' not in result['messages'][-1].content.lower()
