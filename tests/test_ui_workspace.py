"""Workspace navigation and session-only shopping list isolation."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_order_navigation_and_customer_shopping_lists(monkeypatch):
    monkeypatch.setenv('DEMO_MODE', '1')
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'main.py').run(timeout=30)
    app.chat_input[0].set_value('What was in my latest order?').run(timeout=30)
    thread = app.session_state['thread_id']
    messages = list(app.session_state['messages'])
    app.radio[0].set_value('Orders').run(timeout=30)
    assert not app.exception
    assert any('1405149' in expander.label for expander in app.expander)
    assert 'Lemon Yogurt' in '\n'.join(element.value for element in app.markdown)
    app.radio[0].set_value('Shopping list').run(timeout=30)
    assert not app.exception
    import duckdb
    from db.queries import DB
    with duckdb.connect(str(DB), read_only=True) as con:
        ids = [row[0] for row in con.execute("SELECT product_id FROM products ORDER BY product_name LIMIT 2").fetchall()]
    app.multiselect[0].set_value(ids).run(timeout=30)
    app.checkbox[0].check().run(timeout=30)
    assert app.checkbox[0].value
    assert len(app.checkbox) == 2
    app.radio[0].set_value('Chat').run(timeout=30)
    assert app.session_state['thread_id'] == thread
    assert app.session_state['messages'] == messages
    app.radio[0].set_value('Shopping list').run(timeout=30)
    assert app.multiselect[0].value == ids
    assert app.checkbox[0].value
    app.selectbox[0].set_value(89).run(timeout=30)
    assert app.multiselect[0].value == []
    assert len(app.checkbox) == 0
    app.selectbox[0].set_value(5117).run(timeout=30)
    assert app.multiselect[0].value == ids
    app.multiselect[0].set_value(ids[1:]).run(timeout=30)
    assert len(app.checkbox) == 1
    assert not app.exception
