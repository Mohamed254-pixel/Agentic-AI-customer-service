"""Order History Agent.

This module answers order questions with data from the DuckDB helper functions.
It does not use an AI model to create order details, so the database remains the
source of truth.
"""

import re

from db.queries import get_customer_orders, get_order_products


DATABASE_ERROR_TEXT = (
    "I couldn't access the order database right now. Please try again later."
)

HELP_TEXT = (
    "I can help with order history questions. For example, ask: "
    "'Show my order history', 'What was in my latest order?', or "
    "'Show order 1662937'."
)

UNAVAILABLE_DATA_TEXT = (
    "That information is not available in this order-history dataset. "
    "It does not contain live delivery status or tracking, prices, order "
    "totals, payment information, exact calendar dates, refunds, product "
    "quantities, customer names, or addresses."
)


def _find_requested_order_id(question):
    """Return an order ID mentioned after the word 'order', or None."""

    match = re.search(r"\border\s*(?:#|number\s*)?(\d+)\b", question)
    if match:
        return int(match.group(1))
    return None


def _asks_for_unavailable_data(question):
    """Return True when the question asks for data absent from this dataset."""

    unavailable_patterns = [
        r"\b(where is|track|tracking|delivery|status|arriv\w*)\b",
        r"\b(price|prices|cost|costs|order total|payment|paid|how much)\b",
        r"\b(exact date|calendar date|what date|when did|when was)\b",
        r"\b(refund|refunded)\b",
        r"\b(quantity|quantities|how many of each)\b",
        r"\b(customer name|my name|address)\b",
    ]
    return any(re.search(pattern, question) for pattern in unavailable_patterns)


def _format_product_text(order_id, products):
    """Build readable text from product rows returned by the database."""

    product_names = [product["product_name"] for product in products]
    item_lines = "\n".join(f"- {name}" for name in product_names)
    return f"Order {order_id} contained:\n{item_lines}"


def _get_customer_orders_safely(user_id):
    """Load orders and return (orders, error_text)."""

    try:
        return get_customer_orders(user_id), None
    except Exception:
        # This also handles a missing/unavailable DuckDB database gracefully.
        return [], DATABASE_ERROR_TEXT


def _answer_for_specific_order(orders, requested_order_id):
    """Answer only after confirming that the order belongs to this customer."""

    owned_order_ids = {int(order["order_id"]) for order in orders}
    if requested_order_id not in owned_order_ids:
        # Use one message for unknown and unowned orders so no other customer's
        # order information is revealed.
        return {
            "text": f"Order {requested_order_id} was not found in your order history.",
            "rows": [],
        }

    # This database call happens only after the ownership check above.
    try:
        products = get_order_products(requested_order_id)
    except Exception:
        return {"text": DATABASE_ERROR_TEXT, "rows": []}

    if not products:
        return {
            "text": f"Order {requested_order_id} exists, but its product details are unavailable in this dataset.",
            "rows": [],
        }

    return {
        "text": _format_product_text(requested_order_id, products),
        "rows": products,
    }


def handle_order_question(user_id, question):
    """Answer an order-history question for one customer.

    Args:
        user_id: The current customer's ID.
        question: The customer's order-related question.

    Returns:
        A dictionary with display text and database rows used in the answer.
    """

    if not isinstance(question, str) or not question.strip():
        return {"text": f"Please enter an order question. {HELP_TEXT}", "rows": []}

    normalized_question = question.strip().lower()

    requested_order_id = _find_requested_order_id(normalized_question)
    asks_for_unavailable_data = _asks_for_unavailable_data(normalized_question)
    asks_for_latest = bool(
        re.search(
            r"\b(latest order|last order|last time|most recent order)\b",
            normalized_question,
        )
    )
    asks_for_history = bool(
        re.search(
            r"\b(order history|previous orders|past orders|how many orders|all orders)\b",
            normalized_question,
        )
    )

    if not (
        requested_order_id
        or asks_for_unavailable_data
        or asks_for_latest
        or asks_for_history
    ):
        return {"text": HELP_TEXT, "rows": []}

    # A general unavailable-data question needs no database lookup. A question
    # naming an order is handled below so ownership is checked first.
    if asks_for_unavailable_data and requested_order_id is None:
        return {"text": UNAVAILABLE_DATA_TEXT, "rows": []}

    orders, error_text = _get_customer_orders_safely(user_id)
    if error_text:
        return {"text": error_text, "rows": []}

    if not orders:
        return {"text": "I couldn't find any orders for this customer.", "rows": []}

    if requested_order_id is not None:
        if asks_for_unavailable_data:
            owned_order_ids = {int(order["order_id"]) for order in orders}
            if requested_order_id not in owned_order_ids:
                return {
                    "text": (
                        f"Order {requested_order_id} was not found in your order history."
                    ),
                    "rows": [],
                }
            return {"text": UNAVAILABLE_DATA_TEXT, "rows": []}
        return _answer_for_specific_order(orders, requested_order_id)

    if asks_for_latest:
        # order_number records the sequence of a customer's orders. It is more
        # recent when its value is larger; exact calendar dates are unavailable.
        latest_order = max(orders, key=lambda order: order["order_number"])
        return _answer_for_specific_order(orders, int(latest_order["order_id"]))

    order_ids = [str(order["order_id"]) for order in orders]
    if "how many orders" in normalized_question:
        text = f"You have {len(orders)} historical orders."
    else:
        text = (
            f"You have {len(orders)} historical orders. "
            f"Order IDs (newest first): {', '.join(order_ids)}."
        )
    return {"text": text, "rows": orders}
