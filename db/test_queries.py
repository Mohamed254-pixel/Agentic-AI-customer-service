
from db.queries import get_customer_orders, get_order_products

# Customer present in our sample
user_id = 5117

print("CUSTOMER ORDER HISTORY")
orders = get_customer_orders(user_id)

for order in orders:
    print(order)

print("\nPRODUCTS IN ORDER 1662937")

products = get_order_products(1662937)

for product in products:
    print(product)

print(f"\nTotal products: {len(products)}")
