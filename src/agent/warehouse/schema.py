"""The data contract: which tables and columns the agent may see.

Single source of truth for the SQL guard (what the scoped views expose) and for
the prompt (what the model is told exists). PII columns of `users` are simply
absent: first_name, last_name, email, street_address, postal_code, latitude,
longitude, user_geom.
"""

from __future__ import annotations

SCHEMA: dict[str, dict[str, str]] = {
    "orders": {
        "order_id": "INT64",
        "user_id": "INT64",
        "status": "STRING",
        "gender": "STRING",
        "created_at": "TIMESTAMP",
        "returned_at": "TIMESTAMP",
        "shipped_at": "TIMESTAMP",
        "delivered_at": "TIMESTAMP",
        # For a restricted user sql_guard recounts this from their order_items: only the items within their scope.
        "num_of_item": "INT64",
    },
    "order_items": {
        "id": "INT64",
        "order_id": "INT64",
        "user_id": "INT64",
        "product_id": "INT64",
        "inventory_item_id": "INT64",
        "status": "STRING",
        "created_at": "TIMESTAMP",
        "shipped_at": "TIMESTAMP",
        "delivered_at": "TIMESTAMP",
        "returned_at": "TIMESTAMP",
        "sale_price": "FLOAT64",
    },
    "products": {
        "id": "INT64",
        "cost": "FLOAT64",
        "category": "STRING",
        "name": "STRING",
        "brand": "STRING",
        "retail_price": "FLOAT64",
        "department": "STRING",
        "sku": "STRING",
        "distribution_center_id": "INT64",
    },
    "users": {
        "id": "INT64",
        "age": "INT64",
        "gender": "STRING",
        "state": "STRING",
        "country": "STRING",
        "city": "STRING",
        "traffic_source": "STRING",
        "created_at": "TIMESTAMP",
    },
}

# Columns of `products` a user's access can be scoped by (config/users.yaml).
SCOPE_KEYS = frozenset({"brand"})


def schema_text() -> str:
    """Compact schema description for the system prompt."""
    return "\n".join(
        f"- {table}({', '.join(f'{col} {typ}' for col, typ in cols.items())})" for table, cols in SCHEMA.items()
    )
