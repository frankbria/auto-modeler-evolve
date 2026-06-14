"""Authentication and authorization for the AutoModeler backend.

- ``security``: password hashing (bcrypt) and JWT access tokens.
- ``dependencies``: the ``get_current_user`` FastAPI dependency.
- ``scoping``: owner/tenant authorization helpers for resources.
"""
