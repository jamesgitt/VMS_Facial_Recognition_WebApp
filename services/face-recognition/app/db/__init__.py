"""
Database Module

Provides database connection management and query functions.

Usage:
    from app.db import get_connection, get_visitor_images_from_db
    from app.db.connection import init_connection_pool, test_connection
    from app.db.queries import get_visitor_details, update_visitor_features
"""

# Connection management
from app.db.connection import (
    get_db_connection,
    get_connection,
    init_connection_pool,
    test_connection,
    close_connection_pool,
)

# Query functions
from app.db.queries import (
    get_visitor_images_from_db,
    get_visitors_with_features_only,
    get_visitor_details,
    update_visitor_features,
    _quote_column,  # Exposed for advanced use cases
)

# Models and types
from app.db.models import (
    VisitorBase,
    VisitorWithImage,
    VisitorWithFeatures,
    VisitorFull,
    VisitorDict,
    VisitorList,
    validate_visitor_id,
    validate_table_name,
)

# Backward compatibility: maintain old database module interface
# This allows existing code to continue using `import database`
import sys
from types import ModuleType

# Create a compatibility module
_compat_module = ModuleType('database')
_compat_module.get_db_connection = get_db_connection
_compat_module.get_connection = get_connection
_compat_module.init_connection_pool = init_connection_pool
_compat_module.test_connection = test_connection
_compat_module.close_connection_pool = close_connection_pool
_compat_module.get_visitor_images_from_db = get_visitor_images_from_db
_compat_module.get_visitors_with_features_only = get_visitors_with_features_only
_compat_module.get_visitor_details = get_visitor_details
_compat_module.update_visitor_features = update_visitor_features

# For backward compatibility, expose as 'database' module
# Note: This is a compatibility layer - new code should use explicit imports
__all__ = [
    # Connection
    'get_db_connection',
    'get_connection',
    'init_connection_pool',
    'test_connection',
    'close_connection_pool',
    # Queries
    'get_visitor_images_from_db',
    'get_visitors_with_features_only',
    'get_visitor_details',
    'update_visitor_features',
    # Models
    'VisitorBase',
    'VisitorWithImage',
    'VisitorWithFeatures',
    'VisitorFull',
    'VisitorDict',
    'VisitorList',
    'validate_visitor_id',
    'validate_table_name',
]
