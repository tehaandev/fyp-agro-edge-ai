"""
Route blueprints initialization
"""
from flask import Blueprint

# Create blueprints
main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Import routes to register them
from . import main_routes, api_routes
