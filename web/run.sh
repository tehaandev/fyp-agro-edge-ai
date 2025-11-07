#!/bin/bash
# Startup script for Smart Irrigation Dashboard

echo "=========================================="
echo "Smart Irrigation Dashboard - Startup"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "../.venv" ]; then
    echo "✓ Activating virtual environment..."
    source ../.venv/bin/activate
else
    echo "⚠ Warning: Virtual environment not found"
    echo "  Run: python3 -m venv ../.venv && source ../.venv/bin/activate"
fi

# Check if requirements are installed
echo "✓ Checking dependencies..."
python -c "import flask" 2>/dev/null || {
    echo "⚠ Flask not found. Installing requirements..."
    pip install -r requirements.txt
}

# Initialize database if it doesn't exist
if [ ! -f "database.db" ]; then
    echo "✓ Initializing database..."
    python -c "from app import create_app; from models import db; app = create_app(); app.app_context().push(); db.create_all()"
    echo "✓ Database initialized"

    # Seed with sample data
    read -p "Seed database with sample data? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "✓ Seeding database..."
        FLASK_APP=app.py flask seed-db
    fi
fi

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

echo ""
echo "=========================================="
echo "Starting Flask server..."
echo "Dashboard will be available at:"
echo "  - http://localhost:5000"
echo "  - http://$(hostname -I | awk '{print $1}'):5000"
echo "=========================================="
echo ""

# Run Flask app
python app.py
