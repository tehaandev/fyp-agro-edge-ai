"""
Flask application for Edge-Based Smart Irrigation Dashboard
Main application entry point
"""
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from config import config
from models import db, init_db

def create_app(config_name='development'):
    """
    Application factory pattern

    Args:
        config_name: 'development' or 'production'

    Returns:
        Flask app instance
    """
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config[config_name])

    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize database
    init_db(app)

    # Initialize services
    init_services(app)

    # Register blueprints
    from routes import main_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    # Error handlers
    register_error_handlers(app)

    # Background tasks (optional)
    if app.config.get('SENSOR_MOCK_MODE'):
        init_background_tasks(app)

    print("=" * 60)
    print("🌾 EDGE-BASED SMART IRRIGATION DASHBOARD")
    print("=" * 60)
    print(f"✅ App initialized in {config_name.upper()} mode")
    print(f"📊 Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"🔧 Mock sensors: {app.config['SENSOR_MOCK_MODE']}")
    print("=" * 60)

    return app

def init_services(app):
    """Initialize ML services and sensor handler"""
    from services.irrigation_service import IrrigationService
    from services.disease_service import DiseaseDetectionService
    from services.sensor_handler import SensorHandler
    import services.irrigation_service as irr_module
    import services.disease_service as dis_module
    import services.sensor_handler as sensor_module

    # Initialize irrigation service
    irr_module.irrigation_service = IrrigationService(
        model_path=app.config['IRRIGATION_MODEL_PATH'],
        scaler_path=app.config['SCALER_PATH']
    )

    # Initialize disease detection service
    dis_module.disease_service = DiseaseDetectionService(
        model_path=app.config['DISEASE_MODEL_PATH'],
        img_size=app.config['IMG_SIDE_LENGTH']
    )

    # Initialize sensor handler
    sensor_module.sensor_handler = SensorHandler(
        mock_mode=app.config['SENSOR_MOCK_MODE']
    )

    print("✅ Services initialized")

def init_background_tasks(app):
    """
    Initialize background tasks for periodic sensor reading
    (In production, use APScheduler or system cron)
    """
    # Import here to avoid circular imports
    from threading import Thread
    import time
    from services.sensor_handler import sensor_handler
    from models import SensorData

    def sensor_reading_task():
        """Background task to periodically read and store sensor data"""
        with app.app_context():
            while True:
                try:
                    # Read sensors
                    reading = sensor_handler.read_sensors()

                    # Store in database
                    sensor_data = SensorData(
                        temperature=reading['temperature'],
                        humidity=reading['humidity'],
                        soil_moisture=reading['soil_moisture'],
                        atmospheric_temp=reading.get('atmospheric_temp'),
                        soil_temp=reading.get('soil_temp'),
                        dew_point=reading.get('dew_point')
                    )

                    db.session.add(sensor_data)
                    db.session.commit()

                    print(f"📊 Sensor data recorded: {reading['timestamp']}")

                except Exception as e:
                    print(f"❌ Error in sensor reading task: {e}")
                    db.session.rollback()

                # Sleep for configured interval
                time.sleep(app.config['SENSOR_READ_INTERVAL'])

    # Start background thread (only in development mode)
    if app.config.get('DEBUG'):
        thread = Thread(target=sensor_reading_task, daemon=True)
        thread.start()
        print(f"✅ Background sensor reading started (interval: {app.config['SENSOR_READ_INTERVAL']}s)")

def register_error_handlers(app):
    """Register custom error handlers"""

    @app.errorhandler(404)
    def not_found(error):
        from flask import render_template
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        from flask import render_template
        db.session.rollback()
        return render_template('errors/500.html'), 500

# CLI commands for database management
def register_cli_commands(app):
    """Register Flask CLI commands"""
    import click

    @app.cli.command('init-db')
    def init_db_command():
        """Initialize the database"""
        with app.app_context():
            db.create_all()
            click.echo('✅ Database initialized.')

    @app.cli.command('seed-db')
    def seed_db_command():
        """Seed database with sample data"""
        from datetime import datetime, timedelta
        from services.sensor_handler import sensor_handler

        with app.app_context():
            click.echo('Seeding database with sample sensor data...')

            # Generate 24 hours of sample data (hourly)
            for i in range(24, 0, -1):
                reading = sensor_handler.read_sensors()
                timestamp = datetime.utcnow() - timedelta(hours=i)

                sensor_data = SensorData(
                    timestamp=timestamp,
                    temperature=reading['temperature'],
                    humidity=reading['humidity'],
                    soil_moisture=reading['soil_moisture'],
                    atmospheric_temp=reading.get('atmospheric_temp'),
                    soil_temp=reading.get('soil_temp'),
                    dew_point=reading.get('dew_point')
                )

                db.session.add(sensor_data)

            db.session.commit()
            click.echo(f'✅ Seeded 24 hours of sensor data.')

    @app.cli.command('clear-db')
    @click.confirmation_option(prompt='Are you sure you want to clear all data?')
    def clear_db_command():
        """Clear all data from database"""
        with app.app_context():
            from models import SensorData, IrrigationLog, DiseaseLog

            SensorData.query.delete()
            IrrigationLog.query.delete()
            DiseaseLog.query.delete()
            db.session.commit()

            click.echo('✅ Database cleared.')

# Main entry point
if __name__ == '__main__':
    # Get environment
    env = os.environ.get('FLASK_ENV', 'development')

    # Create app
    app = create_app(config_name=env)

    # Register CLI commands
    register_cli_commands(app)

    # Run app
    # Use 0.0.0.0 to make accessible on local network (for Raspberry Pi)
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )
