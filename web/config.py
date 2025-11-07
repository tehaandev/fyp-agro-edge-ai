"""
Configuration file for Flask Irrigation Dashboard
"""
import os

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

class Config:
    """Base configuration"""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False

    # Database
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(BASE_DIR, "database.db")}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # Model paths
    IRRIGATION_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'irrigation_rf_final.pkl')
    DISEASE_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'plant_disease__binary_model.tflite')
    SCALER_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'irrigation', 'scaler.pkl')

    # Sensor settings
    SENSOR_READ_INTERVAL = 60  # seconds
    SENSOR_MOCK_MODE = True  # Set to False when using real sensors

    # Irrigation settings
    IRRIGATION_DURATION_HIGH = 30  # minutes
    IRRIGATION_DURATION_LOW = 15   # minutes

    # Disease detection
    IMG_SIDE_LENGTH = 224  # Model input size
    CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for disease prediction

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SENSOR_MOCK_MODE = True

class ProductionConfig(Config):
    """Production configuration (Raspberry Pi)"""
    DEBUG = False
    SENSOR_MOCK_MODE = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
