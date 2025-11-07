#!/usr/bin/env python3
"""
Test script to verify Flask dashboard setup
Run after installing requirements: pip install -r requirements.txt
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("Testing Flask Dashboard Setup")
    print("=" * 60)

    required_packages = [
        ('flask', 'Flask'),
        ('flask_sqlalchemy', 'Flask-SQLAlchemy'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
    ]

    optional_packages = [
        ('tensorflow', 'TensorFlow'),
        ('sklearn', 'scikit-learn'),
    ]

    print("\n✓ Checking required packages...")
    failed = []

    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            failed.append(package_name)

    print("\n✓ Checking optional packages (ML models)...")
    for module_name, package_name in optional_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ⚠ {package_name} - Not installed (ML features disabled)")

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("   Run: pip install -r requirements.txt")
        return False

    return True

def test_config():
    """Test configuration loading"""
    print("\n✓ Testing configuration...")
    try:
        from config import config
        dev_config = config['development']
        print(f"  ✓ Development config loaded: {dev_config}")
        return True
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False

def test_models():
    """Test database models"""
    print("\n✓ Testing database models...")
    try:
        from models import SensorData, IrrigationLog, DiseaseLog
        print("  ✓ SensorData model")
        print("  ✓ IrrigationLog model")
        print("  ✓ DiseaseLog model")
        return True
    except Exception as e:
        print(f"  ✗ Models error: {e}")
        return False

def test_services():
    """Test ML services"""
    print("\n✓ Testing ML services...")

    # Sensor handler
    try:
        from services.sensor_handler import SensorHandler
        handler = SensorHandler(mock_mode=True)
        reading = handler.read_sensors()
        print(f"  ✓ Sensor handler (mock mode)")
        print(f"    Temperature: {reading['temperature']}°C")
        print(f"    Humidity: {reading['humidity']}%")
        print(f"    Soil Moisture: {reading['soil_moisture']}%")
    except Exception as e:
        print(f"  ✗ Sensor handler error: {e}")
        return False

    # Irrigation service
    try:
        from services.irrigation_service import IrrigationService
        model_path = '../models/irrigation_rf_final.pkl'
        service = IrrigationService(model_path=model_path)
        if service.model:
            print("  ✓ Irrigation service (model loaded)")
        else:
            print("  ⚠ Irrigation service (model not found - using fallback)")
    except Exception as e:
        print(f"  ✗ Irrigation service error: {e}")
        return False

    # Disease detection service
    try:
        from services.disease_service import DiseaseDetectionService
        model_path = '../models/plant_disease__binary_model.tflite'
        service = DiseaseDetectionService(model_path=model_path)
        if service.interpreter:
            print("  ✓ Disease detection service (TFLite loaded)")
        else:
            print("  ⚠ Disease detection service (model not found)")
    except Exception as e:
        print(f"  ⚠ Disease detection service: {e}")
        # This is okay if TensorFlow isn't installed

    return True

def test_app_creation():
    """Test Flask app creation"""
    print("\n✓ Testing Flask app creation...")
    try:
        from app import create_app
        app = create_app('development')
        print(f"  ✓ Flask app created: {app.name}")
        print(f"  ✓ Debug mode: {app.config['DEBUG']}")
        print(f"  ✓ Mock sensors: {app.config['SENSOR_MOCK_MODE']}")
        return True
    except Exception as e:
        print(f"  ✗ App creation error: {e}")
        return False

def check_model_files():
    """Check if ML model files exist"""
    print("\n✓ Checking ML model files...")

    models = [
        ('../models/irrigation_rf_final.pkl', 'Irrigation RandomForest'),
        ('../models/plant_disease__binary_model.tflite', 'Disease Detection TFLite'),
        ('../data/processed/irrigation/scaler.pkl', 'Feature Scaler'),
    ]

    found_models = []
    missing_models = []

    for path, name in models:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"  ✓ {name}: {size:.1f} KB")
            found_models.append(name)
        else:
            print(f"  ✗ {name}: NOT FOUND")
            missing_models.append(name)

    if missing_models:
        print("\n⚠ Missing model files:")
        print("  Train models using notebooks:")
        print("    - notebooks/5_irrigation_rf_deployment.ipynb")
        print("    - notebooks/1_disease_detection.ipynb")
        print("  The app will use fallback logic until models are available.")

    return len(found_models) > 0

def main():
    """Run all tests"""
    print("\n")

    all_passed = True

    # Test imports
    if not test_imports():
        print("\n❌ Setup incomplete. Install requirements first.")
        sys.exit(1)

    # Test components
    all_passed &= test_config()
    all_passed &= test_models()
    all_passed &= test_services()
    all_passed &= test_app_creation()

    # Check models (not critical)
    check_model_files()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Flask dashboard is ready.")
        print("\nNext steps:")
        print("  1. Initialize database: flask init-db")
        print("  2. Seed sample data: flask seed-db")
        print("  3. Run app: ./run.sh or python app.py")
        print("  4. Open browser: http://localhost:5000")
    else:
        print("❌ Some tests failed. Check errors above.")
        sys.exit(1)

    print("=" * 60)
    print()

if __name__ == '__main__':
    main()
