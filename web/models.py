"""
Database models for irrigation dashboard
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class SensorData(db.Model):
    """Store sensor readings"""
    __tablename__ = 'sensor_data'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    temperature = db.Column(db.Float, nullable=False)  # Celsius
    humidity = db.Column(db.Float, nullable=False)  # Percentage
    soil_moisture = db.Column(db.Float, nullable=False)  # Percentage
    atmospheric_temp = db.Column(db.Float, nullable=True)  # For model compatibility
    soil_temp = db.Column(db.Float, nullable=True)  # For model compatibility
    dew_point = db.Column(db.Float, nullable=True)  # For model compatibility

    def __repr__(self):
        return f'<SensorData {self.timestamp}: T={self.temperature}°C, H={self.humidity}%, SM={self.soil_moisture}%>'

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'temperature': round(self.temperature, 2),
            'humidity': round(self.humidity, 2),
            'soil_moisture': round(self.soil_moisture, 2),
            'atmospheric_temp': round(self.atmospheric_temp, 2) if self.atmospheric_temp else None,
            'soil_temp': round(self.soil_temp, 2) if self.soil_temp else None,
            'dew_point': round(self.dew_point, 2) if self.dew_point else None
        }

class IrrigationLog(db.Model):
    """Log irrigation decisions"""
    __tablename__ = 'irrigation_log'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    decision = db.Column(db.String(50), nullable=False)  # 'Irrigate_High', 'Irrigate_Low', 'No_Irrigation'
    duration = db.Column(db.Integer, nullable=True)  # Duration in minutes (if irrigating)
    confidence = db.Column(db.Float, nullable=True)  # Model confidence
    sensor_data_id = db.Column(db.Integer, db.ForeignKey('sensor_data.id'), nullable=True)

    # Relationship
    sensor_data = db.relationship('SensorData', backref=db.backref('irrigation_decisions', lazy=True))

    def __repr__(self):
        return f'<IrrigationLog {self.timestamp}: {self.decision} ({self.duration}min)>'

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'decision': self.decision,
            'duration': self.duration,
            'confidence': round(self.confidence, 4) if self.confidence else None,
            'sensor_data_id': self.sensor_data_id
        }

class DiseaseLog(db.Model):
    """Log disease detection results"""
    __tablename__ = 'disease_log'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    image_path = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)  # 'Healthy' or 'Diseased'
    confidence = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    inference_time_ms = db.Column(db.Float, nullable=True)  # Performance metric

    def __repr__(self):
        return f'<DiseaseLog {self.timestamp}: {self.prediction} ({self.confidence:.2%})>'

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'image_path': self.image_path,
            'prediction': self.prediction,
            'confidence': round(self.confidence, 4),
            'inference_time_ms': round(self.inference_time_ms, 2) if self.inference_time_ms else None
        }

def init_db(app):
    """Initialize database"""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("✅ Database initialized successfully")
