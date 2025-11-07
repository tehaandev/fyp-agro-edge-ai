"""
API routes for JSON endpoints
"""
from flask import jsonify, request, current_app
from datetime import datetime, timedelta

from routes import api_bp
from models import db, SensorData, IrrigationLog, DiseaseLog
from services.sensor_handler import sensor_handler
from services.irrigation_service import irrigation_service
from services.disease_service import disease_service

@api_bp.route('/sensor-data', methods=['GET'])
def get_sensor_data():
    """
    Get latest sensor readings
    Query params:
        - hours: number of hours of history (default: 1)
        - latest: if true, return only latest reading
    """
    latest_only = request.args.get('latest', 'false').lower() == 'true'
    hours = request.args.get('hours', 1, type=int)

    if latest_only:
        latest = SensorData.query.order_by(SensorData.timestamp.desc()).first()
        if latest:
            return jsonify(latest.to_dict())
        else:
            return jsonify({'error': 'No sensor data available'}), 404

    # Get historical data
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    readings = SensorData.query.filter(
        SensorData.timestamp >= cutoff_time
    ).order_by(SensorData.timestamp.asc()).all()

    return jsonify({
        'count': len(readings),
        'data': [r.to_dict() for r in readings]
    })

@api_bp.route('/sensor-data', methods=['POST'])
def record_sensor_data():
    """
    Manually record sensor data (for testing or manual entry)
    Body: JSON with temperature, humidity, soil_moisture
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    required_fields = ['temperature', 'humidity', 'soil_moisture']
    if not all(field in data for field in required_fields):
        return jsonify({'error': f'Missing required fields: {required_fields}'}), 400

    try:
        sensor_reading = SensorData(
            temperature=float(data['temperature']),
            humidity=float(data['humidity']),
            soil_moisture=float(data['soil_moisture']),
            atmospheric_temp=float(data.get('atmospheric_temp', data['temperature'])),
            soil_temp=float(data.get('soil_temp', data['temperature'] - 2)),
            dew_point=float(data.get('dew_point', data['temperature'] - (100 - data['humidity']) / 5))
        )

        db.session.add(sensor_reading)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'data': sensor_reading.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_bp.route('/read-sensors', methods=['POST'])
def read_sensors():
    """
    Trigger sensor reading and store in database
    This would be called by a cron job or periodic task
    """
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

        return jsonify({
            'status': 'success',
            'data': sensor_data.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_bp.route('/irrigation-decision', methods=['GET'])
def get_irrigation_decision():
    """
    Get irrigation recommendation based on current conditions
    Uses historical sensor data + ML model
    """
    try:
        # Check if model is available
        if irrigation_service is None or irrigation_service.model is None:
            # Fallback to rule-based
            use_fallback = True
        else:
            use_fallback = False

        # Get historical sensor data (last 13 hours for feature engineering)
        cutoff_time = datetime.utcnow() - timedelta(hours=13)
        historical_data = SensorData.query.filter(
            SensorData.timestamp >= cutoff_time
        ).order_by(SensorData.timestamp.asc()).all()

        if len(historical_data) == 0:
            return jsonify({
                'error': 'No sensor data available',
                'recommendation': 'Please wait for sensor readings'
            }), 404

        # Convert to list of dicts
        sensor_list = [r.to_dict() for r in historical_data]

        # Get prediction
        result = irrigation_service.predict(sensor_list, use_simple_fallback=True)

        if 'error' in result:
            return jsonify(result), 500

        # Log decision to database
        latest_sensor = historical_data[-1]
        log_entry = IrrigationLog(
            decision=result['decision'],
            duration=result.get('duration', 0),
            confidence=result.get('confidence'),
            sensor_data_id=latest_sensor.id
        )
        db.session.add(log_entry)
        db.session.commit()

        # Add log ID to result
        result['log_id'] = log_entry.id

        return jsonify(result)

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_bp.route('/irrigation-history', methods=['GET'])
def get_irrigation_history():
    """
    Get irrigation decision history
    Query params:
        - days: number of days (default: 7)
        - limit: max number of records (default: 100)
    """
    days = request.args.get('days', 7, type=int)
    limit = request.args.get('limit', 100, type=int)

    cutoff_time = datetime.utcnow() - timedelta(days=days)

    logs = IrrigationLog.query.filter(
        IrrigationLog.timestamp >= cutoff_time
    ).order_by(IrrigationLog.timestamp.desc()).limit(limit).all()

    return jsonify({
        'count': len(logs),
        'data': [log.to_dict() for log in logs]
    })

@api_bp.route('/disease-history', methods=['GET'])
def get_disease_history():
    """Get disease detection history"""
    limit = request.args.get('limit', 50, type=int)

    logs = DiseaseLog.query.order_by(
        DiseaseLog.timestamp.desc()
    ).limit(limit).all()

    return jsonify({
        'count': len(logs),
        'data': [log.to_dict() for log in logs]
    })

@api_bp.route('/stats', methods=['GET'])
def get_statistics():
    """
    Get dashboard statistics
    Returns: summary stats for sensor data, irrigation decisions, disease detections
    """
    try:
        # Recent time window
        cutoff_24h = datetime.utcnow() - timedelta(hours=24)
        cutoff_7d = datetime.utcnow() - timedelta(days=7)

        # Sensor stats
        total_readings = SensorData.query.count()
        recent_readings_24h = SensorData.query.filter(SensorData.timestamp >= cutoff_24h).count()

        # Latest sensor values
        latest_sensor = SensorData.query.order_by(SensorData.timestamp.desc()).first()

        # Irrigation stats
        total_irrigations = IrrigationLog.query.filter(
            IrrigationLog.decision.in_(['Irrigate_High', 'Irrigate_Low'])
        ).count()

        irrigations_7d = IrrigationLog.query.filter(
            IrrigationLog.timestamp >= cutoff_7d,
            IrrigationLog.decision.in_(['Irrigate_High', 'Irrigate_Low'])
        ).count()

        # Disease stats
        total_scans = DiseaseLog.query.count()
        diseased_count = DiseaseLog.query.filter(DiseaseLog.prediction == 'Diseased').count()
        healthy_count = DiseaseLog.query.filter(DiseaseLog.prediction == 'Healthy').count()

        return jsonify({
            'sensors': {
                'total_readings': total_readings,
                'recent_24h': recent_readings_24h,
                'latest': latest_sensor.to_dict() if latest_sensor else None
            },
            'irrigation': {
                'total_irrigations': total_irrigations,
                'recent_7d': irrigations_7d
            },
            'disease': {
                'total_scans': total_scans,
                'diseased_count': diseased_count,
                'healthy_count': healthy_count,
                'disease_rate': diseased_count / total_scans if total_scans > 0 else 0
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'database': db.session.is_active,
            'irrigation_model': irrigation_service is not None and irrigation_service.model is not None,
            'disease_model': disease_service is not None and disease_service.interpreter is not None,
            'sensor_handler': sensor_handler is not None
        }
    })
