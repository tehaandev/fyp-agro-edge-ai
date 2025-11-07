"""
Main web routes for dashboard UI
"""
from flask import render_template, request, redirect, url_for, flash, current_app
from werkzeug.utils import secure_filename
import os
from datetime import datetime, timedelta

from routes import main_bp
from models import db, SensorData, IrrigationLog, DiseaseLog

@main_bp.route('/')
def index():
    """Main dashboard page"""
    # Get latest sensor reading
    latest_sensor = SensorData.query.order_by(SensorData.timestamp.desc()).first()

    # Get latest irrigation decision
    latest_irrigation = IrrigationLog.query.order_by(IrrigationLog.timestamp.desc()).first()

    # Get recent disease detections (last 5)
    recent_diseases = DiseaseLog.query.order_by(DiseaseLog.timestamp.desc()).limit(5).all()

    # Get sensor history for charts (last 24 hours)
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    sensor_history = SensorData.query.filter(
        SensorData.timestamp >= cutoff_time
    ).order_by(SensorData.timestamp.asc()).all()

    return render_template(
        'dashboard.html',
        latest_sensor=latest_sensor,
        latest_irrigation=latest_irrigation,
        recent_diseases=recent_diseases,
        sensor_history=sensor_history
    )

@main_bp.route('/history')
def history():
    """View historical data"""
    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 50

    # Get irrigation logs
    irrigation_logs = IrrigationLog.query.order_by(
        IrrigationLog.timestamp.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    return render_template('history.html', logs=irrigation_logs)

@main_bp.route('/disease-detection')
def disease_detection():
    """Disease detection page"""
    # Get recent detections
    recent = DiseaseLog.query.order_by(DiseaseLog.timestamp.desc()).limit(10).all()

    return render_template('disease_detection.html', recent_detections=recent)

@main_bp.route('/upload-leaf', methods=['POST'])
def upload_leaf():
    """Handle leaf image upload for disease detection"""
    if 'leaf_image' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('main.disease_detection'))

    file = request.files['leaf_image']

    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('main.disease_detection'))

    # Check file extension
    allowed_extensions = current_app.config['ALLOWED_EXTENSIONS']
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        flash('Invalid file type. Please upload PNG, JPG, or JPEG', 'error')
        return redirect(url_for('main.disease_detection'))

    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        # Run disease detection
        from services.disease_service import disease_service

        if disease_service is None or disease_service.interpreter is None:
            flash('Disease detection model not available', 'error')
            return redirect(url_for('main.disease_detection'))

        result = disease_service.predict(filepath)

        if 'error' in result:
            flash(f"Detection error: {result['error']}", 'error')
            return redirect(url_for('main.disease_detection'))

        # Save to database
        log_entry = DiseaseLog(
            image_path=f"uploads/{filename}",
            prediction=result['prediction'],
            confidence=result['confidence'],
            inference_time_ms=result.get('inference_time_ms')
        )
        db.session.add(log_entry)
        db.session.commit()

        flash(f"Detection complete: {result['prediction']} ({result['confidence']:.1%} confidence)", 'success')

    except Exception as e:
        flash(f"Upload error: {str(e)}", 'error')

    return redirect(url_for('main.disease_detection'))

@main_bp.route('/settings')
def settings():
    """Settings and configuration page"""
    return render_template('settings.html', config=current_app.config)

@main_bp.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')
