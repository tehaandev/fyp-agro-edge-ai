# Smart Irrigation Dashboard - Flask Web Application

**Edge-Based ML Dashboard for Real-Time Irrigation Decision Support**

This Flask web application provides an offline dashboard for smallholder farmers, integrating machine learning models for smart irrigation recommendations and plant disease detection.

---

## 🌟 Features

- **📊 Real-Time Sensor Monitoring**: Display temperature, humidity, and soil moisture readings
- **💧 Smart Irrigation**: ML-powered irrigation recommendations using RandomForest model
- **🌿 Disease Detection**: Plant disease identification from leaf images using TFLite
- **📈 Historical Data**: View sensor history and irrigation logs
- **🔌 Offline Operation**: Runs completely offline on Raspberry Pi 4B
- **📱 Responsive UI**: Bootstrap-based interface accessible from any device

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- Trained ML models in `../models/` directory
  - `irrigation_rf_final.pkl` (RandomForest)
  - `plant_disease__binary_model.tflite` (MobileNetV2)

### Installation

```bash
# 1. Create and activate virtual environment (from project root)
cd /path/to/fyp-agro-edge-ai
python3 -m venv .venv
source .venv/bin/activate

# 2. Install web app dependencies
cd web
pip install -r requirements.txt

# 3. Initialize database
python -c "from app import create_app; from models import db; app = create_app(); app.app_context().push(); db.create_all()"

# 4. (Optional) Seed with sample data
export FLASK_APP=app.py
flask seed-db

# 5. Run the application
./run.sh
```

### Alternative: Direct Python Execution

```bash
cd web
python app.py
```

The dashboard will be available at:
- **Local**: `http://localhost:5000`
- **Network**: `http://<your-ip>:5000` (accessible from other devices)

---

## 📁 Project Structure

```
web/
├── app.py                    # Main Flask application
├── config.py                 # Configuration settings
├── models.py                 # Database models (SQLAlchemy)
├── requirements.txt          # Python dependencies
├── run.sh                    # Startup script
│
├── routes/                   # Flask blueprints
│   ├── __init__.py
│   ├── main_routes.py        # Web page routes
│   └── api_routes.py         # JSON API endpoints
│
├── services/                 # Business logic
│   ├── sensor_handler.py     # Sensor reading (mock + real)
│   ├── irrigation_service.py # Irrigation ML model
│   └── disease_service.py    # Disease detection ML model
│
├── templates/                # Jinja2 HTML templates
│   ├── base.html
│   ├── dashboard.html        # Main dashboard
│   ├── disease_detection.html
│   ├── history.html
│   ├── about.html
│   └── errors/
│       ├── 404.html
│       └── 500.html
│
├── static/                   # Static assets
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── main.js
│   │   └── dashboard.js
│   └── uploads/              # Disease detection image uploads
│
└── database.db               # SQLite database (created on first run)
```

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Development mode (mock sensors)
FLASK_ENV=development
SENSOR_MOCK_MODE=True

# Production mode (real sensors on Raspberry Pi)
FLASK_ENV=production
SENSOR_MOCK_MODE=False
```

### Model Paths

Ensure these models exist in the parent `models/` directory:

```python
IRRIGATION_MODEL_PATH = '../models/irrigation_rf_final.pkl'
DISEASE_MODEL_PATH = '../models/plant_disease__binary_model.tflite'
SCALER_PATH = '../data/processed/irrigation/scaler.pkl'
```

---

## 🌐 API Endpoints

### Sensor Data

- **GET** `/api/sensor-data?latest=true` - Get latest sensor reading
- **GET** `/api/sensor-data?hours=24` - Get sensor history
- **POST** `/api/read-sensors` - Trigger new sensor reading

### Irrigation

- **GET** `/api/irrigation-decision` - Get irrigation recommendation
- **GET** `/api/irrigation-history?days=7` - Get irrigation log

### Disease Detection

- **GET** `/api/disease-history?limit=50` - Get detection history

### Statistics

- **GET** `/api/stats` - Get dashboard statistics
- **GET** `/api/health` - Health check endpoint

---

## 🖥️ Web Pages

- **`/`** - Main dashboard (sensor status, irrigation, quick actions)
- **`/disease-detection`** - Upload leaf images for disease analysis
- **`/history`** - View irrigation decision history
- **`/about`** - Project information and technical specs

---

## 🤖 ML Model Integration

### Irrigation Service (`services/irrigation_service.py`)

- **Model**: RandomForest (97.89% accuracy)
- **Input**: 6-hour window of sensor data (222 features)
- **Output**: `Irrigate_High`, `Irrigate_Low`, or `No_Irrigation`
- **Fallback**: Rule-based logic if model unavailable

**Feature Engineering**:
- 37 engineered features per hour (rolling stats, change rates, time features)
- Handles temporal data with 6-hour sliding window
- Auto-scaling with StandardScaler

### Disease Detection Service (`services/disease_service.py`)

- **Model**: MobileNetV2 TFLite (98% accuracy, 9.1 MB)
- **Input**: 224×224 RGB leaf image
- **Output**: `Healthy` or `Diseased` with confidence
- **Preprocessing**: Auto-resize, normalize, TFLite inference

---

## 🔌 Sensor Integration

### Mock Mode (Development)

Generates realistic sensor data with temporal patterns:

```python
# Automatic in config.py
SENSOR_MOCK_MODE = True
```

### Real Sensors (Raspberry Pi)

1. Install hardware libraries:
```bash
pip install Adafruit-DHT RPi.GPIO spidev
```

2. Configure GPIO pins in `services/sensor_handler.py`:
```python
dht_pin = 4      # DHT11 sensor
soil_pin = 17    # Soil moisture (via ADC)
```

3. Enable production mode:
```python
SENSOR_MOCK_MODE = False
```

---

## 📊 Database Schema

### `sensor_data`

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| timestamp | DateTime | Reading time |
| temperature | Float | Air temperature (°C) |
| humidity | Float | Air humidity (%) |
| soil_moisture | Float | Soil moisture (%) |
| atmospheric_temp | Float | Atmospheric temp |
| soil_temp | Float | Soil temperature |
| dew_point | Float | Dew point |

### `irrigation_log`

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| timestamp | DateTime | Decision time |
| decision | String | Irrigation recommendation |
| duration | Integer | Duration (minutes) |
| confidence | Float | Model confidence |
| sensor_data_id | Integer | Foreign key |

### `disease_log`

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| timestamp | DateTime | Scan time |
| image_path | String | Uploaded image path |
| prediction | String | `Healthy` or `Diseased` |
| confidence | Float | Model confidence |
| inference_time_ms | Float | Inference time |

---

## 🚀 Deployment to Raspberry Pi

### 1. Transfer Files

```bash
# From development machine
rsync -avz web/ pi@raspberrypi.local:/home/pi/irrigation-dashboard/
rsync -avz models/ pi@raspberrypi.local:/home/pi/irrigation-dashboard/models/
```

### 2. Install Dependencies

```bash
ssh pi@raspberrypi.local
cd /home/pi/irrigation-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r web/requirements.txt

# Install sensor libraries
pip install Adafruit-DHT RPi.GPIO spidev
```

### 3. Run on Boot (systemd)

Create `/etc/systemd/system/irrigation-dashboard.service`:

```ini
[Unit]
Description=Smart Irrigation Dashboard
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/irrigation-dashboard/web
Environment="FLASK_ENV=production"
ExecStart=/home/pi/irrigation-dashboard/.venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable irrigation-dashboard
sudo systemctl start irrigation-dashboard
sudo systemctl status irrigation-dashboard
```

### 4. Access Dashboard

- **Local**: `http://raspberrypi.local:5000`
- **Network**: `http://192.168.x.x:5000`

---

## 🧪 Testing

### Manual Testing

1. **Sensor Reading**:
```bash
curl -X POST http://localhost:5000/api/read-sensors
```

2. **Irrigation Decision**:
```bash
curl http://localhost:5000/api/irrigation-decision
```

3. **Health Check**:
```bash
curl http://localhost:5000/api/health
```

### Database Management

```bash
# Initialize database
flask init-db

# Seed with sample data
flask seed-db

# Clear database
flask clear-db
```

---

## 🐛 Troubleshooting

### Models Not Loading

**Error**: `⚠️ Model not found at ...`

**Solution**: Ensure models are trained and saved in correct paths:
```bash
ls -lh ../models/irrigation_rf_final.pkl
ls -lh ../models/plant_disease__binary_model.tflite
```

### TensorFlow Import Error

**Error**: `ImportError: No module named 'tensorflow'`

**Solution**:
```bash
pip install tensorflow==2.15.0
# For Raspberry Pi:
pip install tensorflow-aarch64==2.15.0
```

### Database Locked

**Error**: `sqlite3.OperationalError: database is locked`

**Solution**: Restart Flask app (SQLite doesn't support concurrent writes)

### Permission Denied (GPIO)

**Error**: `RuntimeError: No access to /dev/mem`

**Solution**: Run with sudo or add user to `gpio` group:
```bash
sudo usermod -a -G gpio pi
```

---

## 📝 Development Notes

- **Mock sensors** generate realistic temporal patterns (diurnal temperature, moisture drying)
- **Auto-refresh** can be enabled in `static/js/main.js` (disabled by default)
- **Background sensor reading** runs in development mode (60s interval)
- **Upload folder** is auto-created in `static/uploads/`
- **Database** is SQLite for simplicity (consider PostgreSQL for production scale)

---

## 🔐 Security Considerations

- Change `SECRET_KEY` in production
- Disable `DEBUG` mode in production
- Restrict file upload types and sizes
- Use HTTPS with reverse proxy (nginx/Apache)
- Implement user authentication if needed

---

## 📚 Related Documentation

- **Main Project**: `../README.md`
- **Model Selection**: `../docs/MODEL_SELECTION_RATIONALE.md`
- **Training Notebooks**: `../notebooks/`
- **Deployment Guide**: `../DEVELOPMENT_HANDOFF.md`

---

## 🤝 Contributing

This is a final year project (FYP 2025). For academic purposes, contributions are limited to project team members.

---

## 📄 License

Academic project - All rights reserved.

---

## 👨‍💻 Author

**Final Year Project 2025**
Computer Science
Edge-Based ML for Small-Scale Farms

---

**Need help?** Check the main project documentation or raise an issue in the GitHub repository.
