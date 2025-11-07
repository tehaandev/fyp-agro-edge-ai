# Quick Start Guide - Flask Irrigation Dashboard

**Get the dashboard running in 5 minutes!**

---

## ⚡ Quick Installation (Development Mode)

```bash
# 1. Navigate to project root
cd /path/to/fyp-agro-edge-ai

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install web app dependencies
pip install -r web/requirements.txt

# 4. Navigate to web directory
cd web

# 5. Test setup (optional but recommended)
python test_setup.py

# 6. Initialize database
python -c "from app import create_app; from models import db; app = create_app(); app.app_context().push(); db.create_all(); print('✅ Database initialized')"

# 7. Run the app!
python app.py
```

**Dashboard URL**: Open browser → `http://localhost:5000`

---

## 🎯 Alternative: One-Command Start

```bash
cd /path/to/fyp-agro-edge-ai
source .venv/bin/activate
cd web
./run.sh
```

The script handles:
- ✅ Virtual environment activation
- ✅ Dependency check
- ✅ Database initialization
- ✅ Optional sample data seeding
- ✅ Server startup

---

## 📊 Seed Sample Data (Optional)

To populate the dashboard with 24 hours of mock sensor data:

```bash
# Method 1: Flask CLI command
export FLASK_APP=app.py
flask seed-db

# Method 2: During startup
./run.sh
# When prompted: "Seed database with sample data? (y/n):" → press Y
```

---

## 🧪 Verify Installation

Run the test script to check everything:

```bash
python test_setup.py
```

Expected output:
```
✅ All tests passed! Flask dashboard is ready.
```

---

## 🔧 What If Models Are Missing?

The dashboard works **without trained models** using fallback logic:

- **Irrigation**: Rule-based recommendations (soil moisture thresholds)
- **Disease Detection**: Requires TFLite model (train first)

### Train Models (if needed):

```bash
# From project root
cd notebooks

# 1. Train irrigation model
jupyter notebook 5_irrigation_rf_deployment.ipynb
# Run all cells → saves models/irrigation_rf_final.pkl

# 2. Train disease detection model
jupyter notebook 1_disease_detection.ipynb
# Run all cells → saves models/plant_disease__binary_model.tflite
```

---

## 🌐 Access Dashboard from Other Devices

The dashboard runs on `0.0.0.0:5000`, making it accessible on your local network:

```bash
# Find your IP address
hostname -I | awk '{print $1}'

# Access from phone/tablet
http://<your-ip>:5000
```

**Example**: `http://192.168.1.100:5000`

---

## 📱 Dashboard Features

Once running, you can:

1. **View Sensor Data**
   - Real-time (mocked) temperature, humidity, soil moisture
   - 24-hour history charts

2. **Get Irrigation Recommendations**
   - Click "Get New Recommendation" button
   - See ML-powered decisions (or rule-based fallback)

3. **Detect Plant Diseases**
   - Navigate to Disease Detection page
   - Upload leaf image
   - View prediction (Healthy/Diseased)

4. **View History**
   - Check past irrigation decisions
   - See logged sensor readings

---

## 🐛 Common Issues

### Issue: `ModuleNotFoundError: No module named 'flask'`

**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### Issue: `sqlite3.OperationalError: no such table: sensor_data`

**Solution**: Initialize database
```bash
python -c "from app import create_app; from models import db; app = create_app(); app.app_context().push(); db.create_all()"
```

### Issue: Dashboard shows "No sensor data available"

**Solution**: Trigger sensor reading
```bash
# Visit in browser: http://localhost:5000/api/read-sensors (POST)
# Or use curl:
curl -X POST http://localhost:5000/api/read-sensors
```

### Issue: Models not loading (warnings about models not found)

**Solution**: This is normal if models aren't trained yet. The app uses fallback logic.
- Irrigation: Rule-based decisions work without model
- Disease Detection: Requires TFLite model to function

---

## 🚀 Production Deployment (Raspberry Pi)

See `README.md` for full Raspberry Pi deployment instructions, including:
- Real sensor integration (DHT11, soil moisture)
- Systemd service setup
- Auto-start on boot

---

## 📚 Next Steps

After getting the dashboard running:

1. **Explore the UI**: Navigate through all pages
2. **Test API endpoints**: See `README.md` for API documentation
3. **Train models**: Use Jupyter notebooks to create ML models
4. **Customize settings**: Edit `config.py` for your needs
5. **Deploy to Pi**: Follow production deployment guide

---

## 💡 Pro Tips

- **Auto-refresh sensors**: Enabled by default in development mode (60s interval)
- **Database management**: Use Flask CLI commands (`flask init-db`, `flask seed-db`, `flask clear-db`)
- **Debug mode**: Already enabled in development config
- **Mock sensors**: Generate realistic temporal patterns (diurnal temperature cycles)

---

## 🆘 Need Help?

1. Check `README.md` for comprehensive documentation
2. Run `python test_setup.py` to diagnose issues
3. Review Flask app logs in terminal
4. Check API health: `curl http://localhost:5000/api/health`

---

**Ready to go! 🚀**

Happy farming! 🌾
