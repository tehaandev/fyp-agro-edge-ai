# Irrigation Control Model - Quick Reference

## 📊 Final Model Selection

**Production Model**: **RandomForestClassifier**
- 50 trees, max_depth=10
- File: `models/irrigation_rf_final.pkl`
- Deployment notebook: `notebooks/5_irrigation_rf_deployment.ipynb`

---

## 🎯 Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Test Accuracy | 97.89% | ✅ Excellent |
| Test F1-score | 92.22% | ✅ Excellent |
| Test Precision | 91.51% | ✅ Excellent |
| Test Recall | 93.09% | ✅ Excellent |
| CV Accuracy | 97.43% | ✅ Good generalization |
| Inference Time | 0.146 ms | ✅ Edge-compatible |
| Model Size | ~50 KB | ✅ Lightweight |

---

## ⚠️ Why Not DecisionTree?

Despite DecisionTree achieving **100% test accuracy**, we chose RandomForest because:

1. ❌ **100% is unrealistic** for real sensor data (overfitting)
2. ❌ Uses only **3 features** (oversimplification)
3. ❌ **High production risk** (brittle to sensor noise)
4. ❌ Likely **data leakage** from 6-hour sliding window

---

## 🔧 Model Configuration

```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
```

---

## 📁 Deployment Files

### Required Artifacts
1. **Model**: `models/irrigation_rf_final.pkl` (50 KB)
2. **Scaler**: `data/processed/irrigation/scaler.pkl`
3. **Feature Names**: `data/processed/irrigation/feature_names.json`
4. **Metadata**: `models/irrigation_rf_deployment_metadata.json`

### Input Requirements
- **Window size**: 6 hours of sensor readings
- **Features**: 37 engineered features per hour
- **Total input**: (6 hours × 37 features) = 222 features

---

## 🎯 Prediction Classes

| Class | Description | Training Samples |
|-------|-------------|------------------|
| **No_Irrigation** | Sufficient soil moisture | 6,288 (89.8%) |
| **Irrigate_Low** | Moderate irrigation needed | 510 (7.3%) |
| **Irrigate_High** | Urgent irrigation needed | 206 (2.9%) |

---

## 📊 Top 10 Most Important Features (Estimated)

1. Soil_Moisture_rolling_mean_6h (~12%)
2. Atmospheric_Temp_rolling_mean_6h (~11%)
3. recent_rain (~10%)
4. Soil_Moisture (~9%)
5. Atmospheric_Temp (~8%)
6. Humidity_rolling_mean_12h (~7%)
7. Soil_Moisture_change_3h (~6%)
8. hour_cos (~5%)
9. is_daytime (~4%)
10. Humidity_change_1h (~3%)

---

## 🚀 Deployment Workflow

### Step 1: Load Model

```python
import pickle
with open('irrigation_rf_final.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Step 2: Collect 6-Hour Sensor Window

```python
sensor_buffer = []  # Store last 6 hours
# Collect readings every hour
# sensor_buffer = [hour1_reading, hour2_reading, ..., hour6_reading]
```

### Step 3: Engineer Features

```python
# Apply feature engineering pipeline:
# - Rolling averages (3h, 6h, 12h)
# - Rate of change (1h, 3h)
# - Time features (hour, day, season)
# - Rainfall detection
features = engineer_features(sensor_buffer)
```

### Step 4: Scale Features

```python
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
features_scaled = scaler.transform(features)
```

### Step 5: Make Prediction

```python
prediction = model.predict(features_scaled)[0]
probabilities = model.predict_proba(features_scaled)[0]

if prediction == 'Irrigate_High':
    activate_pump(duration='high')
elif prediction == 'Irrigate_Low':
    activate_pump(duration='low')
else:
    deactivate_pump()
```

---

## 🎓 For FYP Defense

### Key Achievements
- ✅ 97.89% accuracy on real sensor data
- ✅ Recognized 100% as overfitting (domain knowledge)
- ✅ Edge-compatible (0.146 ms inference on Pi 4B)
- ✅ Multi-class irrigation control (3 levels)
- ✅ Context-aware features (time, season, weather)

### Novelty Points
1. **Intelligent labeling**: Multi-factor logic (soil + temp + time + rain)
2. **Rainfall detection**: Humidity spike analysis (prevents over-irrigation)
3. **Agricultural context**: Sri Lankan seasons (Yala/Maha), midday avoidance
4. **Production mindset**: Chose 97.89% over 100% for reliability

---

## 📚 Related Documents

- **Model Selection Rationale**: `docs/MODEL_SELECTION_RATIONALE.md` (detailed justification)
- **Preprocessing Pipeline**: `notebooks/3_irrigation_preprocessing.ipynb`
- **Model Comparison**: `notebooks/4_irrigation_model_comparison.ipynb`
- **Deployment Guide**: `notebooks/5_irrigation_rf_deployment.ipynb`

---

## 📞 Quick Commands

```bash
# Run deployment notebook
jupyter notebook notebooks/5_irrigation_rf_deployment.ipynb

# Check model file
ls -lh models/irrigation_rf_final.pkl

# View training logs
cat logs/irrigation_control/training_log_20251102_125502.json | jq

# Test model loading
python -c "import pickle; m=pickle.load(open('models/irrigation_rf_final.pkl','rb')); print(m)"
```

---

**Last Updated**: November 2, 2025
**Status**: ✅ Production Ready
