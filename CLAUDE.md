# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Edge-Based ML Solution for Real-Time Decision Support in Small-Scale Farms. Final Year Project developing lightweight ML models optimized for Raspberry Pi 4B deployment, targeting smallholder farms (<2 hectares) in Sri Lanka that lack reliable internet connectivity.

**Target Hardware**: Raspberry Pi 4B (4GB RAM, ARM Cortex-A72 CPU, <5W power consumption)

**Performance Targets**: <1s inference, ≤5W power, <50MB per model, >95% accuracy

## Current Project Status (Nov 2, 2025)

### ✅ Production-Ready Models

1. **Disease Detection**: MobileNetV2 → TFLite (98% accuracy, 9.1 MB)
2. **Irrigation Control**: RandomForest (97.89% accuracy, 50 KB)
   - **Critical**: RandomForest was chosen over DecisionTree's 100% accuracy due to overfitting concerns
   - See `docs/MODEL_SELECTION_RATIONALE.md` for detailed justification

### 🎯 Next Priority

Integrate irrigation model into `notebooks/0_poc.ipynb` to demonstrate complete multi-model system.

## Development Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (TensorFlow, scikit-learn, OpenCV, etc.)
pip install -r requirements.txt

# Launch Jupyter for notebook-based development
jupyter notebook
```

## Notebook-Based Development Workflow

**All ML development happens in Jupyter notebooks, not scripts.** Each notebook is self-contained with:
- Data loading and preprocessing
- Model training with hyperparameters
- Evaluation and metrics
- Model export (TFLite for images, pickle for tabular)
- Automated training log generation (JSON format in `logs/`)

### Key Notebooks (Execution Order)

1. **`notebooks/1_disease_detection.ipynb`** - Binary image classification (Healthy/Diseased)
   - MobileNetV2 transfer learning → TFLite conversion
   - Data augmentation: RandomFlip, RandomRotation, RandomZoom, RandomBrightness
   - Input: 224×224 RGB → configurable via `IMG_SIDE_LENGTH`

2. **`notebooks/3_irrigation_preprocessing.ipynb`** - Feature engineering pipeline
   - Transforms raw sensor data (5 features) → 37 engineered features
   - Rolling statistics (3h/6h/12h), change rates (1h/3h), time/seasonal features
   - Intelligent multi-class labeling: Irrigate_High, Irrigate_Low, No_Irrigation
   - 6-hour sliding window approach, 80/20 chronological split

3. **`notebooks/4_irrigation_model_comparison.ipynb`** - Model selection
   - Compares DecisionTree (100% acc), RandomForest (97.89%), LogisticRegression
   - Handles class imbalance via SMOTE + class weighting
   - **Important**: DecisionTree's 100% was rejected as overfitting

4. **`notebooks/5_irrigation_rf_deployment.ipynb`** - Production model
   - Trains RandomForest Config1 (50 trees, max_depth=10)
   - Feature importance analysis, confusion matrices
   - Saves `models/irrigation_rf_final.pkl` + deployment metadata

5. **`notebooks/0_poc.ipynb`** - Proof of concept demo
   - **Current**: Disease detection only
   - **Next**: Add irrigation control section (Priority 1)
   - Simulates Pi 4B constraints (3.2GB RAM, 4-core CPU threading)

## Loading Production Models

### Disease Detection (TFLite)
```python
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/plant_disease__binary_model.tflite")
interpreter.allocate_tensors()

# Preprocess: resize to 224×224, normalize to [0,1]
image_array = np.expand_dims(image_array, axis=0) / 255.0

# Predict
interpreter.set_tensor(input_details[0]['index'], image_array)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
```

### Irrigation Control (RandomForest)
```python
import pickle
import numpy as np

# Load model + preprocessing artifacts
with open('models/irrigation_rf_final.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/processed/irrigation/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Input: (1, 222) - 6-hour window × 37 features (already scaled)
prediction = model.predict(X_test[0:1])[0]  # Returns: 'Irrigate_High', 'Irrigate_Low', or 'No_Irrigation'
probabilities = model.predict_proba(X_test[0:1])[0]
```

## Data Pipeline Architecture

### Image Data (Disease Detection)
- **Datasets**: PlantVillage (multi-class), CCMT (cassava) in `data/raw/`
- **Preprocessing**: Binary classification (Healthy vs all Diseased classes merged)
- **Splits**: 80% train, 10% val, 10% test via `scripts/split_dataset_binary.py`
- **Augmentation**: Applied during training via `tf.keras.Sequential` layers

### Tabular Data (Irrigation)
- **Dataset**: `data/raw/soil_and_rain_data.csv` (8,761 hourly readings)
- **Raw features**: Humidity, Atmospheric_Temp, Soil_Temp, Soil_Moisture, Dew_Point
- **Engineered features** (37 total):
  - Rolling statistics: 3h/6h/12h means and stds for moisture/humidity/temp
  - Change rates: 1h/3h differences (captures drying/wetting trends)
  - Time features: hour_of_day, day_of_week, is_daytime (06:00-18:00), cyclical encoding (hour_sin, hour_cos)
  - Seasonal: is_yala_season (May-Aug vs Sep-Mar for Sri Lankan agriculture)
  - Rainfall proxy: humidity_spike (>10% increase/hour), recent_rain (spike in last 3h)
- **Windowing**: 6-hour sliding window (stride=1h) → 222 features (6 × 37)
- **Scaling**: StandardScaler (saved as `scaler.pkl` for deployment)
- **Split**: Chronological 80/20 (prevents temporal leakage)

## Intelligent Irrigation Labeling Logic

The multi-class labels are generated programmatically using domain-aware rules:

```
Priority 1: Recent rainfall → No_Irrigation (prevents over-irrigation)
Priority 2: Soil moisture + Temperature + Time of day
  - Very dry (<P25) + Hot (>P75) + Midday (9-15h) → Irrigate_Low (avoid leaf scorch)
  - Very dry (<P25) + Hot (>P75) + Non-midday → Irrigate_High
  - Moderately dry (P25-P50) → Irrigate_Low
  - Sufficient (>P50) → No_Irrigation
```

This context-aware approach considers:
- Water efficiency (post-rain prevention)
- Plant physiology (midday irrigation stress avoidance)
- Adaptive thresholds (quantile-based, not hardcoded)

## Model Selection Decision (Critical)

**RandomForest was chosen over DecisionTree despite lower accuracy:**

| Model | Test Accuracy | F1-score | Features Used | Production Risk |
|-------|--------------|----------|---------------|-----------------|
| DecisionTree | 100.00% | 100.00% | 3 (98.65% weight) | ⚠️ HIGH (overfitting) |
| RandomForest | 97.89% | 92.22% | 10+ distributed | ✅ LOW (ensemble robust) |

**Rationale**: 100% accuracy on real sensor data indicates overfitting/data leakage. RandomForest provides:
- Ensemble robustness (50 trees, majority voting)
- Broader feature utilization (reduces brittleness)
- Production reliability (graceful degradation under noise)
- Still edge-compatible (0.146 ms << 1000 ms target)

See `docs/MODEL_SELECTION_RATIONALE.md` for full analysis. This decision demonstrates production mindset and is a strong FYP defense point.

## Training Log Format

All training runs generate JSON logs in `logs/{model_type}/training_log_{timestamp}.json`:

```json
{
  "timestamp": "ISO format",
  "model_info": {"model_type": "...", "hyperparameters": {...}},
  "dataset_info": {"train_samples": int, "class_distribution": {...}},
  "performance_metrics": {
    "test_accuracy": float,
    "test_f1_macro": float,
    "per_class_metrics": {...},
    "confusion_matrix": [[...]]
  },
  "efficiency_metrics": {
    "train_time_seconds": float,
    "inference_time_ms_per_sample": float,
    "edge_compatible": bool
  },
  "feature_importance": {"top_10_features": [...]}
}
```

**Before retraining**: Always review existing logs to understand previous hyperparameter choices.

## Dataset Preparation Scripts

Located in `scripts/` for splitting and merging datasets:

```bash
# Binary classification splits (Healthy vs Diseased)
python scripts/split_dataset_binary.py        # PlantVillage
python scripts/split_dataset_binary_ccmt.py   # CCMT cassava

# Merge multiple datasets
python scripts/merge_datasets_binary.py

# Test GPU availability
python scripts/gpu_test.py
```

Conventions:
- Source datasets: `data/raw/`
- Processed splits: `data/processed/` (80/10/10 or 80/20)
- Use `random_state=42` for reproducibility

## Edge Deployment Constraints

### Raspberry Pi 4B Simulation (in POC)
```python
import resource
import tensorflow as tf

# Memory limit: 3.2GB (Pi 4B usable RAM)
memory_limit_gb = 3.2
memory_limit_bytes = int(memory_limit_gb * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))

# CPU threading: 4 cores (ARM Cortex-A72)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(1)
```

### Performance Validation
- **Disease detection**: ~10 ms inference (TFLite optimized)
- **Irrigation control**: 0.146 ms inference (lightweight RandomForest)
- **Combined system**: <200 ms total (well under 1s target)

## Git Workflow

```bash
# Check status (models/data/images are gitignored)
git status

# Standard workflow
git add notebooks/ scripts/ docs/ README.md requirements.txt
git commit -m "Descriptive message

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
git push origin master
```

**Gitignored**: `data/`, `models/`, `*.h5`, `*.tflite`, `.venv/`, `__pycache__/`, `*.jpg`, `*.csv`

## Important Development Notes

1. **Notebook-first workflow**: Primary development is in Jupyter, not Python scripts
2. **Reproducibility**: Always use `random_state=42` for splits and training
3. **TFLite conversion mandatory**: Disease detection models must export to TFLite
4. **Class imbalance**: Irrigation has 89% No_Irrigation class - use SMOTE + class weighting
5. **Model size constraints**: <50MB per model for edge deployment
6. **Binary classification focus**: Current scope is Healthy/Diseased, not multi-class disease identification
7. **Temporal data caution**: 6-hour sliding window may cause train/test leakage - RandomForest mitigates this better than DecisionTree
8. **Production mindset**: Prioritize reliability over theoretical accuracy (97.89% > suspicious 100%)

## Research Context

**Target users**: Smallholder farmers in Sri Lanka (<2 hectares)
**Problem**: 40% agricultural output lost due to inefficiencies
**Constraints**: No reliable internet, low cost, low technical literacy
**Solution**: Offline edge ML on Raspberry Pi 4B
**Methodology**: RAD (Rapid Application Development), Pragmatism philosophy
**Evaluation**: Accuracy, Precision, Recall, F1, inference time, memory footprint, power (<5W)

## Key Reference Documents

- **`DEVELOPMENT_HANDOFF.md`**: Full project context and next steps
- **`CLAUDE_CODE_CONTEXT.md`**: Quick context for Claude Code sessions
- **`QUICK_START.md`**: Instant resume instructions
- **`docs/MODEL_SELECTION_RATIONALE.md`**: Why RandomForest over DecisionTree (critical for FYP defense)
- **`docs/IRRIGATION_MODEL_SUMMARY.md`**: Model specifications and deployment guide

## Future Hardware Deployment

When deploying to actual Raspberry Pi:
1. Transfer TFLite models + RandomForest pickle to Pi
2. Use TensorFlow Lite runtime (not full TensorFlow)
3. Connect sensors via GPIO: soil moisture, temperature, humidity
4. Integrate camera module for disease detection
5. Control irrigation pump via relay module
6. Run inference loop with hourly sensor readings (6-hour buffer for irrigation)
