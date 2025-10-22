# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Edge-Based ML Solution for Real-Time Decision Support in Small-Scale Farms. This is a Final Year Project developing lightweight ML models optimized for Raspberry Pi 4B deployment, targeting smallholder farms (<2 hectares) that lack reliable internet connectivity.

**Target Hardware**: Raspberry Pi 4B (4GB RAM, ARM Cortex-A72 CPU, <5W power consumption)

## Key Performance Targets

- Inference latency: <1 second per prediction
- Power consumption: ≤5W on Raspberry Pi
- Model size: Edge-friendly (<50MB per model)
- Test accuracy: >95% across all models

## Project Structure

```
├── data/                    # Data directories (gitignored)
│   ├── raw/                # Original datasets (PlantVillage, CCMT, irrigation.csv)
│   ├── processed/          # Train/val/test splits
│   └── custom/             # Test images for validation
├── models/                  # Trained models (gitignored)
├── logs/                    # Training logs and metrics (JSON format)
│   ├── disease_detection/
│   └── irrigation_control/
├── notebooks/              # Jupyter notebooks for ML experimentation
│   ├── 0_poc.ipynb        # Proof of concept - integrated system demo
│   ├── 1_disease_detection.ipynb  # Plant disease detection (binary classification)
│   └── 2_irrigation_control.ipynb # Irrigation control (decision tree)
├── scripts/                # Dataset preparation and utilities
└── src/                    # Production code (minimal - mostly notebooks)
```

## Three ML Pipelines

### 1. Disease Detection (Binary Classification)
- **Model**: MobileNetV2 (frozen) + custom dense layers
- **Framework**: TensorFlow → TensorFlow Lite
- **Input**: 224x224 RGB images (configurable via IMG_SIDE_LENGTH)
- **Output**: Binary classification (Healthy/Diseased)
- **Datasets**: PlantVillage (multi-class), CCMT (cassava)
- **Current performance**: ~98% accuracy, 9.1MB model size
- **Notebook**: `notebooks/1_disease_detection.ipynb`

### 2. Irrigation Control (Decision Tree)
- **Model**: sklearn DecisionTreeClassifier
- **Input**: Soil Moisture, Temperature, Air Humidity (tabular data)
- **Output**: Pump ON/OFF
- **Dataset**: `data/raw/irrigation.csv`
- **Current performance**: >99% accuracy
- **Notebook**: `notebooks/2_irrigation_control.ipynb`

### 3. Crop Recommendation
- **Status**: Simple rule-based logic (in POC)
- **Input**: Season, soil type
- **Output**: Crop suggestions
- **Future**: ML-based recommendation system

## Development Workflow

### Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Working with Jupyter Notebooks

All ML development happens in Jupyter notebooks:

```bash
# Start Jupyter
jupyter notebook

# Or use VS Code with Jupyter extension
code notebooks/
```

**Important**: Notebooks contain both experimentation and training. Each notebook includes:
- Data loading and preprocessing
- Data augmentation (for disease detection)
- Model training with callbacks (EarlyStopping, ReduceLROnPlateau)
- Evaluation and metrics
- Training log generation (saved to `logs/`)
- TFLite conversion (for edge deployment)

### Dataset Preparation Scripts

Located in `scripts/`, these handle dataset splitting and merging:

```bash
# Split PlantVillage dataset (multi-class → train/val/test)
python scripts/split_dataset.py

# Split for binary classification (Healthy vs Diseased)
python scripts/split_dataset_binary.py

# Split CCMT dataset (cassava)
python scripts/split_dataset_ccmt.py
python scripts/split_dataset_binary_ccmt.py

# Merge multiple datasets for binary classification
python scripts/merge_datasets_binary.py

# Move augmented data
python scripts/move_augmented_ccmt.py

# Test GPU availability
python scripts/gpu_test.py
```

**Dataset conventions**:
- Source datasets go in `data/raw/`
- Processed datasets (split) go in `data/processed/`
- Standard split: 80% train, 10% val, 10% test
- Binary classification: all non-healthy classes → "Diseased" class

### Training Models

Training is done via notebooks, not standalone scripts:

1. Open relevant notebook (`notebooks/1_disease_detection.ipynb` or `notebooks/2_irrigation_control.ipynb`)
2. Adjust hyperparameters in the configuration cells at the top
3. Run all cells sequentially
4. Models are saved to `models/`
5. Training logs are saved to `logs/{model_type}/training_log_{timestamp}.json`

**Disease Detection Hyperparameters** (configurable):
- `BATCH_SIZE`: Default 32
- `IMG_SIDE_LENGTH`: Default 128 (affects model input size)
- `EPOCHS`: Default 30 (early stopping may reduce this)
- `LR`: Learning rate, default 1e-4
- `LABEL_MODE`: 'binary' for binary classification

**Irrigation Control Hyperparameters**:
- `MAX_DEPTH`: Default 5 (None for unlimited)
- `MAX_SAMPLE_SPLIT`: Default 2
- `MIN_SAMPLE_LEAF`: Default 1
- `CRITERION`: 'gini' or 'entropy'

### TensorFlow Lite Conversion

Disease detection models are automatically converted to TFLite at the end of training:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(f"{PROJECT_DIR}/models/plant_disease__binary_model.tflite", "wb") as f:
    f.write(tflite_model)
```

### Raspberry Pi Simulation

The POC notebook (`0_poc.ipynb`) simulates Pi 4B constraints:

```python
def apply_pi4b_constraints():
    # Memory limited to 3.2GB (Pi 4B usable RAM)
    memory_limit_gb = 3.2
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))

    # Configure for Pi 4B ARM CPU (4 cores)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(1)
```

## Data Architecture

### Image Data (Disease Detection)
- **PlantVillage**: Multi-class dataset with various plant diseases
- **CCMT**: Cassava-specific disease dataset
- **Custom images**: Test images in `data/custom/` for validation
- **Preprocessing**:
  - Resize to target size (default 128x128 or 224x224)
  - Normalize to [0, 1] via `/255.0`
  - Data augmentation: RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast, RandomTranslation, GaussianNoise

### Tabular Data (Irrigation)
- **irrigation.csv**: Sensor readings (Soil Moisture, Temperature, Air Humidity) → Pump ON/OFF
- **No scaling required**: Decision tree handles raw sensor values
- **Stratified split**: Ensures balanced train/test distribution

## Training Logs

All training runs are logged to JSON files in `logs/{model_type}/`:

```json
{
  "timestamp": "ISO format",
  "hyperparameters": { ... },
  "dataset_info": { ... },
  "training_time": {
    "total_seconds": float,
    "formatted": "HH:MM:SS"
  },
  "results": {
    "final_train_accuracy": float,
    "final_val_accuracy": float,
    "test_accuracy": float,
    "test_f1_score": float,
    ...
  },
  "training_history": { ... }
}
```

These logs are critical for tracking experiments and model performance over time.

## Model Optimization

### Current Optimizations
- **MobileNetV2**: Lightweight base architecture for disease detection
- **Frozen base model**: Transfer learning with frozen ImageNet weights
- **TFLite conversion**: Edge-optimized inference
- **Decision trees**: Inherently lightweight for tabular data

### Future Optimizations (Not Yet Implemented)
- **Quantization**: Post-training quantization (int8) for TFLite models
- **Pruning**: Remove low-magnitude weights
- **Knowledge distillation**: Train smaller student models

## Git Workflow

```bash
# Check status (models, data, images are gitignored)
git status

# Common workflow
git add notebooks/ scripts/ README.md requirements.txt
git commit -m "Descriptive message"
git push origin master
```

**Gitignored items**:
- `data/`, `models/`, `*.h5`, `*.tflite` (large files)
- `.venv/` (Python virtual environment)
- `__pycache__/`, `*.pyc` (Python cache)
- `*.jpg`, `*.JPG`, `*.csv` (datasets and images)

## Testing Models

### Disease Detection
```python
from keras.utils import load_img, img_to_array
import numpy as np

# Load and preprocess image
image = load_img("path/to/image.jpg", target_size=(128, 128))
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0) / 255.0

# Predict
predictions = model.predict(image_array)
predicted_class = (predictions[0][0] > 0.5).astype(int)  # Binary threshold
```

### Irrigation Control
```python
# Example: [Soil Moisture, Temperature, Air Humidity]
sample = [[700, 30, 65]]
prediction = clf.predict(sample)
print("Pump ON" if prediction[0] == 1 else "Pump OFF")
```

## Key Dependencies

- **tensorflow**: ML framework
- **keras**: High-level neural networks API
- **scikit-learn**: Traditional ML algorithms (decision trees)
- **opencv-python**: Image processing
- **matplotlib**: Plotting and visualization
- **pandas**: Tabular data manipulation
- **jupyter**: Interactive notebooks
- **psutil**: System resource monitoring

## Research Context

- **Target users**: Smallholder farmers in Sri Lanka (<2 hectares)
- **Problem**: 40% of agricultural output lost due to inefficiencies
- **Constraints**: No reliable internet, low cost, low technical literacy
- **Solution**: Offline, edge-based ML models on low-power hardware
- **Methodology**: RAD (Rapid Application Development), Pragmatism philosophy
- **Evaluation**: Accuracy, Precision, Recall, F1, inference time, memory footprint, power usage

## Important Notes for Development

1. **Always work in notebooks**: The primary development environment is Jupyter notebooks, not standalone Python scripts
2. **Check training logs**: Before retraining, review existing logs in `logs/` to understand previous hyperparameter choices
3. **Dataset splits must be reproducible**: Use `random_state=42` for consistency
4. **TFLite conversion is mandatory**: All disease detection models must be converted to TFLite for edge deployment
5. **Resource constraints are real**: Always test with Pi 4B simulation constraints applied
6. **Binary classification**: Current focus is binary (Healthy/Diseased), not multi-class disease identification
7. **Model size matters**: Keep models under 50MB for edge deployment
8. **Inference latency is critical**: Target <1s per prediction, ideally <500ms

## Hardware Deployment (Future)

When deploying to actual Raspberry Pi:
1. Transfer `.tflite` models to Pi
2. Use TensorFlow Lite runtime (not full TensorFlow)
3. Connect sensors (soil moisture, temperature, humidity) via GPIO
4. Integrate camera module for disease detection
5. Run inference loop with sensor readings
6. Control irrigation pump via relay module
