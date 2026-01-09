#!/usr/bin/env python3
"""
Edge Orchestrator for Agro Edge AI System
==========================================

This script orchestrates three independently trained ML models for edge deployment
on a Raspberry Pi 4B, demonstrating real-world agricultural decision support.

Models:
    1. Disease Detection (TFLite) - Plant leaf image → disease classification
    2. Irrigation Control (RandomForest) - Sensor data → irrigation decision  
    3. Crop Suitability (TFLite) - Soil/environment → tomato suitability score

Fusion Strategy:
    Late fusion with confidence-aware decision making. Disease detection results
    influence irrigation decisions (fungal diseases require reduced watering).
    Low-confidence predictions trigger conservative fallback behaviors.

Edge Constraints:
    - Single-process, synchronous execution
    - Minimal memory footprint (lazy loading)
    - No GPU dependencies
    - Fast startup (<2s target)

Author: Agro Edge AI Project
Target: Raspberry Pi 4B (ARMv8, 4GB RAM)
"""

import json
import sys
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf


import numpy as np

# Suppress sklearn version warnings (common on Pi with different versions)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION - Mock sensor inputs (replace with real sensor reads in production)
# =============================================================================

# Path to test image for disease detection
IMAGE_PATH = "./data/custom/image.png"

# Simulated sensor readings (realistic values for Sri Lankan agriculture)
SENSOR_DATA = {
    # Current environmental readings
    "humidity": 72.5,              # % relative humidity
    "atmospheric_temp": 28.3,      # °C
    "soil_temp": 26.1,             # °C  
    "soil_moisture": 45.2,         # Soil moisture sensor value
    "dew_point": 22.4,             # °C
    
    # Time context (for irrigation model)
    "hour_of_day": 14,             # 24h format
    "day_of_week": 2,              # 0=Monday
    
    # Soil nutrients (for crop suitability)
    "nitrogen": 120.0,             # kg/ha
    "phosphorus": 65.0,            # kg/ha
    "potassium": 155.0,            # kg/ha
    "soil_ph": 6.4,                # pH value
    "rainfall": 850.0,             # mm annual
}

# Historical data for rolling features (last 12 hours, hourly readings)
# In production, this would come from a time-series database or ring buffer
HISTORICAL_DATA = {
    "soil_moisture": [48.1, 47.5, 46.8, 46.2, 45.9, 45.5, 45.3, 45.2, 45.2, 45.1, 45.2, 45.2],
    "humidity": [68.2, 69.1, 70.3, 71.2, 71.8, 72.1, 72.3, 72.4, 72.5, 72.5, 72.5, 72.5],
    "atmospheric_temp": [24.1, 25.2, 26.1, 26.8, 27.3, 27.8, 28.0, 28.2, 28.3, 28.3, 28.3, 28.3],
}

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"

# Model files
DISEASE_MODEL_PATH = MODELS_DIR / "tomato_disease_multiclass_model.tflite"
IRRIGATION_MODEL_PATH = MODELS_DIR / "irrigation_rf_final.pkl"
IRRIGATION_SCALER_PATH = DATA_DIR / "processed" / "irrigation" / "scaler.pkl"
IRRIGATION_FEATURES_PATH = DATA_DIR / "processed" / "irrigation" / "feature_names.json"
CROP_MODEL_PATH = MODELS_DIR / "tomato_suitability.tflite"
CROP_SCALER_PATH = MODELS_DIR / "tomato_scaler.pkl"

# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================

# Disease detection: Below this confidence, we flag as "uncertain"
DISEASE_CONFIDENCE_THRESHOLD = 0.60

# Disease detection: Below this, we refuse to make a definitive call
DISEASE_MIN_CONFIDENCE = 0.35

# Irrigation: Probability threshold for multi-class decision
IRRIGATION_CONFIDENCE_THRESHOLD = 0.50

# Crop suitability: Score boundaries
CROP_SUITABLE_THRESHOLD = 0.65
CROP_MARGINAL_THRESHOLD = 0.40

# =============================================================================
# DISEASE CLASSES (from training notebook - order matches model output)
# =============================================================================

DISEASE_CLASSES = [
    "Healthy",
    "Leaf_Mold",
    "Leaf_blight", 
    "Septoria_leaf_spot",
    "Verticillium_wilt",
    "Yellow_Leaf_Curl_Virus"
]

# Diseases that are fungal in nature (affects irrigation strategy)
# Rationale: Fungal pathogens thrive in moist conditions, so we reduce watering
FUNGAL_DISEASES = {"Leaf_Mold", "Septoria_leaf_spot", "Verticillium_wilt"}

# Viral diseases (different management, not affected by irrigation)
VIRAL_DISEASES = {"Yellow_Leaf_Curl_Virus"}

# =============================================================================
# LAZY MODEL LOADERS
# =============================================================================

class LazyModelLoader:
    """
    Lazy loader for ML models to minimize startup memory footprint.
    Models are only loaded when first accessed.
    
    This is critical for edge deployment where:
    - RAM is limited (4GB shared with OS)
    - Not all models may be needed for every inference cycle
    - Fast startup is preferred over fast first-inference
    """
    
    def __init__(self):
        self._disease_interpreter = None
        self._irrigation_model = None
        self._irrigation_scaler = None
        self._irrigation_features = None
        self._crop_interpreter = None
        self._crop_scaler = None
    
    @property
    def disease_interpreter(self):
        """Load TFLite interpreter for disease detection on first access."""
        if self._disease_interpreter is None:
            import tensorflow as tf
            self._disease_interpreter = tf.lite.Interpreter(
                model_path=str(DISEASE_MODEL_PATH)
            )
            self._disease_interpreter.allocate_tensors()
        return self._disease_interpreter
    
    @property
    def irrigation_model(self):
        """Load RandomForest irrigation model on first access."""
        if self._irrigation_model is None:
            with open(IRRIGATION_MODEL_PATH, "rb") as f:
                self._irrigation_model = pickle.load(f)
        return self._irrigation_model
    
    @property
    def irrigation_scaler(self):
        """Load StandardScaler for irrigation features on first access."""
        if self._irrigation_scaler is None:
            with open(IRRIGATION_SCALER_PATH, "rb") as f:
                self._irrigation_scaler = pickle.load(f)
        return self._irrigation_scaler
    
    @property
    def irrigation_features(self):
        """Load feature names for irrigation model on first access."""
        if self._irrigation_features is None:
            with open(IRRIGATION_FEATURES_PATH, "r") as f:
                self._irrigation_features = json.load(f)
        return self._irrigation_features
    
    @property
    def crop_interpreter(self):
        """Load TFLite interpreter for crop suitability on first access."""
        if self._crop_interpreter is None:
            import tensorflow as tf
            self._crop_interpreter = tf.lite.Interpreter(
                model_path=str(CROP_MODEL_PATH)
            )
            self._crop_interpreter.allocate_tensors()
        return self._crop_interpreter
    
    @property
    def crop_scaler(self):
        """Load StandardScaler for crop features on first access."""
        if self._crop_scaler is None:
            import joblib
            self._crop_scaler = joblib.load(CROP_SCALER_PATH)
        return self._crop_scaler


# Global lazy loader instance
models = LazyModelLoader()

# =============================================================================
# DISEASE DETECTION INFERENCE
# =============================================================================

def load_and_preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load and preprocess image for disease detection model.
    
    Preprocessing matches training notebook (1_disease_detection.ipynb):
    - Resize to 224x224 (MobileNetV2 input size)
    - Convert to float32
    - Note: Rescaling (1/255) is built into the model's first layer
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image array of shape (1, 224, 224, 3) or None if failed
    """
    try:
        # Use PIL for image loading (lighter than OpenCV for edge)
        from PIL import Image
        
        img = Image.open(image_path)
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize to model input size
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and add batch dimension
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Note: Do NOT normalize here - the model has Rescaling(1./255) as first layer
        # This matches the training notebook exactly
        
        return img_array
        
    except Exception as e:
        return None


def run_disease_detection(image_path: str) -> Dict[str, Any]:
    """
    Run disease detection inference on a plant leaf image.
    
    Returns a dict with:
    - predicted_class: Most likely disease (or "Healthy")
    - confidence: Probability of predicted class
    - all_probabilities: Full probability distribution
    - confidence_level: "high", "medium", "low" based on thresholds
    - is_fungal: Whether detected disease is fungal (affects irrigation)
    """
    result = {
        "status": "error",
        "predicted_class": None,
        "confidence": 0.0,
        "all_probabilities": {},
        "confidence_level": "none",
        "is_fungal": False,
        "is_viral": False,
        "error": None
    }
    
    # Check if image exists
    if not os.path.exists(image_path):
        result["error"] = f"Image not found: {image_path}"
        return result
    
    # Load and preprocess image
    img_array = load_and_preprocess_image(image_path)
    if img_array is None:
        result["error"] = "Failed to load or preprocess image"
        return result
    
    try:
        # Get interpreter and tensor details
        interpreter = models.disease_interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]["index"], img_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get output probabilities (softmax output from model)
        predictions = interpreter.get_tensor(output_details[0]["index"])[0]
        
        # Find predicted class
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        predicted_class = DISEASE_CLASSES[predicted_idx]
        
        # Build probability distribution
        all_probs = {
            DISEASE_CLASSES[i]: float(predictions[i]) 
            for i in range(len(DISEASE_CLASSES))
        }
        
        # Determine confidence level
        if confidence >= DISEASE_CONFIDENCE_THRESHOLD:
            confidence_level = "high"
        elif confidence >= DISEASE_MIN_CONFIDENCE:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Check disease type
        is_fungal = predicted_class in FUNGAL_DISEASES
        is_viral = predicted_class in VIRAL_DISEASES
        
        result.update({
            "status": "success",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()},
            "confidence_level": confidence_level,
            "is_fungal": is_fungal,
            "is_viral": is_viral,
            "error": None
        })
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# =============================================================================
# IRRIGATION CONTROL INFERENCE
# =============================================================================

def compute_rolling_features(
    values: List[float], 
    windows: List[int] = [3, 6, 12]
) -> Dict[str, float]:
    """
    Compute rolling mean and std for a time series.
    
    This matches the feature engineering in 3_irrigation_preprocessing.ipynb.
    Windows are in hours (assuming hourly data).
    """
    features = {}
    arr = np.array(values)
    
    for window in windows:
        if len(arr) >= window:
            window_data = arr[-window:]
            features[f"rolling_mean_{window}h"] = float(np.mean(window_data))
            features[f"rolling_std_{window}h"] = float(np.std(window_data))
        else:
            # Fallback for insufficient data
            features[f"rolling_mean_{window}h"] = float(np.mean(arr))
            features[f"rolling_std_{window}h"] = float(np.std(arr)) if len(arr) > 1 else 0.0
    
    return features


def compute_change_features(values: List[float]) -> Dict[str, float]:
    """
    Compute rate of change features (1h and 3h differences).
    
    Matches 3_irrigation_preprocessing.ipynb feature engineering.
    """
    arr = np.array(values)
    
    features = {
        "change_1h": float(arr[-1] - arr[-2]) if len(arr) >= 2 else 0.0,
        "change_3h": float(arr[-1] - arr[-4]) if len(arr) >= 4 else 0.0,
    }
    
    return features


def engineer_irrigation_features(
    sensor_data: Dict[str, float],
    historical_data: Dict[str, List[float]]
) -> np.ndarray:
    """
    Engineer all features required by the irrigation model.
    
    The model expects 37 features per timestep, with a window of 6 timesteps.
    Total input shape: (1, 6 * 37) = (1, 222) flattened.
    
    Feature engineering logic from 3_irrigation_preprocessing.ipynb:
    1. Base sensor readings (5): Humidity, Atmospheric_Temp, Soil_Temp, Soil_Moisture, Dew_Point
    2. Rolling features (18): 3h, 6h, 12h windows for Soil_Moisture, Humidity, Atmospheric_Temp
    3. Change features (6): 1h and 3h changes for Soil_Moisture, Humidity, Atmospheric_Temp
    4. Time features (6): hour_of_day, day_of_week, is_daytime, hour_sin, hour_cos, is_yala_season
    5. Binary features (2): humidity_spike, recent_rain
    
    Total: 37 features
    """
    feature_names = models.irrigation_features
    
    # Current sensor values
    humidity = sensor_data["humidity"]
    atm_temp = sensor_data["atmospheric_temp"]
    soil_temp = sensor_data["soil_temp"]
    soil_moisture = sensor_data["soil_moisture"]
    dew_point = sensor_data["dew_point"]
    hour = sensor_data["hour_of_day"]
    day_of_week = sensor_data["day_of_week"]
    
    # Compute rolling features for each variable
    sm_rolling = compute_rolling_features(historical_data["soil_moisture"])
    hum_rolling = compute_rolling_features(historical_data["humidity"])
    temp_rolling = compute_rolling_features(historical_data["atmospheric_temp"])
    
    # Compute change features
    sm_change = compute_change_features(historical_data["soil_moisture"])
    hum_change = compute_change_features(historical_data["humidity"])
    temp_change = compute_change_features(historical_data["atmospheric_temp"])
    
    # Time-based features
    is_daytime = 1 if 6 <= hour < 18 else 0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Season (Sri Lankan: Yala = May-Aug)
    current_month = datetime.now().month
    is_yala_season = 1 if 5 <= current_month <= 8 else 0
    
    # Rainfall proxy: humidity spike detection
    # A spike is >10% increase in 1 hour (from preprocessing notebook)
    humidity_spike = 1 if hum_change["change_1h"] > 10 else 0
    
    # Recent rain: any humidity spike in last 3 hours
    recent_rain = 0
    if len(historical_data["humidity"]) >= 4:
        for i in range(-3, 0):
            if i + 1 < 0:
                diff = historical_data["humidity"][i + 1] - historical_data["humidity"][i]
                if diff > 10:
                    recent_rain = 1
                    break
    
    # Build feature vector in exact order expected by model
    # Order from feature_names.json
    feature_dict = {
        "Humidity": humidity,
        "Atmospheric_Temp": atm_temp,
        "Soil_Temp": soil_temp,
        "Soil_Moisture": soil_moisture,
        "Dew_Point": dew_point,
        "Soil_Moisture_rolling_mean_3h": sm_rolling["rolling_mean_3h"],
        "Soil_Moisture_rolling_std_3h": sm_rolling["rolling_std_3h"],
        "Soil_Moisture_rolling_mean_6h": sm_rolling["rolling_mean_6h"],
        "Soil_Moisture_rolling_std_6h": sm_rolling["rolling_std_6h"],
        "Soil_Moisture_rolling_mean_12h": sm_rolling["rolling_mean_12h"],
        "Soil_Moisture_rolling_std_12h": sm_rolling["rolling_std_12h"],
        "Humidity_rolling_mean_3h": hum_rolling["rolling_mean_3h"],
        "Humidity_rolling_std_3h": hum_rolling["rolling_std_3h"],
        "Humidity_rolling_mean_6h": hum_rolling["rolling_mean_6h"],
        "Humidity_rolling_std_6h": hum_rolling["rolling_std_6h"],
        "Humidity_rolling_mean_12h": hum_rolling["rolling_mean_12h"],
        "Humidity_rolling_std_12h": hum_rolling["rolling_std_12h"],
        "Atmospheric_Temp_rolling_mean_3h": temp_rolling["rolling_mean_3h"],
        "Atmospheric_Temp_rolling_std_3h": temp_rolling["rolling_std_3h"],
        "Atmospheric_Temp_rolling_mean_6h": temp_rolling["rolling_mean_6h"],
        "Atmospheric_Temp_rolling_std_6h": temp_rolling["rolling_std_6h"],
        "Atmospheric_Temp_rolling_mean_12h": temp_rolling["rolling_mean_12h"],
        "Atmospheric_Temp_rolling_std_12h": temp_rolling["rolling_std_12h"],
        "Soil_Moisture_change_1h": sm_change["change_1h"],
        "Soil_Moisture_change_3h": sm_change["change_3h"],
        "Atmospheric_Temp_change_1h": temp_change["change_1h"],
        "Atmospheric_Temp_change_3h": temp_change["change_3h"],
        "Humidity_change_1h": hum_change["change_1h"],
        "Humidity_change_3h": hum_change["change_3h"],
        "hour_of_day": hour,
        "day_of_week": day_of_week,
        "is_daytime": is_daytime,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_yala_season": is_yala_season,
        "humidity_spike": humidity_spike,
        "recent_rain": recent_rain,
    }
    
    # Create feature vector in correct order
    feature_vector = np.array([feature_dict[name] for name in feature_names], dtype=np.float32)
    
    # Apply scaling to features that require it
    # From scaling_metadata.json: binary and cyclical features are excluded
    features_to_scale_mask = np.array([
        name not in ["is_daytime", "is_yala_season", "humidity_spike", "recent_rain", "hour_sin", "hour_cos"]
        for name in feature_names
    ])
    
    # Scale the appropriate features
    scaler = models.irrigation_scaler
    scalable_features = feature_vector[features_to_scale_mask].reshape(1, -1)
    scaled_features = scaler.transform(scalable_features)[0]
    feature_vector[features_to_scale_mask] = scaled_features
    
    # The model expects a window of 6 timesteps (from deployment metadata)
    # For simplicity, we replicate the current feature vector across the window
    # In production, this would use actual historical feature vectors
    window_size = 6
    windowed_features = np.tile(feature_vector, window_size)
    
    return windowed_features.reshape(1, -1)


def run_irrigation_control(
    sensor_data: Dict[str, float],
    historical_data: Dict[str, List[float]],
    disease_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run irrigation control inference.
    
    Returns a dict with:
    - decision: "Irrigate_High", "Irrigate_Low", or "No_Irrigation"
    - probabilities: Class probabilities from RandomForest
    - confidence: Probability of chosen class
    - disease_adjusted: Whether decision was modified due to disease
    
    Disease Override Logic:
    ----------------------
    If a fungal disease is detected with sufficient confidence, we apply
    conservative irrigation adjustments:
    
    - Fungal diseases (Leaf_Mold, Septoria_leaf_spot, Verticillium_wilt) thrive
      in moist conditions. Reducing irrigation can help slow disease spread.
    
    - If model says "Irrigate_High" but fungal disease detected:
      → Downgrade to "Irrigate_Low" (unless soil is critically dry)
    
    - If model says "Irrigate_Low" but fungal disease detected:
      → Consider "No_Irrigation" (unless plant stress is imminent)
    
    This implements a conservative, safety-first approach that prioritizes
    disease management over aggressive irrigation.
    """
    result = {
        "status": "error",
        "decision": None,
        "original_decision": None,
        "probabilities": {},
        "confidence": 0.0,
        "disease_adjusted": False,
        "adjustment_reason": None,
        "error": None
    }
    
    try:
        # Engineer features
        features = engineer_irrigation_features(sensor_data, historical_data)
        
        # Get model and predict
        model = models.irrigation_model
        
        # Get class probabilities
        probabilities = model.predict_proba(features)[0]
        classes = model.classes_
        
        # Get predicted class and confidence
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        decision = classes[predicted_idx]
        original_decision = decision
        
        # Build probability dict
        prob_dict = {
            classes[i]: round(float(probabilities[i]), 4) 
            for i in range(len(classes))
        }
        
        # =====================================================================
        # DISEASE-AWARE ADJUSTMENT LOGIC
        # =====================================================================
        # Rationale: Fungal pathogens spread faster in wet conditions.
        # If we detect a fungal disease, we conservatively reduce irrigation
        # to create less favorable conditions for pathogen growth.
        # This is a domain-informed safety mechanism, not a random heuristic.
        # =====================================================================
        
        disease_adjusted = False
        adjustment_reason = None
        
        if disease_override == "fungal":
            # Check current soil moisture to avoid plant death
            soil_moisture = sensor_data["soil_moisture"]
            
            # Critical threshold: below this, plant stress is more dangerous than disease
            # This value is conservative; real deployment would use crop-specific thresholds
            CRITICAL_MOISTURE_THRESHOLD = 25.0
            
            if decision == "Irrigate_High":
                if soil_moisture > CRITICAL_MOISTURE_THRESHOLD:
                    # Soil is not critically dry - safe to reduce irrigation
                    decision = "Irrigate_Low"
                    disease_adjusted = True
                    adjustment_reason = (
                        "Downgraded from Irrigate_High to Irrigate_Low due to fungal "
                        "disease detection. Reducing moisture inhibits fungal spread."
                    )
                else:
                    # Soil is critically dry - plant survival takes priority
                    adjustment_reason = (
                        "Fungal disease detected, but soil moisture is critically low. "
                        "Maintaining Irrigate_High to prevent plant death."
                    )
            
            elif decision == "Irrigate_Low":
                if soil_moisture > CRITICAL_MOISTURE_THRESHOLD + 10:
                    # Some buffer above critical - safe to skip irrigation
                    decision = "No_Irrigation"
                    disease_adjusted = True
                    adjustment_reason = (
                        "Changed from Irrigate_Low to No_Irrigation due to fungal "
                        "disease detection. Drier conditions help control spread."
                    )
                else:
                    adjustment_reason = (
                        "Fungal disease detected, but soil moisture is borderline. "
                        "Maintaining Irrigate_Low as a compromise."
                    )
        
        result.update({
            "status": "success",
            "decision": decision,
            "original_decision": original_decision,
            "probabilities": prob_dict,
            "confidence": round(confidence, 4),
            "disease_adjusted": disease_adjusted,
            "adjustment_reason": adjustment_reason,
            "error": None
        })
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# =============================================================================
# CROP SUITABILITY INFERENCE
# =============================================================================

def run_crop_suitability(sensor_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Run tomato suitability inference.
    
    Returns a dict with:
    - suitability_score: 0-100 scale
    - status: "Suitable", "Marginal", or "Not Suitable"
    - limiting_factors: List of parameters outside optimal range
    
    The model was trained on augmented tomato data with optimal ranges
    learned from the training distribution (see 6_crop_recommendation.ipynb).
    """
    result = {
        "status": "error",
        "suitability_score": 0.0,
        "suitability_status": None,
        "limiting_factors": [],
        "error": None
    }
    
    try:
        # Prepare input features in correct order
        # From notebook: N, P, K, temperature, humidity, ph, rainfall
        input_values = np.array([[
            sensor_data["nitrogen"],
            sensor_data["phosphorus"],
            sensor_data["potassium"],
            sensor_data["atmospheric_temp"],
            sensor_data["humidity"],
            sensor_data["soil_ph"],
            sensor_data["rainfall"]
        ]], dtype=np.float32)
        
        # Scale features using the saved scaler
        scaler = models.crop_scaler
        input_scaled = scaler.transform(input_values).astype(np.float32)
        
        # Get interpreter and run inference
        interpreter = models.crop_interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]["index"], input_scaled)
        interpreter.invoke()
        
        # Get output (sigmoid, 0-1)
        output = interpreter.get_tensor(output_details[0]["index"])[0][0]
        suitability_score = float(output) * 100  # Convert to percentage
        
        # Determine status
        if suitability_score >= CROP_SUITABLE_THRESHOLD * 100:
            suitability_status = "Suitable"
        elif suitability_score >= CROP_MARGINAL_THRESHOLD * 100:
            suitability_status = "Marginal"
        else:
            suitability_status = "Not_Suitable"
        
        # =====================================================================
        # LIMITING FACTOR ANALYSIS
        # =====================================================================
        # Compare current values against known optimal ranges for tomatoes.
        # These ranges are derived from agricultural literature and the
        # training data distribution (see TOMATO_OPTIMAL_CONDITIONS.md).
        # =====================================================================
        
        TOMATO_OPTIMAL_RANGES = {
            "nitrogen": (100, 150),      # kg/ha
            "phosphorus": (50, 90),      # kg/ha
            "potassium": (140, 200),     # kg/ha
            "temperature": (20, 28),     # °C
            "humidity": (60, 80),        # %
            "ph": (6.0, 7.0),            # pH
            "rainfall": (600, 1200),     # mm/year
        }
        
        limiting_factors = []
        
        # Check nitrogen
        n = sensor_data["nitrogen"]
        n_range = TOMATO_OPTIMAL_RANGES["nitrogen"]
        if n < n_range[0]:
            limiting_factors.append(f"Nitrogen too low ({n:.1f} kg/ha, optimal: {n_range[0]}-{n_range[1]})")
        elif n > n_range[1]:
            limiting_factors.append(f"Nitrogen too high ({n:.1f} kg/ha, optimal: {n_range[0]}-{n_range[1]})")
        
        # Check phosphorus
        p = sensor_data["phosphorus"]
        p_range = TOMATO_OPTIMAL_RANGES["phosphorus"]
        if p < p_range[0]:
            limiting_factors.append(f"Phosphorus too low ({p:.1f} kg/ha, optimal: {p_range[0]}-{p_range[1]})")
        elif p > p_range[1]:
            limiting_factors.append(f"Phosphorus too high ({p:.1f} kg/ha, optimal: {p_range[0]}-{p_range[1]})")
        
        # Check potassium
        k = sensor_data["potassium"]
        k_range = TOMATO_OPTIMAL_RANGES["potassium"]
        if k < k_range[0]:
            limiting_factors.append(f"Potassium too low ({k:.1f} kg/ha, optimal: {k_range[0]}-{k_range[1]})")
        elif k > k_range[1]:
            limiting_factors.append(f"Potassium too high ({k:.1f} kg/ha, optimal: {k_range[0]}-{k_range[1]})")
        
        # Check temperature
        temp = sensor_data["atmospheric_temp"]
        temp_range = TOMATO_OPTIMAL_RANGES["temperature"]
        if temp < temp_range[0]:
            limiting_factors.append(f"Temperature too low ({temp:.1f}°C, optimal: {temp_range[0]}-{temp_range[1]})")
        elif temp > temp_range[1]:
            limiting_factors.append(f"Temperature too high ({temp:.1f}°C, optimal: {temp_range[0]}-{temp_range[1]})")
        
        # Check humidity
        hum = sensor_data["humidity"]
        hum_range = TOMATO_OPTIMAL_RANGES["humidity"]
        if hum < hum_range[0]:
            limiting_factors.append(f"Humidity too low ({hum:.1f}%, optimal: {hum_range[0]}-{hum_range[1]})")
        elif hum > hum_range[1]:
            limiting_factors.append(f"Humidity too high ({hum:.1f}%, optimal: {hum_range[0]}-{hum_range[1]})")
        
        # Check pH
        ph = sensor_data["soil_ph"]
        ph_range = TOMATO_OPTIMAL_RANGES["ph"]
        if ph < ph_range[0]:
            limiting_factors.append(f"Soil pH too low ({ph:.1f}, optimal: {ph_range[0]}-{ph_range[1]})")
        elif ph > ph_range[1]:
            limiting_factors.append(f"Soil pH too high ({ph:.1f}, optimal: {ph_range[0]}-{ph_range[1]})")
        
        # Check rainfall
        rain = sensor_data["rainfall"]
        rain_range = TOMATO_OPTIMAL_RANGES["rainfall"]
        if rain < rain_range[0]:
            limiting_factors.append(f"Rainfall too low ({rain:.1f} mm, optimal: {rain_range[0]}-{rain_range[1]})")
        elif rain > rain_range[1]:
            limiting_factors.append(f"Rainfall too high ({rain:.1f} mm, optimal: {rain_range[0]}-{rain_range[1]})")
        
        result.update({
            "status": "success",
            "suitability_score": round(suitability_score, 2),
            "suitability_status": suitability_status,
            "limiting_factors": limiting_factors,
            "error": None
        })
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# =============================================================================
# ORCHESTRATION & FUSION LOGIC
# =============================================================================

def generate_final_advice(
    disease_result: Dict[str, Any],
    irrigation_result: Dict[str, Any],
    crop_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate unified advisory output using late fusion of all model results.
    
    Fusion Strategy:
    ----------------
    1. PRIORITY: Safety-critical decisions take precedence
       - Disease management overrides growth optimization
       - Critical moisture levels override disease management
    
    2. CONFIDENCE-AWARE: Low-confidence predictions trigger conservative actions
       - If disease detection confidence is low, don't override irrigation
       - If any model fails, provide partial advice with appropriate caveats
    
    3. CONTEXT-AWARE: Models inform each other
       - Disease type affects irrigation strategy
       - Crop suitability affects long-term recommendations
       - Current conditions determine urgency
    
    4. ACTIONABLE: Output specific, implementable recommendations
       - Not just "irrigate" but when and how much
       - Not just "disease found" but what to do about it
    """
    advice = {
        "summary": "",
        "urgency": "normal",  # "low", "normal", "high", "critical"
        "actions": [],
        "warnings": [],
        "confidence_assessment": "",
    }
    
    actions = []
    warnings = []
    urgency_level = 1  # 1=low, 2=normal, 3=high, 4=critical
    
    # =========================================================================
    # DISEASE ASSESSMENT
    # =========================================================================
    
    if disease_result["status"] == "success":
        disease = disease_result["predicted_class"]
        confidence = disease_result["confidence"]
        confidence_level = disease_result["confidence_level"]
        
        if disease != "Healthy":
            if confidence_level == "high":
                if disease_result["is_fungal"]:
                    actions.append(
                        f"DISEASE ACTION: {disease} detected (fungal). "
                        f"Apply appropriate fungicide. Reduce irrigation frequency. "
                        f"Improve air circulation around plants."
                    )
                    urgency_level = max(urgency_level, 3)
                elif disease_result["is_viral"]:
                    actions.append(
                        f"DISEASE ACTION: {disease} detected (viral). "
                        f"Remove and destroy infected plants. Control whitefly vectors. "
                        f"No chemical cure available."
                    )
                    urgency_level = max(urgency_level, 4)
                else:
                    actions.append(
                        f"DISEASE ACTION: {disease} detected. "
                        f"Consult local agricultural extension for treatment options."
                    )
                    urgency_level = max(urgency_level, 3)
            elif confidence_level == "medium":
                warnings.append(
                    f"DISEASE WARNING: Possible {disease} detected (confidence: {confidence:.0%}). "
                    f"Recommend visual inspection to confirm."
                )
                urgency_level = max(urgency_level, 2)
            else:  # low confidence
                warnings.append(
                    f"DISEASE NOTICE: Inconclusive detection (best guess: {disease}, "
                    f"confidence: {confidence:.0%}). Recommend retaking image in better lighting."
                )
        else:
            if confidence_level in ["high", "medium"]:
                actions.append("PLANT HEALTH: No disease detected. Continue monitoring.")
    else:
        warnings.append(f"DISEASE DETECTION UNAVAILABLE: {disease_result.get('error', 'Unknown error')}")
    
    # =========================================================================
    # IRRIGATION ASSESSMENT
    # =========================================================================
    
    if irrigation_result["status"] == "success":
        decision = irrigation_result["decision"]
        original = irrigation_result["original_decision"]
        adjusted = irrigation_result["disease_adjusted"]
        
        # Map decision to action
        irrigation_actions = {
            "Irrigate_High": (
                "IRRIGATION: Immediate watering required. Soil moisture is low. "
                "Apply 25-30mm of water within the next 2 hours."
            ),
            "Irrigate_Low": (
                "IRRIGATION: Light watering recommended. "
                "Apply 10-15mm of water, preferably in early morning or evening."
            ),
            "No_Irrigation": (
                "IRRIGATION: No watering needed. Soil moisture is adequate. "
                "Next check in 4-6 hours."
            )
        }
        
        actions.append(irrigation_actions.get(decision, f"IRRIGATION: {decision}"))
        
        if adjusted:
            warnings.append(
                f"IRRIGATION ADJUSTED: Original recommendation was {original}, "
                f"modified due to disease status. {irrigation_result['adjustment_reason']}"
            )
        
        if decision == "Irrigate_High":
            urgency_level = max(urgency_level, 3)
        elif decision == "Irrigate_Low":
            urgency_level = max(urgency_level, 2)
    else:
        warnings.append(f"IRRIGATION CONTROL UNAVAILABLE: {irrigation_result.get('error', 'Unknown error')}")
    
    # =========================================================================
    # CROP SUITABILITY ASSESSMENT
    # =========================================================================
    
    if crop_result["status"] == "success":
        score = crop_result["suitability_score"]
        status = crop_result["suitability_status"]
        limiting = crop_result["limiting_factors"]
        
        if status == "Suitable":
            actions.append(
                f"ENVIRONMENT: Conditions are suitable for tomatoes (score: {score:.1f}%). "
                f"Optimal for growth and fruit development."
            )
        elif status == "Marginal":
            actions.append(
                f"ENVIRONMENT: Conditions are marginal for tomatoes (score: {score:.1f}%). "
                f"Growth possible but may be suboptimal."
            )
            if limiting:
                actions.append(f"LIMITING FACTORS: {'; '.join(limiting)}")
        else:
            warnings.append(
                f"ENVIRONMENT WARNING: Conditions are not suitable for tomatoes "
                f"(score: {score:.1f}%). Consider protective measures or alternative crops."
            )
            if limiting:
                warnings.append(f"CRITICAL FACTORS: {'; '.join(limiting)}")
            urgency_level = max(urgency_level, 2)
    else:
        warnings.append(f"CROP SUITABILITY UNAVAILABLE: {crop_result.get('error', 'Unknown error')}")
    
    # =========================================================================
    # CONFIDENCE ASSESSMENT
    # =========================================================================
    
    confidence_notes = []
    
    if disease_result["status"] == "success":
        conf_level = disease_result["confidence_level"]
        if conf_level == "high":
            confidence_notes.append("Disease detection: HIGH confidence")
        elif conf_level == "medium":
            confidence_notes.append("Disease detection: MEDIUM confidence (verify manually)")
        else:
            confidence_notes.append("Disease detection: LOW confidence (unreliable)")
    
    if irrigation_result["status"] == "success":
        conf = irrigation_result["confidence"]
        if conf >= 0.8:
            confidence_notes.append("Irrigation decision: HIGH confidence")
        elif conf >= 0.5:
            confidence_notes.append("Irrigation decision: MODERATE confidence")
        else:
            confidence_notes.append("Irrigation decision: LOW confidence (borderline conditions)")
    
    advice["confidence_assessment"] = "; ".join(confidence_notes) if confidence_notes else "Unable to assess confidence"
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    urgency_map = {1: "low", 2: "normal", 3: "high", 4: "critical"}
    advice["urgency"] = urgency_map.get(urgency_level, "normal")
    advice["actions"] = actions
    advice["warnings"] = warnings
    
    # Generate human-readable summary
    if urgency_level >= 4:
        advice["summary"] = "CRITICAL: Immediate attention required. See actions and warnings."
    elif urgency_level >= 3:
        advice["summary"] = "HIGH PRIORITY: Action needed soon. Review recommendations."
    elif urgency_level >= 2:
        advice["summary"] = "ATTENTION: Some conditions need monitoring. Review details."
    else:
        advice["summary"] = "NORMAL: All systems nominal. Continue standard operations."
    
    return advice

# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def run_orchestration() -> Dict[str, Any]:
    """
    Main orchestration function that runs all models and produces unified output.
    
    Returns a comprehensive JSON-serializable dictionary with:
    - Individual model results
    - Fused final advice
    - System metadata
    """
    start_time = datetime.now()
    
    output = {
        "timestamp": start_time.isoformat(),
        "disease_detection": {},
        "irrigation_decision": {},
        "crop_recommendation": {},
        "final_advice": {},
        "system_notes": {
            "models_loaded": [],
            "execution_time_ms": 0,
            "warnings": []
        }
    }
    
    # =========================================================================
    # RUN DISEASE DETECTION
    # =========================================================================
    
    disease_result = run_disease_detection(IMAGE_PATH)
    output["disease_detection"] = disease_result
    
    if disease_result["status"] == "success":
        output["system_notes"]["models_loaded"].append("disease_detection")
    
    # Determine disease override for irrigation
    disease_override = None
    if (disease_result["status"] == "success" and 
        disease_result["confidence_level"] in ["high", "medium"] and
        disease_result["is_fungal"]):
        disease_override = "fungal"
    
    # =========================================================================
    # RUN IRRIGATION CONTROL
    # =========================================================================
    
    irrigation_result = run_irrigation_control(
        SENSOR_DATA, 
        HISTORICAL_DATA,
        disease_override=disease_override
    )
    output["irrigation_decision"] = irrigation_result
    
    if irrigation_result["status"] == "success":
        output["system_notes"]["models_loaded"].append("irrigation_control")
    
    # =========================================================================
    # RUN CROP SUITABILITY
    # =========================================================================
    
    crop_result = run_crop_suitability(SENSOR_DATA)
    output["crop_recommendation"] = crop_result
    
    if crop_result["status"] == "success":
        output["system_notes"]["models_loaded"].append("crop_suitability")
    
    # =========================================================================
    # GENERATE FUSED ADVICE
    # =========================================================================
    
    final_advice = generate_final_advice(disease_result, irrigation_result, crop_result)
    output["final_advice"] = final_advice
    
    # =========================================================================
    # FINALIZE
    # =========================================================================
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds() * 1000
    output["system_notes"]["execution_time_ms"] = round(execution_time, 2)
    
    # Add any system-level warnings
    if len(output["system_notes"]["models_loaded"]) < 3:
        output["system_notes"]["warnings"].append(
            f"Only {len(output['system_notes']['models_loaded'])}/3 models loaded successfully"
        )
    
    return output


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Entry point for edge orchestration.
    
    Outputs a single JSON object to stdout for integration with:
    - Node-RED flows
    - Home Assistant automations  
    - Custom dashboards
    - Logging systems
    """
    
    try:
        result = run_orchestration()
        
        # Output clean JSON to stdout
        print(json.dumps(result, indent=2))
        
        sys.exit(0)
        
    except Exception as e:
        # Even errors should produce valid JSON for downstream processing
        error_output = {
            "timestamp": datetime.now().isoformat(),
            "status": "fatal_error",
            "error": str(e),
            "disease_detection": {"status": "not_run"},
            "irrigation_decision": {"status": "not_run"},
            "crop_recommendation": {"status": "not_run"},
            "final_advice": {
                "summary": "SYSTEM ERROR: Orchestration failed",
                "urgency": "critical",
                "actions": ["Check system logs", "Restart orchestration service"],
                "warnings": [str(e)]
            },
            "system_notes": {
                "models_loaded": [],
                "execution_time_ms": 0,
                "warnings": ["Fatal error during orchestration"]
            }
        }
        
        print(json.dumps(error_output, indent=2))
        sys.exit(1)
