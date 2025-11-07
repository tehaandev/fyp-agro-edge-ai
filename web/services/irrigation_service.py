"""
Irrigation recommendation service using RandomForest model
"""
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

class IrrigationService:
    """Service for irrigation decision-making"""

    def __init__(self, model_path, scaler_path=None):
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.load_model()

    def load_model(self):
        """Load RandomForest model and scaler"""
        try:
            # Load RandomForest model
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✅ Irrigation model loaded from {self.model_path}")
            else:
                print(f"⚠️ Model not found at {self.model_path}")
                self.model = None

            # Load scaler (optional)
            if self.scaler_path and os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Scaler loaded from {self.scaler_path}")
            else:
                print("⚠️ Scaler not found - using raw features")
                self.scaler = None

        except Exception as e:
            print(f"❌ Error loading irrigation model: {e}")
            self.model = None
            self.scaler = None

    def engineer_features(self, sensor_data_list):
        """
        Engineer features from raw sensor data
        Implements the same feature engineering as in training notebooks

        Args:
            sensor_data_list: List of sensor readings (dicts) - at least 13 hours for 12h rolling stats

        Returns:
            numpy array of engineered features (37 features per row)
        """
        if len(sensor_data_list) < 13:
            # Not enough data - use simplified features
            return self._simplified_features(sensor_data_list)

        # Convert to DataFrame
        df = pd.DataFrame(sensor_data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        features_list = []

        for i in range(12, len(df)):  # Need 12 previous hours for rolling stats
            current_idx = df.index[i]
            current_time = df.loc[current_idx, 'timestamp']

            # Base features
            humidity = df.loc[current_idx, 'humidity']
            soil_moisture = df.loc[current_idx, 'soil_moisture']
            temperature = df.loc[current_idx, 'temperature']
            soil_temp = df.loc[current_idx, 'soil_temp'] if 'soil_temp' in df.columns else temperature - 2
            dew_point = df.loc[current_idx, 'dew_point'] if 'dew_point' in df.columns else temperature - (100 - humidity) / 5

            # Get recent data windows
            recent_3h = df.iloc[max(0, i-3):i]
            recent_6h = df.iloc[max(0, i-6):i]
            recent_12h = df.iloc[max(0, i-12):i]

            # Rolling statistics (3h, 6h, 12h)
            moisture_3h_mean = recent_3h['soil_moisture'].mean() if len(recent_3h) > 0 else soil_moisture
            moisture_6h_mean = recent_6h['soil_moisture'].mean() if len(recent_6h) > 0 else soil_moisture
            moisture_12h_mean = recent_12h['soil_moisture'].mean() if len(recent_12h) > 0 else soil_moisture

            moisture_3h_std = recent_3h['soil_moisture'].std() if len(recent_3h) > 1 else 0
            moisture_6h_std = recent_6h['soil_moisture'].std() if len(recent_6h) > 1 else 0
            moisture_12h_std = recent_12h['soil_moisture'].std() if len(recent_12h) > 1 else 0

            humidity_3h_mean = recent_3h['humidity'].mean() if len(recent_3h) > 0 else humidity
            humidity_6h_mean = recent_6h['humidity'].mean() if len(recent_6h) > 0 else humidity
            humidity_12h_mean = recent_12h['humidity'].mean() if len(recent_12h) > 0 else humidity

            temp_3h_mean = recent_3h['temperature'].mean() if len(recent_3h) > 0 else temperature
            temp_6h_mean = recent_6h['temperature'].mean() if len(recent_6h) > 0 else temperature

            # Change rates (1h, 3h)
            moisture_1h_change = soil_moisture - df.iloc[i-1]['soil_moisture'] if i > 0 else 0
            moisture_3h_change = soil_moisture - moisture_3h_mean

            humidity_1h_change = humidity - df.iloc[i-1]['humidity'] if i > 0 else 0
            humidity_3h_change = humidity - humidity_3h_mean

            # Time features
            hour_of_day = current_time.hour
            day_of_week = current_time.weekday()
            is_daytime = 1 if 6 <= hour_of_day <= 18 else 0

            # Cyclical encoding
            hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
            hour_cos = np.cos(2 * np.pi * hour_of_day / 24)

            # Seasonal (Yala season: May-Aug, Maha: Sep-Mar in Sri Lanka)
            is_yala = 1 if 5 <= current_time.month <= 8 else 0

            # Rainfall proxy
            humidity_spike = 1 if humidity_1h_change > 10 else 0
            recent_rain = 1 if recent_3h['humidity'].diff().max() > 10 if len(recent_3h) > 1 else 0 else 0

            # Interaction features
            temp_moisture_interaction = temperature * soil_moisture
            humidity_temp_ratio = humidity / temperature if temperature > 0 else 0

            # Compile features (37 total)
            features = [
                # Base features (5)
                humidity, temperature, soil_temp, soil_moisture, dew_point,
                # Rolling means (8)
                moisture_3h_mean, moisture_6h_mean, moisture_12h_mean,
                humidity_3h_mean, humidity_6h_mean, humidity_12h_mean,
                temp_3h_mean, temp_6h_mean,
                # Rolling stds (3)
                moisture_3h_std, moisture_6h_std, moisture_12h_std,
                # Change rates (4)
                moisture_1h_change, moisture_3h_change,
                humidity_1h_change, humidity_3h_change,
                # Time features (7)
                hour_of_day, day_of_week, is_daytime, hour_sin, hour_cos, is_yala,
                # Rainfall proxy (2)
                humidity_spike, recent_rain,
                # Interactions (2)
                temp_moisture_interaction, humidity_temp_ratio,
                # Derived (6)
                soil_moisture / humidity if humidity > 0 else 0,  # Moisture-humidity ratio
                temperature - soil_temp,  # Temperature gradient
                100 - soil_moisture,  # Dryness index
                (100 - soil_moisture) * temperature / 100,  # Weighted dryness
                abs(temperature - temp_3h_mean),  # Temp variability
                abs(soil_moisture - moisture_3h_mean)  # Moisture variability
            ]

            features_list.append(features)

        return np.array(features_list)

    def _simplified_features(self, sensor_data_list):
        """
        Fallback: simplified features when insufficient historical data
        Uses only current reading + basic derived features
        """
        if len(sensor_data_list) == 0:
            return None

        latest = sensor_data_list[-1]
        humidity = latest['humidity']
        soil_moisture = latest['soil_moisture']
        temperature = latest['temperature']
        soil_temp = latest.get('soil_temp', temperature - 2)
        dew_point = latest.get('dew_point', temperature - (100 - humidity) / 5)

        hour = datetime.now().hour
        month = datetime.now().month

        # Create simplified 37 features (with defaults for unavailable data)
        features = [
            humidity, temperature, soil_temp, soil_moisture, dew_point,
            soil_moisture, soil_moisture, soil_moisture,  # Rolling means (same as current)
            humidity, humidity, humidity,
            temperature, temperature,
            0, 0, 0,  # Rolling stds (no variability data)
            0, 0, 0, 0,  # Change rates (no history)
            hour, datetime.now().weekday(), 1 if 6 <= hour <= 18 else 0,
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            1 if 5 <= month <= 8 else 0,
            0, 0,  # Rainfall proxy
            temperature * soil_moisture, humidity / temperature if temperature > 0 else 0,
            soil_moisture / humidity if humidity > 0 else 0,
            temperature - soil_temp,
            100 - soil_moisture,
            (100 - soil_moisture) * temperature / 100,
            0, 0  # Variability metrics
        ]

        return np.array([features])

    def predict(self, sensor_data_list, use_simple_fallback=False):
        """
        Make irrigation recommendation

        Args:
            sensor_data_list: List of sensor readings (6+ hours recommended)
            use_simple_fallback: Use rule-based logic if model unavailable

        Returns:
            dict with decision, duration, confidence, probabilities
        """
        start_time = time.time()

        # Fallback to rule-based if model not available
        if self.model is None:
            if use_simple_fallback:
                return self._rule_based_decision(sensor_data_list)
            else:
                return {'error': 'Model not loaded', 'decision': 'Unknown'}

        try:
            # Engineer features
            features = self.engineer_features(sensor_data_list)

            if features is None or len(features) == 0:
                return {'error': 'Insufficient data for prediction'}

            # Use the latest engineered feature row
            X = features[-1:, :]  # Shape: (1, 37)

            # Create 6-hour window (repeat latest features 6 times as fallback)
            # Ideally, you'd have 6 hourly readings
            X_window = np.tile(X, (6, 1)).flatten().reshape(1, -1)  # Shape: (1, 222)

            # Scale if scaler available
            if self.scaler is not None:
                X_window = self.scaler.transform(X_window)

            # Predict
            prediction = self.model.predict(X_window)[0]
            probabilities = self.model.predict_proba(X_window)[0]

            # Map prediction to duration
            duration_map = {
                'Irrigate_High': 30,
                'Irrigate_Low': 15,
                'No_Irrigation': 0
            }

            duration = duration_map.get(prediction, 0)
            confidence = np.max(probabilities)

            inference_time = (time.time() - start_time) * 1000  # ms

            return {
                'decision': prediction,
                'duration': duration,
                'confidence': float(confidence),
                'probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.model.classes_, probabilities)
                },
                'inference_time_ms': inference_time,
                'model_type': 'RandomForest'
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {'error': str(e), 'decision': 'Error'}

    def _rule_based_decision(self, sensor_data_list):
        """
        Simple rule-based irrigation logic (fallback)
        Based on domain knowledge from project documentation
        """
        if len(sensor_data_list) == 0:
            return {'decision': 'No_Irrigation', 'duration': 0, 'method': 'rule_based'}

        latest = sensor_data_list[-1]
        soil_moisture = latest['soil_moisture']
        temperature = latest['temperature']
        humidity = latest['humidity']
        hour = datetime.now().hour

        # Check for recent rain (humidity spike)
        recent_rain = False
        if len(sensor_data_list) >= 3:
            recent_humidities = [r['humidity'] for r in sensor_data_list[-3:]]
            if max(recent_humidities) - min(recent_humidities) > 15:
                recent_rain = True

        # Decision logic
        if recent_rain:
            decision = 'No_Irrigation'
            duration = 0
        elif soil_moisture < 30:  # Very dry
            if temperature > 30 and 9 <= hour <= 15:  # Hot midday
                decision = 'Irrigate_Low'  # Avoid leaf scorch
                duration = 15
            else:
                decision = 'Irrigate_High'
                duration = 30
        elif soil_moisture < 50:  # Moderately dry
            decision = 'Irrigate_Low'
            duration = 15
        else:  # Sufficient moisture
            decision = 'No_Irrigation'
            duration = 0

        return {
            'decision': decision,
            'duration': duration,
            'confidence': 0.8,  # Fixed confidence for rule-based
            'method': 'rule_based',
            'reasoning': f"Moisture={soil_moisture}%, Temp={temperature}°C, Rain={recent_rain}"
        }

# Singleton instance (will be initialized in app)
irrigation_service = None
