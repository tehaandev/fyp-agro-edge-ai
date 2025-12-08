import time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
import random

# --- CONFIGURATION ---
PROJECT_ROOT = "/home/tehaan/projects/fyp-agro-edge-ai"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TOMATO_MODEL_PATH = os.path.join(MODEL_DIR, "tomato_suitability.tflite")
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease__binary_model.tflite")
IRRIGATION_MODEL_PATH = os.path.join(MODEL_DIR, "irrigation_dt.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "tomato_scaler.pkl")

# Image Config (from notebook 1)
IMG_SIZE = (128, 128)

class EdgeOrchestrator:
    def __init__(self):
        print("Initializing Edge Orchestrator...")
        self.load_models()
        print("System Ready.")

    def load_models(self):
        # 1. Load Tomato Scaler
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run notebook 6 first.")
        self.scaler = joblib.load(SCALER_PATH)

        # 2. Load Irrigation Model (Sklearn)
        if not os.path.exists(IRRIGATION_MODEL_PATH):
            raise FileNotFoundError(f"Irrigation model not found at {IRRIGATION_MODEL_PATH}. Run notebook 2 first.")
        self.irrigation_model = joblib.load(IRRIGATION_MODEL_PATH)

        # 3. Load TFLite Interpreters
        # Tomato
        if not os.path.exists(TOMATO_MODEL_PATH):
             raise FileNotFoundError(f"Tomato model not found at {TOMATO_MODEL_PATH}. Run notebook 6 first.")
        self.tomato_interpreter = tf.lite.Interpreter(model_path=TOMATO_MODEL_PATH)
        self.tomato_interpreter.allocate_tensors()
        self.tomato_input_idx = self.tomato_interpreter.get_input_details()[0]['index']
        self.tomato_output_idx = self.tomato_interpreter.get_output_details()[0]['index']

        # Disease
        if not os.path.exists(DISEASE_MODEL_PATH):
             print(f"Warning: Disease model not found at {DISEASE_MODEL_PATH}. Using Mock.")
             self.disease_interpreter = None
        else:
            self.disease_interpreter = tf.lite.Interpreter(model_path=DISEASE_MODEL_PATH)
            self.disease_interpreter.allocate_tensors()
            self.disease_input_idx = self.disease_interpreter.get_input_details()[0]['index']
            self.disease_output_idx = self.disease_interpreter.get_output_details()[0]['index']

    def run_tomato_monitor(self, sensors):
        """
        Returns suitability score (0-100)
        """
        input_values = np.array([[
            sensors['N'], sensors['P'], sensors['K'], 
            sensors['temperature'], sensors['humidity'], 
            sensors['ph'], sensors['rainfall']
        ]], dtype=np.float32)
        
        # Scale
        input_scaled = self.scaler.transform(input_values).astype(np.float32)
        
        # Inference
        self.tomato_interpreter.set_tensor(self.tomato_input_idx, input_scaled)
        self.tomato_interpreter.invoke()
        output = self.tomato_interpreter.get_tensor(self.tomato_output_idx)
        
        return output[0][0] * 100

    def run_disease_detection(self, image_data):
        """
        Returns (is_diseased, confidence)
        """
        if self.disease_interpreter is None:
            # Mock prediction
            return random.choice([True, False]), random.uniform(0.7, 0.99)

        # Preprocess image (Resize and Normalize)
        # Assuming image_data is already a numpy array of shape (128, 128, 3)
        input_tensor = np.expand_dims(image_data, axis=0).astype(np.float32) / 255.0
        
        self.disease_interpreter.set_tensor(self.disease_input_idx, input_tensor)
        self.disease_interpreter.invoke()
        output = self.disease_interpreter.get_tensor(self.disease_output_idx)
        
        # Assuming binary output (sigmoid)
        score = output[0][0]
        is_diseased = score > 0.5
        return is_diseased, score

    def run_irrigation_control(self, sensors, disease_risk):
        """
        Returns 'ON' or 'OFF'
        """
        # Base Decision
        input_df = pd.DataFrame([[sensors['soil_moisture'], sensors['temperature'], sensors['humidity']]], 
                                columns=["Soil Moisture", "Temperature", "Air Humidity"])
        base_decision = self.irrigation_model.predict(input_df)[0]
        
        # Disease Override Logic (Same as Notebook 2)
        final_decision = base_decision
        
        if base_decision == 1: # Wants to turn ON
            if disease_risk == "High":
                # Stricter threshold for disease
                # Assuming standard threshold is ~500 (dataset dependent)
                # We override if moisture is > 300 (not critically dry)
                if sensors['soil_moisture'] > 300:
                    print("    [Override] Disease Risk High: Skipping watering.")
                    final_decision = 0
                else:
                    print("    [Critical] Disease Risk High, but Soil Critical. Watering.")
        
        return "ON" if final_decision == 1 else "OFF"

    def process_cycle(self, sensors, image_data):
        print("\n--- New Cycle ---")
        print(f"Sensors: {sensors}")
        
        # 1. Check Suitability
        suitability = self.run_tomato_monitor(sensors)
        print(f"1. Tomato Suitability: {suitability:.1f}%")
        if suitability < 50:
            print("   Alert: Environmental conditions not optimal for Tomato.")

        # 2. Check Disease
        is_diseased, conf = self.run_disease_detection(image_data)
        disease_status = "Diseased" if is_diseased else "Healthy"
        print(f"2. Plant Health: {disease_status} (Conf: {conf:.2f})")
        
        # 3. Irrigation Decision
        disease_risk = "High" if is_diseased else "Low"
        pump_status = self.run_irrigation_control(sensors, disease_risk)
        print(f"3. Irrigation Pump: {pump_status}")
        
        return {
            "suitability": suitability,
            "health": disease_status,
            "pump": pump_status
        }

# --- SIMULATION ---
if __name__ == "__main__":
    orchestrator = EdgeOrchestrator()
    
    # Simulated Data Stream
    # Case 1: Good Conditions, Healthy
    sensors_1 = {
        'N': 90, 'P': 50, 'K': 60, 'temperature': 25.0, 'humidity': 60.0, 'ph': 6.5, 'rainfall': 200.0,
        'soil_moisture': 400 # Moderately dry
    }
    img_1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) # Random noise image
    
    orchestrator.process_cycle(sensors_1, img_1)
    
    # Case 2: High Humidity (Disease Risk), Soil Wet
    sensors_2 = {
        'N': 90, 'P': 50, 'K': 60, 'temperature': 28.0, 'humidity': 85.0, 'ph': 6.5, 'rainfall': 200.0,
        'soil_moisture': 400 # Moderately dry
    }
    # Force disease detection (mocking logic inside class handles random, but let's assume we pass a 'diseased' image if we had real ones)
    # Here we rely on the random mock in the class if model missing, or real inference on noise.
    
    print("\n(Simulating High Disease Risk...)")
    # We can manually test the logic by calling the internal method with forced risk if we want, 
    # but let's just run the cycle and see what happens (it might be random if model missing).
    orchestrator.process_cycle(sensors_2, img_1)
