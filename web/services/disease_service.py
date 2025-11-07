"""
Disease detection service using TensorFlow Lite
"""
import os
import time
import numpy as np
from PIL import Image
import cv2

class DiseaseDetectionService:
    """Service for plant disease detection"""

    def __init__(self, model_path, img_size=224):
        self.model_path = model_path
        self.img_size = img_size
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.load_model()

    def load_model(self):
        """Load TFLite model"""
        try:
            import tensorflow as tf

            if os.path.exists(self.model_path):
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()

                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()

                print(f"✅ Disease detection model loaded from {self.model_path}")
                print(f"   Input shape: {self.input_details[0]['shape']}")
                print(f"   Output shape: {self.output_details[0]['shape']}")
            else:
                print(f"⚠️ Model not found at {self.model_path}")
                self.interpreter = None

        except ImportError:
            print("⚠️ TensorFlow not installed. Disease detection unavailable.")
            self.interpreter = None
        except Exception as e:
            print(f"❌ Error loading disease detection model: {e}")
            self.interpreter = None

    def preprocess_image(self, image_path):
        """
        Preprocess image for model input

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed numpy array (1, 224, 224, 3)
        """
        try:
            # Read image
            img = Image.open(image_path).convert('RGB')

            # Resize to model input size
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)

            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)

            # Normalize to [0, 1]
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            return img_array

        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None

    def predict(self, image_path):
        """
        Predict disease from leaf image

        Args:
            image_path: Path to uploaded image

        Returns:
            dict with prediction, confidence, inference_time
        """
        if self.interpreter is None:
            return {
                'error': 'Model not loaded',
                'prediction': 'Unknown',
                'confidence': 0.0
            }

        try:
            start_time = time.time()

            # Preprocess image
            img_array = self.preprocess_image(image_path)

            if img_array is None:
                return {'error': 'Image preprocessing failed'}

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()

            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            # Binary classification: output[0] is probability of "Diseased"
            diseased_prob = float(output[0])
            healthy_prob = 1.0 - diseased_prob

            # Determine prediction
            if diseased_prob > 0.5:
                prediction = 'Diseased'
                confidence = diseased_prob
            else:
                prediction = 'Healthy'
                confidence = healthy_prob

            inference_time = (time.time() - start_time) * 1000  # ms

            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'Healthy': healthy_prob,
                    'Diseased': diseased_prob
                },
                'inference_time_ms': inference_time,
                'model_type': 'MobileNetV2-TFLite'
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'error': str(e),
                'prediction': 'Error',
                'confidence': 0.0
            }

    def get_visualization(self, image_path, prediction_result):
        """
        Create visualization with prediction overlay
        (Optional enhancement)

        Args:
            image_path: Original image path
            prediction_result: Result from predict()

        Returns:
            Path to annotated image
        """
        try:
            # Read image with OpenCV
            img = cv2.imread(image_path)

            if img is None:
                return image_path

            # Add text overlay
            prediction = prediction_result.get('prediction', 'Unknown')
            confidence = prediction_result.get('confidence', 0.0)

            # Color: green for healthy, red for diseased
            color = (0, 255, 0) if prediction == 'Healthy' else (0, 0, 255)

            # Add text
            text = f"{prediction}: {confidence:.1%}"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, color, 2, cv2.LINE_AA)

            # Save annotated image
            output_path = image_path.replace('.', '_annotated.')
            cv2.imwrite(output_path, img)

            return output_path

        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
            return image_path

# Singleton instance (will be initialized in app)
disease_service = None
