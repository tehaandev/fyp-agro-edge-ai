"""
Sensor data handler with mock and real sensor support
"""
import random
import time
from datetime import datetime, timedelta
import numpy as np

class SensorHandler:
    """Handle sensor readings (mock or real)"""

    def __init__(self, mock_mode=True):
        self.mock_mode = mock_mode
        self.last_reading = None

        if not mock_mode:
            try:
                # Try to import real sensor libraries
                import Adafruit_DHT
                import RPi.GPIO as GPIO
                self.dht_sensor = Adafruit_DHT.DHT11
                self.dht_pin = 4  # GPIO pin for DHT11
                self.soil_pin = 17  # GPIO pin for soil moisture sensor (analog via ADC)
                print("✅ Real sensor mode enabled")
            except ImportError:
                print("⚠️ Sensor libraries not found. Falling back to mock mode.")
                self.mock_mode = True

    def read_sensors(self):
        """
        Read sensor data (mock or real)
        Returns: dict with temperature, humidity, soil_moisture
        """
        if self.mock_mode:
            return self._read_mock_sensors()
        else:
            return self._read_real_sensors()

    def _read_mock_sensors(self):
        """
        Generate realistic mock sensor data with temporal patterns
        Simulates Sri Lankan tropical climate patterns
        """
        current_time = datetime.now()
        hour = current_time.hour

        # Base values with diurnal patterns
        # Temperature: cooler at night (23-26°C), hotter midday (28-35°C)
        temp_base = 24 + 8 * np.sin((hour - 6) * np.pi / 12)  # Peak at 2 PM
        temperature = temp_base + random.gauss(0, 1.5)

        # Humidity: inverse relationship with temperature (60-90%)
        humidity_base = 85 - 20 * np.sin((hour - 6) * np.pi / 12)
        humidity = humidity_base + random.gauss(0, 5)
        humidity = max(50, min(95, humidity))  # Clamp to realistic range

        # Soil moisture: gradually decreases during day, slight recovery at night
        if self.last_reading:
            # Temporal continuity: small changes from last reading
            last_moisture = self.last_reading.get('soil_moisture', 50)
            # Dry out during day (6 AM - 6 PM), recover at night
            if 6 <= hour <= 18:
                moisture_change = random.uniform(-2, 0.5)  # Gradual drying
            else:
                moisture_change = random.uniform(-0.5, 1)  # Slight recovery

            soil_moisture = last_moisture + moisture_change
        else:
            soil_moisture = random.uniform(35, 75)  # Initial value

        # Clamp soil moisture to realistic range
        soil_moisture = max(10, min(95, soil_moisture))

        # Additional fields for model compatibility
        reading = {
            'temperature': round(temperature, 2),
            'humidity': round(humidity, 2),
            'soil_moisture': round(soil_moisture, 2),
            'atmospheric_temp': round(temperature + random.gauss(0, 0.5), 2),  # Similar to temp
            'soil_temp': round(temperature - random.uniform(2, 5), 2),  # Soil is cooler
            'dew_point': round(temperature - (100 - humidity) / 5, 2),  # Approximate calculation
            'timestamp': current_time
        }

        self.last_reading = reading
        return reading

    def _read_real_sensors(self):
        """
        Read from actual DHT11 and soil moisture sensors
        Requires: Adafruit_DHT, RPi.GPIO libraries
        """
        try:
            import Adafruit_DHT

            # Read DHT11 sensor
            humidity, temperature = Adafruit_DHT.read_retry(self.dht_sensor, self.dht_pin)

            if humidity is None or temperature is None:
                raise ValueError("Failed to read DHT11 sensor")

            # Read soil moisture sensor (assumes ADC conversion handled elsewhere)
            # For actual implementation, use MCP3008 ADC or similar
            # This is a placeholder - adjust based on your hardware setup
            soil_moisture = self._read_soil_moisture_adc()

            reading = {
                'temperature': round(temperature, 2),
                'humidity': round(humidity, 2),
                'soil_moisture': round(soil_moisture, 2),
                'atmospheric_temp': round(temperature, 2),
                'soil_temp': round(temperature - 2, 2),  # Estimate
                'dew_point': round(temperature - (100 - humidity) / 5, 2),
                'timestamp': datetime.now()
            }

            return reading

        except Exception as e:
            print(f"❌ Error reading real sensors: {e}")
            print("⚠️ Falling back to mock data")
            return self._read_mock_sensors()

    def _read_soil_moisture_adc(self):
        """
        Read soil moisture from analog sensor via ADC
        This is a placeholder - implement based on your ADC setup
        """
        # Example for MCP3008 ADC:
        # import spidev
        # spi = spidev.SpiDev()
        # spi.open(0, 0)
        # raw_value = self._read_adc_channel(0)  # Channel 0
        # moisture_percent = (raw_value / 1023.0) * 100
        # return 100 - moisture_percent  # Invert if sensor reads low when wet

        # Placeholder: return mock value
        return random.uniform(30, 80)

    def get_historical_window(self, db_session, window_hours=6):
        """
        Get historical sensor data for the past N hours
        Required for irrigation model (needs 6-hour window)

        Args:
            db_session: SQLAlchemy database session
            window_hours: Number of hours to look back

        Returns:
            List of sensor readings (dicts)
        """
        from models import SensorData

        cutoff_time = datetime.now() - timedelta(hours=window_hours)

        readings = db_session.query(SensorData)\
            .filter(SensorData.timestamp >= cutoff_time)\
            .order_by(SensorData.timestamp.asc())\
            .all()

        return [r.to_dict() for r in readings]

# Singleton instance
sensor_handler = SensorHandler(mock_mode=True)
