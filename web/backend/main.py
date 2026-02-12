"""
FastAPI Backend for Agro Edge AI Web Interface
================================================
Wraps the edge_orchestrator.py logic into a REST API
that the React frontend can consume.
"""

import json
import os
import sys
import tempfile
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path so we can import the orchestrator
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import edge_orchestrator as orch

app = FastAPI(title="Agro Edge AI", version="1.0.0")

# CORS - allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost",
        "http://localhost:80",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_mock_history(
    soil_moisture: float,
    humidity: float,
    atmospheric_temp: float,
    hours: int = 12,
) -> Dict[str, list]:
    """
    Generate plausible 12-hour historical data from current sensor readings.

    Creates a gradual trend toward the current value with slight noise,
    simulating realistic sensor readings over time.
    """
    def _trend(current: float, spread: float = 5.0) -> list:
        start = current + random.uniform(-spread, spread)
        values = []
        for i in range(hours):
            t = i / (hours - 1)
            val = start + (current - start) * t
            val += random.gauss(0, spread * 0.1)
            values.append(round(val, 1))
        return values

    return {
        "soil_moisture": _trend(soil_moisture, 3.0),
        "humidity": _trend(humidity, 4.0),
        "atmospheric_temp": _trend(atmospheric_temp, 2.0),
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    humidity: float = Form(...),
    atmospheric_temp: float = Form(...),
    soil_temp: float = Form(...),
    soil_moisture: float = Form(...),
    dew_point: float = Form(...),
    hour_of_day: int = Form(...),
    day_of_week: int = Form(...),
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    soil_ph: float = Form(...),
    rainfall: float = Form(...),
):
    """
    Run the full orchestration pipeline.

    Accepts a leaf image and sensor data, generates mock historical data,
    runs all 3 models, and returns fused advice.
    """
    start_time = datetime.now()

    # Save uploaded image to temp file
    suffix = Path(image.filename or "image.png").suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await image.read()
        tmp.write(contents)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    # Build sensor data dict
    sensor_data: Dict[str, Any] = {
        "humidity": humidity,
        "atmospheric_temp": atmospheric_temp,
        "soil_temp": soil_temp,
        "soil_moisture": soil_moisture,
        "dew_point": dew_point,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "soil_ph": soil_ph,
        "rainfall": rainfall,
    }

    # Generate historical data from current readings
    historical_data = generate_mock_history(
        soil_moisture=soil_moisture,
        humidity=humidity,
        atmospheric_temp=atmospheric_temp,
    )

    # --- Run orchestration pipeline ---
    output: Dict[str, Any] = {
        "timestamp": start_time.isoformat(),
        "disease_detection": {},
        "irrigation_decision": {},
        "crop_recommendation": {},
        "final_advice": {},
        "system_notes": {
            "models_loaded": [],
            "execution_time_ms": 0,
            "warnings": [],
        },
    }

    # 1. Disease detection
    disease_result = orch.run_disease_detection(tmp_path)
    output["disease_detection"] = disease_result
    if disease_result["status"] == "success":
        output["system_notes"]["models_loaded"].append("disease_detection")

    # Determine disease override for irrigation
    disease_override = None
    if (
        disease_result["status"] == "success"
        and disease_result["confidence_level"] in ["high", "medium"]
        and disease_result["is_fungal"]
    ):
        disease_override = "fungal"

    # 2. Irrigation control
    irrigation_result = orch.run_irrigation_control(
        sensor_data, historical_data, disease_override=disease_override
    )
    output["irrigation_decision"] = irrigation_result
    if irrigation_result["status"] == "success":
        output["system_notes"]["models_loaded"].append("irrigation_control")

    # 3. Crop suitability
    crop_result = orch.run_crop_suitability(sensor_data)
    output["crop_recommendation"] = crop_result
    if crop_result["status"] == "success":
        output["system_notes"]["models_loaded"].append("crop_suitability")

    # 4. Fused advice
    final_advice = orch.generate_final_advice(
        disease_result, irrigation_result, crop_result
    )
    output["final_advice"] = final_advice

    # Finalize
    end_time = datetime.now()
    output["system_notes"]["execution_time_ms"] = round(
        (end_time - start_time).total_seconds() * 1000, 2
    )

    if len(output["system_notes"]["models_loaded"]) < 3:
        output["system_notes"]["warnings"].append(
            f"Only {len(output['system_notes']['models_loaded'])}/3 models loaded successfully"
        )

    # Cleanup temp file
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    # Convert any numpy types for JSON serialization
    return json.loads(json.dumps(output, default=str))
