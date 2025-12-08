# Edge Orchestrator Setup

This system integrates three models to provide a complete decision support system for Tomato farming.

## Models Used

1.  **Tomato Environment Monitor** (`tomato_suitability.tflite`): Checks if N, P, K, Temp, Humidity, pH, Rainfall are suitable.
2.  **Disease Detection** (`plant_disease__binary_model.tflite`): Checks camera images for disease.
3.  **Irrigation Control** (`irrigation_dt.pkl`): Controls water pump based on soil moisture AND disease risk.

## How to Run

1.  **Train the Models** (if not already done):

    - Run `notebooks/6_crop_recommendation.ipynb` to generate `tomato_suitability.tflite` and `tomato_scaler.pkl`.
    - Run `notebooks/2_irrigation_control.ipynb` to generate `irrigation_dt.pkl`.
    - Ensure `plant_disease__binary_model.tflite` is in `models/`.

2.  **Run the Orchestrator**:
    ```bash
    python scripts/edge_orchestrator.py
    ```

## Logic Flow

1.  **Sensors** read environment data.
2.  **Tomato Monitor** calculates a suitability score. If low, it alerts the farmer.
3.  **Camera** takes an image. **Disease Model** checks for health issues.
4.  **Irrigation Controller** decides to water or not.
    - _Crucial Feature_: If the plant is **Diseased**, the system **overrides** the standard watering schedule to keep the foliage dry and inhibit fungal spread, unless the soil is critically dry.

