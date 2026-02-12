# ---- Backend: FastAPI + ML models ----
FROM python:3.11-slim AS backend

WORKDIR /app

# Install system dependencies for OpenCV and TFLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (only what's needed for serving)
COPY web/backend/requirements.txt /tmp/backend-requirements.txt
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/backend-requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements-docker.txt

# Copy the orchestrator script
COPY scripts/edge_orchestrator.py /app/scripts/edge_orchestrator.py

# Copy models
COPY models/ /app/models/

# Copy irrigation preprocessing data needed at inference time
COPY data/processed/irrigation/scaler.pkl /app/data/processed/irrigation/scaler.pkl
COPY data/processed/irrigation/feature_names.json /app/data/processed/irrigation/feature_names.json

# Copy backend code
COPY web/backend/ /app/web/backend/

EXPOSE 8000

CMD ["uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
