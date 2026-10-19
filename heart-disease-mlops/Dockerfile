# ---- Serving image for the Heart Disease API ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/best_model.joblib

WORKDIR /app

# Install dependencies first: this layer is cached unless requirements change.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs (see .dockerignore for what is excluded).
COPY src/__init__.py ./src/__init__.py
COPY src/data/__init__.py src/data/preprocess.py ./src/data/
COPY src/api ./src/api
COPY models ./models

# Run as a non-root user (security best practice).
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
