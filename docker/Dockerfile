# ─── Imagen base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ─── Directorio de trabajo ────────────────────────────────────────────────────
WORKDIR /app

# ─── Dependencias del sistema requeridas por TensorFlow ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─── Dependencias Python ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Artefactos y script de inferencia ───────────────────────────────────────
# Los artefactos son generados por save_artifacts.py antes del docker build
COPY artifacts/ ./artifacts/
COPY inferencia.py .

# ─── Entrypoint ──────────────────────────────────────────────────────────────
# Lee  /app/input.csv  →  escribe  /app/output.csv
CMD ["python", "inferencia.py"]
