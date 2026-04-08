# Dockerfile — Earthquake Scoring API
# Python 3.11 slim (Railway standard — no Apple CLT dependency in prod)
#
# Build context: repo root (ml-project/)
# All paths relative to /app

FROM python:3.11-slim

# System deps for geopandas / shapely
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgeos-dev \
        libproj-dev \
        gdal-bin \
        libgdal-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY requirements.txt requirements_api.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements_api.txt \
 && pip install --no-cache-dir supabase

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY data/external/ ./data/external/

# The historical catalog (data/raw/) is too large to include in the image.
# On first startup, catalog_manager will load what's available and degrade
# gracefully (background_rate_yr will use fallback medians if hist_tree is None).
# For production, mount a volume or download at startup (see catalog_manager.py).

# Expose API port
EXPOSE 8000

# Run from /app so relative paths (models/, data/) resolve correctly
CMD python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
