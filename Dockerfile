# SonarVision Dockerfile — multi-stage, GPU-capable
# =============================================================================
# Stage 1: Python dependencies
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml setup.py setup.cfg ./
RUN pip install --upgrade pip && \
    pip install --prefix=/install "torch>=2.0" "numpy>=1.24" && \
    pip install --prefix=/install -e ".[dev]"

# =============================================================================
# Stage 2: Production runtime
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS production

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.cache/torch \
    SONAR_VISION_HOME=/app

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
WORKDIR /app
COPY . .

EXPOSE 8000 9090
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

ENTRYPOINT ["python3", "sonar-vision-cli.py"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Stage 3: Development with Jupyter
FROM production AS dev

RUN pip install jupyterlab ipywidgets
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
