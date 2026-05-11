FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Системные зависимости:
# - ffmpeg           — для RTSP-стриминга
# - libgl1, libglib2 — нужны opencv
# - libgomp1         — нужен torch/ultralytics
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем только requirements и ставим зависимости отдельным слоем —
# если код изменился, но requirements нет, этот слой берётся из кэша
COPY requirements.txt .
RUN pip install -r requirements.txt

# Код копируем последним — чтобы не инвалидировать кэш pip при каждом изменении
COPY . .

# CPU по умолчанию. Для GPU: образ с CUDA, torch+cu*, nvidia-container-toolkit, ML_DEVICE=0 в compose.
CMD ["python", "-m", "src.app.main"]
