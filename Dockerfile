FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CPU по умолчанию (python-slim + torch из PyPI). Для GPU: образ с CUDA, torch+cu*, nvidia-container-toolkit и ML_DEVICE=0 в compose.
CMD ["python", "-m", "src.app.main"]
