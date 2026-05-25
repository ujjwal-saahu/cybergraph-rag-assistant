FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

RUN pip install --upgrade pip

# Install CPU-only PyTorch first to avoid huge CUDA packages
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu

# Install the rest of the backend dependencies
RUN pip install -r requirements-docker.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]