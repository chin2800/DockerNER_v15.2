FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Add system packages required to build blis and similar packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/

EXPOSE 5000
EXPOSE 9000

ENV MODEL_NAME TextNERModel
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

RUN chown -R 8888 /app

CMD ["python", "-m", "seldon_core.microservice", "TextNERModel", "--service-type", "MODEL", "--persistence", "0"]