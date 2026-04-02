FROM python:3.10-slim

# Use a shorter working directory
WORKDIR /app

# Install OS build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel (pkg_resources comes with setuptools)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app /app/

# Expose prediction and REST ports
EXPOSE 5000
EXPOSE 9000

# Required Seldon environment variables
ENV MODEL_NAME=TextNERModel
ENV SERVICE_TYPE=MODEL
ENV PERSISTENCE=0

# Tell Python to use a short temp directory path for sockets
ENV TMPDIR=/tmp
ENV TEMP=/tmp
ENV TMP=/tmp

# Disable Seldon metrics to avoid additional multiprocessing sockets
ENV SELDON_DISABLE_METRICS=true

# Reduce multiprocessing complexity
ENV MULTIPROCESSING_START_METHOD=spawn

# Fix permissions
RUN chown -R 8888 /app

# Start model with a single worker to avoid multi-process socket issues
CMD ["python", "-m", "seldon_core.microservice", "TextNERModel", "--service-type", "MODEL", "--persistence", "0", "--workers", "1"]
