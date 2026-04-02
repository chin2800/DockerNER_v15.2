FROM python:3.10-slim

WORKDIR /app

# Make sure basic OS packages are available
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Install compatible versions of pip, setuptools, wheel with pkg_resources
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "setuptools<66" wheel

# Then install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app /app/

# Expose any needed ports
EXPOSE 5000
EXPOSE 9000

# Environment variables
ENV MODEL_NAME=TextNERModel
ENV SERVICE_TYPE=MODEL
ENV PERSISTENCE=0

# Fix permissions
RUN chown -R 8888 /app

# Start service
CMD ["python", "-m", "seldon_core.microservice", "TextNERModel", "--service-type", "MODEL", "--persistence", "0"]
