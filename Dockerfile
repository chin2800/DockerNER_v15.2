FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade \
    pip \
    "setuptools>=65.0.0,<81.0.0" \
    wheel

RUN pip install --no-cache-dir -r requirements.txt

# Verify the versions actually installed
RUN python -c "import tensorflow as tf; print('TF:', tf.__version__)"
RUN python -c "import google.protobuf as p; print('PROTO:', p.__version__)"
RUN python -c "import seldon_core; print('SELDON:', seldon_core.__version__)"
RUN python -c "import pkg_resources; print('pkg_resources: OK')"

COPY app /app/

EXPOSE 5000
EXPOSE 9000

ENV MODEL_NAME=TextNERModel
ENV SERVICE_TYPE=MODEL
ENV PERSISTENCE=0

ENV TMPDIR=/tmp
ENV TEMP=/tmp
ENV TMP=/tmp

ENV SELDON_DISABLE_METRICS=true
ENV MULTIPROCESSING_START_METHOD=spawn

RUN chown -R 8888 /app

CMD ["python", "-m", "seldon_core.microservice", "TextNERModel", "--service-type", "MODEL", "--persistence", "0", "--workers", "1"]
