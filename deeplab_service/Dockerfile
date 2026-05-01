FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

COPY deeplab_service.py .

EXPOSE 5002

CMD ["python", "deeplab_service.py"]