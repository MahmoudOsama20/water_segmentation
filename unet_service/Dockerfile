FROM tensorflow/tensorflow:2.19.0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Don't use --ignore-installed — let TF's numpy stay
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

COPY unet_service.py .

EXPOSE 5001

CMD ["python", "unet_service.py"]