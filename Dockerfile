# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Update package lists and install dependencies
RUN apt-get -y update \
    && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    gcc \
    g++ \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN mkdir /segment-anything

COPY . /segment-anything/

# Set the working directory
WORKDIR /segment-anything

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN mkdir /segment-anything/src/sam/weights \
    && wget -P /segment-anything/src/sam/weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Set the command to run the application
CMD ["python3", "serve.py"]