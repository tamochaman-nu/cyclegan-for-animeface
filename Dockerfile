FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

LABEL maintainer="antigravity"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install basic dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    git \
    wget \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    software-properties-common \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python3 to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.5.1 + CUDA 12.1 compatibility
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install additional dependent libraries
RUN pip install numpy scipy pillow matplotlib tqdm facenet-pytorch tensorboard pandas pyarrow

WORKDIR /workspace
