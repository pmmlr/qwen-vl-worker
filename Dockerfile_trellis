FROM runpod/pytorch:2.4.0-py3.10-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang git libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
ENV SPCONV_ALGO=native
ENV ATTN_BACKEND=flash-attn

WORKDIR /app

# Clone TRELLIS
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /tmp/trellis

# Install all inference deps using setup.sh
RUN cd /tmp/trellis && \
    bash -c '. ./setup.sh --basic --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast' && \
    pip install . && \
    rm -rf /tmp/trellis

# Install runpod
RUN pip install --no-cache-dir runpod Pillow

COPY handler.py .

CMD ["python3", "handler.py"]
