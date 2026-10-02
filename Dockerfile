# ==============================================================================
# LongCat-Video-Avatar-1.5 RunPod Serverless Worker
# Repositório Oficial: https://github.com/alexlemosrj/LongCat-Video.git
# ==============================================================================
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# 1. Dependências nativas do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    wget \
    curl \
    ninja-build \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/LongCat-Video

# 2. Copia os arquivos do repositório
COPY . /app/LongCat-Video

# 3. Atualizar pip e ferramentas de empacotamento
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 4. Instalar xFormers pré-compilado (elimina a necessidade de compilar flash-attention do zero)
RUN pip install --no-cache-dir xformers --index-url https://download.pytorch.org/whl/cu121

# 5. Instalar dependências do LongCat Avatar 1.5
RUN pip install --no-cache-dir \
    "diffusers>=0.31.0" \
    "transformers>=4.41.0" \
    "accelerate>=0.30.0" \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "audio-separator[onnx]>=0.17.0" \
    "onnxruntime-gpu" \
    "einops>=0.8.0" \
    "av>=12.0.0" \
    "opencv-python-headless>=4.9.0" \
    "ftfy>=6.2.0" \
    "loguru>=0.7.2" \
    "runpod>=1.7.0" \
    "requests" \
    "huggingface_hub" \
    "tqdm"

# 6. Variáveis de ambiente para cache persistente no Network Volume do RunPod
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/cache/huggingface
ENV TORCH_HOME=/runpod-volume/cache/torch

# 7. Iniciar o Serverless Worker
CMD ["python", "-u", "runpod_handler.py"]
