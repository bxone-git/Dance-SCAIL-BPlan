# Dance SCAIL B Plan - RTX 5090 Serverless
# Based on proven Wan_Animate_Runpod_netvolume_5090/Dockerfile
# Tag: blendx/dance-scail-bplan:5090-v2.2.0-claudecode
# GPU: RTX 5090 (SM120) | CUDA 12.8 | SageAttention 2.2+

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# PyTorch with CUDA 12.8
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# SageAttention 2.2+ (pre-compiled wheel, no GPU needed at build time)
RUN pip install sageattention>=2.2.0

# Python packages for handler
RUN pip install -U "huggingface_hub[hf_transfer]" runpod websocket-client

WORKDIR /

# ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && pip install -r requirements.txt

# Custom Nodes (only what SCAIL workflow needs)
RUN cd /ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-WanAnimatePreprocess && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-SCAIL-pose

# Custom node requirements
RUN cd /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper && pip install -r requirements.txt && \
    cd /ComfyUI/custom_nodes/ComfyUI-KJNodes && pip install -r requirements.txt && \
    cd /ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && pip install -r requirements.txt && \
    cd /ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess && pip install -r requirements.txt && \
    cd /ComfyUI/custom_nodes/ComfyUI-SCAIL-pose && pip install -r requirements.txt

# GPU acceleration
# CRITICAL: Use onnxruntime (CPU) NOT onnxruntime-gpu
# onnxruntime-gpu<=1.22 crashes on SM120 (Blackwell/RTX 5090) with cudaErrorInvalidPtx
# VitPose/YOLO run on CPU via CPUExecutionProvider (set in workflow JSON + handler.py)
RUN pip uninstall -y onnxruntime-gpu 2>/dev/null; pip install onnxruntime triton taichi

# Model directories (symlinked to network volume at runtime)
RUN mkdir -p /ComfyUI/models/diffusion_models \
    /ComfyUI/models/text_encoders \
    /ComfyUI/models/vae \
    /ComfyUI/models/clip_vision \
    /ComfyUI/models/loras \
    /ComfyUI/models/detection \
    /root/.cache/torch/hub/checkpoints

# Project files
COPY handler.py /handler.py
COPY SCAIL_api.json /SCAIL_api.json
COPY asset/default_video.mp4 /ComfyUI/input/default_video.mp4
COPY entrypoint.sh /entrypoint.sh

# ComfyUI Manager config
RUN mkdir -p /ComfyUI/user/default/ComfyUI-Manager
COPY config.ini /ComfyUI/user/default/ComfyUI-Manager/config.ini

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
