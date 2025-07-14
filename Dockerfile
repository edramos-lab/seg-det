# ---------- STAGE 1: Build environment ----------
    FROM nvcr.io/nvidia/tensorrt:22.11-py3 as build-env

    ENV DEBIAN_FRONTEND=noninteractive \
        PYTHONUNBUFFERED=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        CUDA_HOME=/usr/local/cuda \
        PATH=$CUDA_HOME/bin:$PATH \
        LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
        TENSORRT_VERSION=8.5.3.1 \
        CUDA_VERSION=11.4
    
    # System dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip \
        build-essential cmake git wget curl \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*
    
    RUN ln -sf /usr/bin/python3.8 /usr/bin/python
    
    WORKDIR /app
    COPY requirements.txt .
    COPY . .
    
    
    # TensorRT Python bindings are already installed in the base image
    
    # Optional: install torch2trt if needed
    # RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    #     cd torch2trt && python setup.py install && cd .. && rm -rf torch2trt
    
    # Install remaining Python deps
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ---------- STAGE 2: Runtime environment ----------
FROM nvcr.io/nvidia/tensorrt:22.11-py3 as production
    
    # Minimal system dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*
    
    # Add non-root user
    RUN useradd -m -u 1000 appuser
    USER appuser
    WORKDIR /app
    
    # Copy environment from build stage
    COPY --from=build-env --chown=appuser:appuser /usr/local/lib/python3.8 /usr/local/lib/python3.8
    COPY --from=build-env --chown=appuser:appuser /usr/lib/python3.8 /usr/lib/python3.8
    COPY --from=build-env --chown=appuser:appuser /usr/local/bin/pip /usr/local/bin/pip
    
    # Copy application code
    COPY --from=build-env --chown=appuser:appuser /app /app
    
    # Final setup
    RUN mkdir -p /app/models /app/logs /app/results /app/exports /app/data
    RUN chmod +x /app/run_pipeline.sh
    
    EXPOSE 6006
    
    HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
        CMD python -c "import torch; print('OK' if torch.cuda.is_available() else 'FAIL')" || exit 1
    
    CMD ["python", "main.py"]
    