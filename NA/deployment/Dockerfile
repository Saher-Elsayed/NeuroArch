FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
LABEL maintainer="Saher Elsayed <selsayed@seas.upenn.edu>"
LABEL description="NeuroArch SNN+MARL building energy optimization inference server"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] websockets

COPY . .
EXPOSE 8000 8765
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "deployment.inference_server"]
