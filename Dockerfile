# Hugging Face Space (Docker SDK) for the EG2 Navigator backend. CPU-only.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU wheel of PyTorch (much smaller than the default CUDA build), then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Bake the frozen encoder into the image so a cold start doesn't re-download it
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('google/embeddinggemma-2')"

COPY --chown=user eg2_autoregressive_decoding.py server.py ./
COPY --chown=user checkpoints/chat.pt checkpoints/chat.pt

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
