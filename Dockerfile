# Optional prebuilt image (RunPod GitHub build). The endpoint also works with NO build: stock runpod/pytorch image +
# bootstrap.sh as the start command (see README). Kept so the RunPod console path stays available.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git curl && rm -rf /var/lib/apt/lists/*
COPY bootstrap.sh handler.py /app/
RUN chmod +x /app/bootstrap.sh
CMD ["/app/bootstrap.sh"]
