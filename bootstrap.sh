#!/bin/bash
# zeus-omniavatar-worker bootstrap — container start command on RunPod serverless.
# Base image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 (torch 2.4 cu124 = OmniAvatar's pin; NOT reinstalled).
# Heavy things live on the NETWORK VOLUME so only the first cold start is slow:
#   /runpod-volume/omniavatar/src, /pretrained_models, /site (pip deps), /bootstrap.log (readable by a later job)
VOL=/runpod-volume/omniavatar; SRC=$VOL/src; PM=$VOL/pretrained_models; SITE=$VOL/site; LOG=$VOL/bootstrap.log
MODEL="${OMNIAVATAR_MODEL:-1.3B}"
mkdir -p "$VOL" "$PM" "$SITE" /runpod-volume/hf /runpod-volume/tmp
exec > >(tee -a "$LOG") 2>&1
echo "=== bootstrap $(date -u +%FT%TZ) model=$MODEL host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
export HF_HOME=/runpod-volume/hf TMPDIR=/runpod-volume/tmp PYTHONPATH="$SITE:${PYTHONPATH:-}" PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
# apt is not usable in the serverless container: static ffmpeg on the volume instead (once)
export PATH="$VOL/bin:$PATH"
if [ ! -x "$VOL/bin/ffmpeg" ]; then
  echo "[bootstrap] fetching static ffmpeg"; mkdir -p "$VOL/bin" /tmp/ff && curl -sfL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ff/ff.tar.xz \
    && tar -xJf /tmp/ff/ff.tar.xz -C /tmp/ff && cp /tmp/ff/ffmpeg-*-static/ffmpeg /tmp/ff/ffmpeg-*-static/ffprobe "$VOL/bin/" && chmod +x "$VOL/bin/ffmpeg" "$VOL/bin/ffprobe" \
    || { echo "[bootstrap] static ffmpeg failed — trying imageio-ffmpeg"; pip install --target "$SITE" imageio-ffmpeg 2>&1 | tail -1; python -c "import imageio_ffmpeg,shutil; shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), '$VOL/bin/ffmpeg')" && chmod +x "$VOL/bin/ffmpeg"; }
fi
command -v git >/dev/null || echo "[bootstrap] WARNING: git missing"

if [ ! -d "$SRC/.git" ]; then git clone --depth 1 https://github.com/Omni-Avatar/OmniAvatar.git "$SRC" || { echo "[bootstrap] clone failed"; sleep 30; exit 1; }; fi
if [ ! -f "$SITE/.deps-ok" ]; then
  echo "[bootstrap] installing deps into $SITE (torch stays from the image)"
  # drop torch/torchvision/torchaudio pins so pip does not re-download 2.5 GB of what the image already has
  grep -viE '^(torch|torchvision|torchaudio|flash[-_]attn)' "$SRC/requirements.txt" > /tmp/req.txt || true
  pip install --target "$SITE" -r /tmp/req.txt runpod "huggingface_hub[cli]" 2>&1 | tail -5 || { echo "[bootstrap] pip failed"; sleep 30; exit 1; }
  pip install --target "$SITE" --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" 2>&1 | tail -1 || echo "[bootstrap] flash_attn wheel unavailable — continuing without"
  touch "$SITE/.deps-ok"
fi
HFCLI="$SITE/bin/huggingface-cli"; [ -x "$HFCLI" ] || HFCLI="python -c 'from huggingface_hub.commands.huggingface_cli import main; main()'"
dl() { [ -e "$2/$3" ] && return 0; echo "[bootstrap] downloading $1"; eval "$HFCLI" download "$1" --local-dir "$2" 2>&1 | tail -2; }
dl facebook/wav2vec2-base-960h "$PM/wav2vec2-base-960h" config.json
if [ "$MODEL" = "14B" ]; then dl Wan-AI/Wan2.1-T2V-14B "$PM/Wan2.1-T2V-14B" config.json; dl OmniAvatar/OmniAvatar-14B "$PM/OmniAvatar-14B" config.json
else dl Wan-AI/Wan2.1-T2V-1.3B "$PM/Wan2.1-T2V-1.3B" config.json; dl OmniAvatar/OmniAvatar-1.3B "$PM/OmniAvatar-1.3B" config.json; fi
ln -sfn "$PM" "$SRC/pretrained_models"
cd "$SRC"
curl -sfL "https://raw.githubusercontent.com/dandevteam-del/zeus-omniavatar-worker/main/handler.py" -o /tmp/handler.py || { echo "[bootstrap] handler fetch failed"; sleep 30; exit 1; }
python -c "import runpod, torch; print('[bootstrap] runpod', runpod.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" || { echo "[bootstrap] import check failed"; sleep 30; exit 1; }
echo "[bootstrap] ready — model $MODEL — starting handler"; exec python /tmp/handler.py
