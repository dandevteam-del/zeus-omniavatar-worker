#!/bin/bash
# zeus-omniavatar-worker bootstrap — runs as the container start command on RunPod serverless.
# Base image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 (torch 2.4 cu124 = OmniAvatar's pin).
# Everything heavy lives on the NETWORK VOLUME so cold starts after the first are fast:
#   /runpod-volume/omniavatar/src               OmniAvatar repo
#   /runpod-volume/omniavatar/pretrained_models  Wan2.1 base + OmniAvatar LoRA + wav2vec2
#   /runpod-volume/omniavatar/site               pip target (deps installed once)
set -euo pipefail
VOL=/runpod-volume/omniavatar; SRC=$VOL/src; PM=$VOL/pretrained_models; SITE=$VOL/site
MODEL="${OMNIAVATAR_MODEL:-1.3B}"           # 1.3B (fast, ~8-16 GB) or 14B (36 GB, quality)
mkdir -p "$VOL" "$PM" "$SITE" /runpod-volume/hf /runpod-volume/tmp
export HF_HOME=/runpod-volume/hf TMPDIR=/runpod-volume/tmp PYTHONPATH="$SITE:${PYTHONPATH:-}" PIP_NO_CACHE_DIR=1
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq ffmpeg git >/dev/null 2>&1 || true

[ -d "$SRC/.git" ] || git clone --depth 1 https://github.com/Omni-Avatar/OmniAvatar.git "$SRC"
if [ ! -f "$SITE/.deps-ok" ]; then
  echo "[bootstrap] installing deps into $SITE"
  pip install --target "$SITE" --upgrade -r "$SRC/requirements.txt" runpod "huggingface_hub[cli]" 2>&1 | tail -3
  # flash-attn: prebuilt wheel for torch2.4+cu12 py311; compiling from source takes >1 h, so a miss is non-fatal
  pip install --target "$SITE" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" 2>&1 | tail -1 || echo "[bootstrap] flash_attn wheel unavailable — continuing without"
  touch "$SITE/.deps-ok"
fi
HF="python -m huggingface_hub.commands.huggingface_cli download"
[ -f "$PM/wav2vec2-base-960h/config.json" ] || $HF facebook/wav2vec2-base-960h --local-dir "$PM/wav2vec2-base-960h"
if [ "$MODEL" = "14B" ]; then
  [ -f "$PM/Wan2.1-T2V-14B/config.json" ] || $HF Wan-AI/Wan2.1-T2V-14B --local-dir "$PM/Wan2.1-T2V-14B"
  [ -d "$PM/OmniAvatar-14B" ] || $HF OmniAvatar/OmniAvatar-14B --local-dir "$PM/OmniAvatar-14B"
else
  [ -f "$PM/Wan2.1-T2V-1.3B/config.json" ] || $HF Wan-AI/Wan2.1-T2V-1.3B --local-dir "$PM/Wan2.1-T2V-1.3B"
  [ -d "$PM/OmniAvatar-1.3B" ] || $HF OmniAvatar/OmniAvatar-1.3B --local-dir "$PM/OmniAvatar-1.3B"
fi
ln -sfn "$PM" "$SRC/pretrained_models"
cd "$SRC"
curl -sL "https://raw.githubusercontent.com/dandevteam-del/zeus-omniavatar-worker/main/handler.py" -o /tmp/handler.py || cp "$(dirname "$0")/handler.py" /tmp/handler.py
echo "[bootstrap] ready — model $MODEL"; exec python /tmp/handler.py
