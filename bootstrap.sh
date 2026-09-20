#!/bin/bash
# zeus-omniavatar-worker bootstrap — container start command on RunPod serverless.
# Base image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 (torch 2.4 cu124 = OmniAvatar's pin; NOT reinstalled).
# Heavy things live on the NETWORK VOLUME so only the first cold start is slow:
#   /runpod-volume/omniavatar/src, /pretrained_models, /site (pip deps), /bootstrap.log (readable by a later job)
VOL=/runpod-volume/omniavatar; SRC=$VOL/src; PM=$VOL/pretrained_models; SITE=$VOL/site; LOG=$VOL/bootstrap.log
MODEL="${OMNIAVATAR_MODEL:-1.3B}"
mkdir -p "$VOL" "$PM" "$SITE" /runpod-volume/hf /runpod-volume/tmp
exec > >(tee -a "$LOG" /tmp/bootstrap.local.log) 2>&1
echo "=== bootstrap $(date -u +%FT%TZ) model=$MODEL host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
export HF_HOME=/runpod-volume/hf TMPDIR=/tmp PYTHONPATH="$SITE:${PYTHONPATH:-}" PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
# apt is not usable in the serverless container, and copying a binary onto the network volume produced a 0-byte file.
# ffmpeg comes from the imageio-ffmpeg wheel (already in $SITE) and is used IN PLACE; a symlink puts it on PATH.
export PATH="$VOL/bin:$PATH"; mkdir -p "$VOL/bin"
python -c "import imageio_ffmpeg" 2>/dev/null || pip install --target "$SITE" imageio-ffmpeg 2>&1 | tail -1
FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)
if [ -n "$FF" ] && [ -s "$FF" ]; then rm -f "$VOL/bin/ffmpeg" "$VOL/bin/ffprobe"; ln -s "$FF" "$VOL/bin/ffmpeg"; echo "[bootstrap] ffmpeg → $FF ($("$FF" -version 2>&1 | head -1))"
else echo "[bootstrap] WARNING: imageio-ffmpeg has no binary"; fi
command -v git >/dev/null || echo "[bootstrap] WARNING: git missing"

if [ ! -d "$SRC/.git" ]; then git clone --depth 1 https://github.com/Omni-Avatar/OmniAvatar.git "$SRC" || { echo "[bootstrap] clone failed"; sleep 30; exit 1; }; fi
if [ ! -f "$SITE/.deps-ok" ]; then
  echo "[bootstrap] installing deps into $SITE (torch stays from the image)"
  # drop torch/torchvision/torchaudio pins so pip does not re-download 2.5 GB of what the image already has
  grep -viE '^(torch|torchvision|torchaudio|flash[-_]attn)' "$SRC/requirements.txt" > /tmp/req.txt || true
  pip install --target "$SITE" -r /tmp/req.txt runpod "huggingface_hub[cli]" 2>&1 | tail -5 || { echo "[bootstrap] pip failed"; sleep 30; exit 1; }
  touch "$SITE/.deps-ok"
fi
HFCLI="$SITE/bin/huggingface-cli"; [ -x "$HFCLI" ] || HFCLI="python -c 'from huggingface_hub.commands.huggingface_cli import main; main()'"
dl() { [ -e "$2/$3" ] && return 0; echo "[bootstrap] downloading $1"; eval "$HFCLI" download "$1" --local-dir "$2" 2>&1 | tail -2; }
dl facebook/wav2vec2-base-960h "$PM/wav2vec2-base-960h" config.json
if [ "$MODEL" = "14B" ]; then dl Wan-AI/Wan2.1-T2V-14B "$PM/Wan2.1-T2V-14B" config.json; dl OmniAvatar/OmniAvatar-14B "$PM/OmniAvatar-14B" config.json
else dl Wan-AI/Wan2.1-T2V-1.3B "$PM/Wan2.1-T2V-1.3B" config.json; dl OmniAvatar/OmniAvatar-1.3B "$PM/OmniAvatar-1.3B" config.json; fi
# the prebuilt flash_attn wheel does not match this torch build and transformers auto-imports it when present → remove it (SDPA fallback)
rm -rf "$SITE"/flash_attn "$SITE"/flash_attn-*.dist-info "$SITE"/flash_attn_2_cuda* 2>/dev/null
ln -sfn "$PM" "$SRC/pretrained_models"
# the HF hub cache duplicates the --local-dir weights; the 50 GB volume filled up (0-byte files, blank log). Drop it.
rm -rf /runpod-volume/hf/hub /runpod-volume/tmp/* 2>/dev/null; echo "[bootstrap] volume: $(df -h /runpod-volume | tail -1)"
cd "$SRC"
curl -sfL "https://raw.githubusercontent.com/dandevteam-del/zeus-omniavatar-worker/${BOOT_SHA:-main}/handler.py" -o /tmp/handler.py || { echo "[bootstrap] handler fetch failed"; sleep 30; exit 1; }
python -c "import runpod, torch; print('[bootstrap] runpod', runpod.__version__, 'torch', torch.__version__, torch.__file__, 'cuda', torch.cuda.is_available())" || { echo "[bootstrap] import check failed"; sleep 30; exit 1; }
echo "[bootstrap] ready — model $MODEL — starting handler"; exec python /tmp/handler.py
