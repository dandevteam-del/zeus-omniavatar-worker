# zeus-omniavatar-worker

RunPod serverless worker: **OmniAvatar** — one reference image + narration audio + prompt → **full-body talking avatar with
hand gestures, shoulder motion and lip-sync** (Wan2.1-based, Apache-2.0). The moving instructor for Lecture Studio.

No image build: the endpoint runs RunPod's stock `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` with
`bootstrap.sh` as the start command. First cold start clones the repo, installs deps and downloads weights onto the
**network volume** (`/runpod-volume/omniavatar`); every later start is seconds.

Env: `OMNIAVATAR_MODEL=1.3B` (default, ~8-16 GB VRAM, fast) or `14B` (36 GB, quality). 48 GB GPU pool.

**Input** `event.input`: `image_b64`, `audio_b64` (wav), `prompt` (`[first frame]-[behaviour]-[background]`), optional
`num_steps` (30), `guidance_scale` (4.5), `audio_scale` (3.0), `max_tokens` (30000), `overlap_frame` (13), `seed`.
**Output**: `video_b64` (mp4 480p 25fps), `seconds`, `model`.

Client: `zeus/video-studio/worker/modules/omniavatar_runpod.py` · Lecture Studio: `lecture_studio.py --motion omniavatar`.
