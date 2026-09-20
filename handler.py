"""RunPod serverless handler — OmniAvatar: ONE reference image + narration audio + prompt → full-body avatar video with
lip-sync, head motion, shoulders and hand gestures (Wan2.1-based, Apache-2.0). This is the moving avatar for Lecture Studio;
Wav2Lip on a still is the fallback.

Input (event["input"]):
    image_b64        reference portrait/half-body png/jpg (base64)      [required]
    audio_b64        narration wav (base64)                              [required]
    prompt           "[first frame]-[behaviour]-[background]" per the OmniAvatar prompt format
    num_steps        20-50 (default 30) · guidance_scale 4-6 (4.5) · audio_scale (3.0) · max_tokens (30000)
    overlap_frame    1 | 13 (13) · tea_cache_l1_thresh (0.10) · seed (42)
Output: video_b64 (mp4, 480p, 25 fps), seconds, model
"""
import base64, glob, os, shutil, subprocess, time, uuid
import runpod
def _ffmpeg():
    """apt is unavailable and copies onto the network volume come out empty: use imageio-ffmpeg's binary in place."""
    for c in (shutil.which("ffmpeg"), "/runpod-volume/omniavatar/bin/ffmpeg"):
        if c and os.path.isfile(c) and os.path.getsize(c) > 1_000_000: return c
    import imageio_ffmpeg
    p = imageio_ffmpeg.get_ffmpeg_exe(); link = "/runpod-volume/omniavatar/bin/ffmpeg"
    try:
        os.makedirs(os.path.dirname(link), exist_ok=True); os.path.lexists(link) and os.remove(link); os.symlink(p, link)
    except Exception: pass
    return p
FFMPEG = _ffmpeg()
os.environ["PATH"] = "/runpod-volume/omniavatar/bin:" + os.environ.get("PATH", "")

SRC = "/runpod-volume/omniavatar/src"; MODEL = os.environ.get("OMNIAVATAR_MODEL", "1.3B")
CFG = "configs/inference_1.3B.yaml" if MODEL == "1.3B" else "configs/inference.yaml"
DEFAULT_PROMPT = ("A confident instructor in a dark shirt standing in front of a plain solid green background, looking at the camera"
                  "-speaking naturally with expressive hand gestures at chest height, slight head movements, steady posture"
                  "-plain solid green screen background, static camera")


def run(job):
    i = job["input"]
    if i.get("action") == "diag":
        def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout[-1500:]
        return {"ls_vol": sh("ls -la /runpod-volume/omniavatar /runpod-volume/omniavatar/bin 2>&1"), "ffmpeg_magic": sh("head -c 4 /runpod-volume/omniavatar/bin/ffmpeg | od -An -c; file /runpod-volume/omniavatar/bin/ffmpeg 2>&1"),
                "which": sh("which ffmpeg python; python -V; echo $PATH; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"), "bootstrap_head": sh("head -20 /tmp/bootstrap.sh 2>&1"),
                "log_tail": sh("tail -40 /tmp/bootstrap.local.log 2>&1"), "ffmpeg": FFMPEG + " " + str(os.path.getsize(FFMPEG) if os.path.exists(FFMPEG) else -1), "models": sh("ls /runpod-volume/omniavatar/pretrained_models 2>&1; du -sh /runpod-volume/* 2>&1; df -h /runpod-volume /tmp 2>&1")}
    if i.get("action") == "log":
        try: return {"log": open("/tmp/bootstrap.local.log").read()[-6000:]}
        except Exception as e: return {"error": str(e)}
    jid = uuid.uuid4().hex[:8]; work = f"/tmp/omni/{jid}"; os.makedirs(work, exist_ok=True)          # container disk — the network volume ran out of space
    img, wav = f"{work}/ref.png", f"{work}/vo.wav"
    open(img, "wb").write(base64.b64decode(i["image_b64"])); open(wav, "wb").write(base64.b64decode(i["audio_b64"]))
    subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", wav, "-ac", "1", "-ar", "16000", f"{work}/vo16.wav"], check=True)
    prompt = (i.get("prompt") or DEFAULT_PROMPT).replace("\n", " ")
    open(f"{work}/samples.txt", "w").write(f"{prompt}@@{img}@@{work}/vo16.wav\n")
    hp = ",".join(f"{k}={i.get(k, d)}" for k, d in [("num_steps", 30), ("guidance_scale", 4.5), ("audio_scale", 3.0), ("max_tokens", 30000),
                                                      ("overlap_frame", 13), ("tea_cache_l1_thresh", 0.10), ("seed", 42)])
    before = set(glob.glob(f"{SRC}/**/*.mp4", recursive=True)); t0 = time.time()
    cmd = ["torchrun", "--standalone", "--nproc_per_node=1", "scripts/inference.py", "--config", CFG, "--input_file", f"{work}/samples.txt", f"--hp={hp}"]
    p = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
    if p.returncode != 0:
        err = p.stderr or ""; i = err.rfind("Traceback (most recent call last)", 0, err.find("ChildFailedError") if "ChildFailedError" in err else len(err))
        block = err[i:i + 4000] if i >= 0 else err[-4000:]
        import torch; return {"error": "inference failed", "child_traceback": block, "stdout_tail": (p.stdout or "")[-1500:], "torch": torch.__version__ + " @ " + torch.__file__, "cmd": " ".join(cmd)}
    new = sorted(set(glob.glob(f"{SRC}/**/*.mp4", recursive=True)) - before, key=os.path.getmtime)
    if not new:
        return {"error": "no output video produced", "tail": (p.stdout or "")[-2000:]}
    out = new[-1]; muxed = f"{work}/out.mp4"
    # OmniAvatar writes video with audio; re-mux the ORIGINAL narration to be safe and normalise to h264/aac
    subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", out, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", muxed], check=True)
    data = open(muxed, "rb").read(); subprocess.run(["rm", "-rf", work])
    return {"video_b64": base64.b64encode(data).decode(), "seconds": round(time.time() - t0, 1), "model": MODEL, "hp": hp}


runpod.serverless.start({"handler": run})
