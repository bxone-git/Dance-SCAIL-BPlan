"""
Dance SCAIL BPlan - Worker Startup Script
Starts ComfyUI in background, sets up model symlinks, then starts RunPod handler.
Runs entirely in venv Python - avoids bash exec issues.
"""
import subprocess
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("startup")

logger.info("=" * 50)
logger.info("Dance SCAIL B Plan - RTX 5090 Startup (Python)")
logger.info("=" * 50)

# ==========================================
# Network Volume Setup
# ==========================================
NETVOLUME = os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume")
logger.info(f"Network Volume: {NETVOLUME}")

if os.path.isdir(NETVOLUME):
    logger.info("Network Volume found. Creating symlinks...")
    symlinks = {
        "diffusion_models": "diffusion_models",
        "text_encoders": "text_encoders",
        "vae": "vae",
        "clip_vision": "clip_vision",
        "loras": "loras",
        "detection": "detection",
    }
    for model_dir, vol_dir in symlinks.items():
        target = os.path.join(NETVOLUME, "models", vol_dir)
        link = f"/ComfyUI/models/{model_dir}"
        try:
            if os.path.exists(link) or os.path.islink(link):
                os.remove(link) if os.path.islink(link) else __import__('shutil').rmtree(link)
            os.symlink(target, link)
        except Exception as e:
            logger.warning(f"Symlink failed {link} -> {target}: {e}")

    # Torch cache
    torch_cache = "/root/.cache/torch/hub/checkpoints"
    try:
        if os.path.exists(torch_cache) or os.path.islink(torch_cache):
            os.remove(torch_cache) if os.path.islink(torch_cache) else __import__('shutil').rmtree(torch_cache)
        os.symlink(os.path.join(NETVOLUME, "torch_cache"), torch_cache)
    except Exception as e:
        logger.warning(f"Torch cache symlink failed: {e}")

    logger.info("Symlinks created!")

    # Model verification
    models = [
        ("SCAIL 14B", f"{NETVOLUME}/models/diffusion_models/Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors"),
        ("UMT5-XXL", f"{NETVOLUME}/models/text_encoders/umt5-xxl-enc-bf16.safetensors"),
        ("Wan2.1 VAE", f"{NETVOLUME}/models/vae/Wan2.1_VAE.pth"),
        ("CLIP Vision H", f"{NETVOLUME}/models/clip_vision/clip_vision_h.safetensors"),
        ("VitPose-L", f"{NETVOLUME}/models/detection/vitpose-l-wholebody.onnx"),
        ("YOLOv10m", f"{NETVOLUME}/models/detection/yolov10m.onnx"),
    ]
    for name, path in models:
        status = "OK" if os.path.isfile(path) else "MISSING"
        logger.info(f"  [{status}] {name}")
else:
    logger.warning(f"Network Volume not mounted at {NETVOLUME}")

# ==========================================
# Start ComfyUI (using system Python)
# ==========================================
logger.info("Starting ComfyUI with SageAttention...")
NETLOG = os.path.join(NETVOLUME, "comfyui_debug.log")
comfyui_log = open("/tmp/comfyui.log", "w")
comfyui_proc = subprocess.Popen(
    ["/usr/bin/python3.10", "/ComfyUI/main.py", "--listen"],
    stdout=comfyui_log, stderr=subprocess.STDOUT
)
logger.info(f"ComfyUI started PID={comfyui_proc.pid}")

# Mirror logs to network volume
try:
    subprocess.Popen(
        ["tail", "-f", "/tmp/comfyui.log"],
        stdout=open(NETLOG, "w"), stderr=subprocess.DEVNULL
    )
except Exception:
    pass

# ==========================================
# Wait for ComfyUI (up to 180s)
# ==========================================
logger.info("Waiting for ComfyUI to be ready...")
import urllib.request
ready = False
for i in range(90):
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/", timeout=5)
        logger.info(f"ComfyUI is ready! (after {i*2}s)")
        ready = True
        break
    except Exception:
        if comfyui_proc.poll() is not None:
            logger.warning(f"ComfyUI process died after {i*2}s!")
            break
        if i > 0 and i % 15 == 0:
            logger.info(f"  Still waiting... ({i*2}/180s)")
        time.sleep(2)

if not ready:
    logger.warning("ComfyUI not ready after 180s - handler will still start")

# ==========================================
# Start RunPod Handler
# ==========================================
logger.info("Starting RunPod handler...")

# Import and run handler directly in this process
# This avoids bash exec and keeps our Python process as PID 1
sys.path.insert(0, "/")
import importlib.util
spec = importlib.util.spec_from_file_location("handler", "/handler.py")
handler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler_module)
# handler.py calls runpod.serverless.start() at module level, so it starts automatically
