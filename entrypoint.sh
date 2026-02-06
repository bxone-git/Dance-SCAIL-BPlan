#!/bin/bash
# Dance SCAIL B Plan - Serverless Entrypoint
# CRITICAL: NO set -e, NO exit 1 → handler MUST always start
# If entrypoint crashes before handler → worker stuck "initializing" forever

echo "=========================================="
echo "Dance SCAIL B Plan - RTX 5090 Startup"
echo "$(date)"
echo "=========================================="

# ==========================================
# Network Volume Setup
# ==========================================
NETVOLUME="${NETWORK_VOLUME_PATH:-/runpod-volume}"

echo "Checking Network Volume at $NETVOLUME..."
if [ ! -d "$NETVOLUME" ]; then
    echo "WARNING: Network Volume not mounted at $NETVOLUME"
    echo "Models will not be available - jobs will fail but handler will start"
else
    echo "Network Volume found. Creating symlinks..."

    rm -rf /ComfyUI/models/diffusion_models
    rm -rf /ComfyUI/models/text_encoders
    rm -rf /ComfyUI/models/vae
    rm -rf /ComfyUI/models/clip_vision
    rm -rf /ComfyUI/models/loras
    rm -rf /ComfyUI/models/detection
    rm -rf /root/.cache/torch/hub/checkpoints

    ln -sf $NETVOLUME/models/diffusion_models /ComfyUI/models/diffusion_models
    ln -sf $NETVOLUME/models/text_encoders /ComfyUI/models/text_encoders
    ln -sf $NETVOLUME/models/vae /ComfyUI/models/vae
    ln -sf $NETVOLUME/models/clip_vision /ComfyUI/models/clip_vision
    ln -sf $NETVOLUME/models/loras /ComfyUI/models/loras
    ln -sf $NETVOLUME/models/detection /ComfyUI/models/detection
    ln -sf $NETVOLUME/torch_cache /root/.cache/torch/hub/checkpoints

    echo "Symlinks created successfully!"
fi

# ==========================================
# Model Verification (warnings only)
# ==========================================
echo ""
echo "Verifying SCAIL models..."

check_model() {
    if [ -f "$1" ]; then
        echo "  [OK] $2"
    else
        echo "  [MISSING] $2 ($1)"
    fi
}

check_model "$NETVOLUME/models/diffusion_models/Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors" "SCAIL 14B"
check_model "$NETVOLUME/models/text_encoders/umt5-xxl-enc-bf16.safetensors" "UMT5-XXL"
check_model "$NETVOLUME/models/vae/Wan2.1_VAE.pth" "Wan2.1 VAE"
check_model "$NETVOLUME/models/clip_vision/clip_vision_h.safetensors" "CLIP Vision H"
check_model "$NETVOLUME/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" "LightX2V LoRA"
check_model "$NETVOLUME/models/detection/vitpose-l-wholebody.onnx" "VitPose-L (primary)"
check_model "$NETVOLUME/models/detection/vitpose_h_wholebody_model.onnx" "VitPose-H (fallback)"
check_model "$NETVOLUME/models/detection/yolov10m.onnx" "YOLOv10m"

# ==========================================
# Environment Check
# ==========================================
echo ""
echo "Checking environment..."
python --version 2>/dev/null || echo "WARNING: python not found"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "WARNING: PyTorch check failed"
python -c "import sageattention; print(f'SageAttention: {sageattention.__version__}')" 2>/dev/null || echo "WARNING: SageAttention not available"

# ==========================================
# Start ComfyUI
# ==========================================
echo ""
echo "Starting ComfyUI with SageAttention..."
NETLOG="$NETVOLUME/comfyui_debug.log"
python /ComfyUI/main.py --listen > /tmp/comfyui.log 2>&1 &
COMFYUI_PID=$!
# Mirror logs to network volume (persists across crashes for debugging)
tail -f /tmp/comfyui.log > "$NETLOG" 2>/dev/null &
echo "ComfyUI started with PID=$COMFYUI_PID, logs at /tmp/comfyui.log + $NETLOG"

# ==========================================
# Wait for ComfyUI (up to 180s / 3 min)
# Match Wan_Animate proven pattern
# CRITICAL: No exit 1 here - handler MUST start regardless
# ==========================================
echo "Waiting for ComfyUI to be ready..."
max_wait=180
wait_count=0
COMFYUI_READY=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready! (after ${wait_count}s)"
        COMFYUI_READY=1
        break
    fi
    if ! kill -0 $COMFYUI_PID 2>/dev/null; then
        echo "WARNING: ComfyUI process died after ${wait_count}s!"
        break
    fi
    if [ $((wait_count % 30)) -eq 0 ] && [ $wait_count -gt 0 ]; then
        echo "  Still waiting for ComfyUI... (${wait_count}/${max_wait}s)"
    fi
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $COMFYUI_READY -eq 0 ]; then
    echo "WARNING: ComfyUI not ready after ${max_wait}s - handler will still start"
fi

# ==========================================
# Start Handler (MUST ALWAYS REACH HERE)
# ==========================================
echo ""
echo "Starting RunPod handler..."
exec python /handler.py
