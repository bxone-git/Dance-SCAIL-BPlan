# SCAIL 워크플로우 모델 다운로드 가이드

> **Dance SCAIL 워크플로우에 필요한 모델 목록 및 다운로드 링크**

---

## ⚠️ 중요 사항 (필수!)

### SageAttention 설치 필수

> 기본 ComfyUI에는 SageAttention이 없습니다. RTX 5090에서 실행하려면 반드시 설치해야 합니다!

```bash
pip install sageattention>=2.2.0
```

또는 `attention_mode`를 `sdpa`로 변경하세요.

### VAE 형식

> `.pth` 버전을 사용해야 합니다! (safetensors 호환 문제)

### 해상도

| 파라미터 | 값 |
|----------|-----|
| Width | 416 |
| Height | 672 |
| frame_rate | 24 |

---

## 📦 필수 모델

### 1. Diffusion Model (택 1)

| 버전 | 크기 | VRAM | 권장 |
|------|------|------|------|
| FP8 Scaled | ~14GB | 효율적 | ✅ RTX 5090 |
| BF16 | ~28GB | 높음 | 품질 최우선 |

```bash
# FP8 Scaled (권장)
wget -P /runpod-volume/models/diffusion_models/ \
  https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/SCAIL/Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors

# BF16 (고품질)
wget -P /runpod-volume/models/diffusion_models/ \
  https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/SCAIL/Wan21-14B-SCAIL-preview_comfy_bf16.safetensors
```

---

### 2. Text Encoder

```bash
wget -P /runpod-volume/models/text_encoders/ \
  https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors
```

---

### 3. VAE

> ⚠️ **중요**: `.pth` 버전을 사용해야 합니다! (safetensors 호환 문제)

```bash
wget -P /runpod-volume/models/vae/ \
  https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2.1_VAE.pth
```

---

### 4. LoRA (Distill - I2V)

```bash
wget -P /runpod-volume/models/loras/ \
  https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
```

---

## 📋 다운로드 링크 테이블

| 모델 | 폴더 | 다운로드 URL |
|------|------|--------------|
| SCAIL 14B (BF16) | `diffusion_models/` | https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/SCAIL/Wan21-14B-SCAIL-preview_comfy_bf16.safetensors |
| SCAIL 14B (FP8) | `diffusion_models/` | https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/SCAIL/Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors |
| UMT5-XXL | `text_encoders/` | https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors |
| VAE | `vae/` | https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors |
| Distill LoRA | `loras/` | https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors |
| CLIP Vision H | `clip_vision/` | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors |
| VitPose Model | `detection/` | https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_model.onnx |
| VitPose Data | `detection/` | https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_data.bin |
| YOLOv10m | `detection/` | https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx |

---

## 🚀 전체 다운로드 스크립트

```bash
#!/bin/bash

# Network Volume 모델 디렉토리 설정
MODELS_DIR="/runpod-volume/models"
mkdir -p $MODELS_DIR/{diffusion_models,text_encoders,vae,loras,clip_vision,detection}

# 1. Diffusion Model (FP8)
echo "Downloading SCAIL 14B FP8..."
wget -c -P $MODELS_DIR/diffusion_models/ \
  https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/SCAIL/Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors

# 2. Text Encoder
echo "Downloading UMT5-XXL..."
wget -c -P $MODELS_DIR/text_encoders/ \
  https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors

# 3. VAE
echo "Downloading VAE..."
wget -c -P $MODELS_DIR/vae/ \
  https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors

# 4. LoRA
echo "Downloading Distill LoRA..."
wget -c -P $MODELS_DIR/loras/ \
  https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors

echo "✅ 모든 모델 다운로드 완료!"
ls -la $MODELS_DIR/*/
```

---

## 워크플로우 노드별 모델 매핑

| 노드 ID | 노드 타입 | 모델 파라미터 | 파일명 |
|---------|----------|--------------|--------|
| 22 | WanVideoModelLoader | `model` | `Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors` |
| 38 | WanVideoVAELoader | `model_name` | `Wan2_1_VAE_bf16.safetensors` |
| 56 | WanVideoLoraSelect | `lora` | `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` |
| 368 | WanVideoTextEncodeCached | `model_name` | `umt5-xxl-enc-bf16.safetensors` |

---

## 🎭 Pose Detection (Custom Nodes & Models)

### 필수 Custom Nodes

| 노드 | GitHub URL | 설명 |
|------|-----------|------|
| ComfyUI-WanAnimatePreprocess | https://github.com/kijai/ComfyUI-WanAnimatePreprocess | Wan Animate 전처리 |
| ComfyUI-SCAIL-pose | https://github.com/kijai/ComfyUI-SCAIL-pose | SCAIL Pose 처리 |

### Pose Detection 모델

**YOLO (Object Detection):**
```bash
wget -P /runpod-volume/models/detection/ \
  https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx
```

**ViTPose Large (선택):**
```bash
wget -P /runpod-volume/models/detection/ \
  https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx
```

**ViTPose Huge (권장 - 두 파일 모두 필요!):**
```bash
wget -P /runpod-volume/models/detection/ \
  https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_model.onnx

wget -P /runpod-volume/models/detection/ \
  https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_data.bin
```

### Pose Detection 모델 링크 테이블

| 모델 | 크기 | 다운로드 URL |
|------|------|--------------|
| YOLOv10m | - | https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx |
| ViTPose-L | Large | https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx |
| ViTPose-H Model | Huge | https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_model.onnx |
| ViTPose-H Data | Huge | https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_data.bin |

