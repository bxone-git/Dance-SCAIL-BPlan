"""
Dance SCAIL B Plan - RunPod Serverless Handler
SCAIL-only dedicated handler with fixed video + dynamic image input
Based on proven Wan_Animate handler pattern
"""
import runpod
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import subprocess
import time
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())


def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace')
        logger.error(f"ComfyUI prompt rejected (HTTP {e.code}): {error_body[:2000]}")
        raise Exception(f"ComfyUI HTTP {e.code}: {error_body[:2000]}")


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def get_comfyui_log_tail(lines=100):
    """Capture ComfyUI log tail for diagnostics"""
    try:
        with open('/tmp/comfyui.log', 'r') as f:
            return f.read()[-4000:]
    except:
        return "Could not read ComfyUI log"


def get_videos(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_videos = {}

    ws.settimeout(900)
    last_node = "unknown"

    try:
        while True:
            try:
                out = ws.recv()
            except websocket.WebSocketTimeoutException:
                log_tail = get_comfyui_log_tail()
                raise Exception(f"Video generation timeout (15min). Last node: {last_node}. ComfyUI log: {log_tail}")

            if isinstance(out, str):
                message = json.loads(out)
                msg_type = message.get('type', '')
                if msg_type == 'executing':
                    data = message['data']
                    current_node = data.get('node', None)
                    if current_node:
                        last_node = current_node
                        logger.info(f"Executing node: {current_node}")
                    if current_node is None and data['prompt_id'] == prompt_id:
                        break
                elif msg_type == 'execution_error':
                    error_data = message.get('data', {})
                    log_tail = get_comfyui_log_tail()
                    raise Exception(f"ComfyUI execution error on node {error_data.get('node_id','?')}: {error_data.get('exception_message','unknown')}. Log: {log_tail}")
            else:
                continue
    except websocket.WebSocketTimeoutException:
        log_tail = get_comfyui_log_tail()
        raise Exception(f"WebSocket timeout. Last node: {last_node}. Log: {log_tail}")

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        videos_output = []
        if 'gifs' in node_output:
            for video in node_output['gifs']:
                # Use fullpath if available, fallback to filename/subfolder
                if 'fullpath' in video:
                    video_file = video['fullpath']
                else:
                    video_file = os.path.join(
                        '/ComfyUI/output',
                        video.get('subfolder', ''),
                        video['filename']
                    )
                with open(video_file, 'rb') as f:
                    video_data = base64.b64encode(f.read()).decode('utf-8')
                videos_output.append(video_data)
        output_videos[node_id] = videos_output

    return output_videos


def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)


def download_image(url, output_path):
    """Download image from URL (Supabase etc.)"""
    try:
        result = subprocess.run(
            ['wget', '-O', output_path, '--no-verbose', url],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            logger.info(f"Image downloaded: {url} -> {output_path}")
            return output_path
        else:
            raise Exception(f"Download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise Exception("Image download timed out (60s)")


def handler(job):
    job_input = job.get("input", {})
    logger.info(f"Received job input: {json.dumps({k: v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v for k, v in job_input.items()})}")

    # Diagnostic mode: read crash logs without running workflow
    if job_input.get("action") == "read_log":
        logs = {}
        for name, path in [
            ("comfyui_local", "/tmp/comfyui.log"),
            ("comfyui_netvolume", os.path.join(os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume"), "comfyui_debug.log")),
        ]:
            try:
                with open(path, 'r') as f:
                    logs[name] = f.read()[-8000:]
            except Exception as e:
                logs[name] = f"Error reading: {e}"
        # Capture system info via subprocess (torch is in system python, not handler venv)
        try:
            result = subprocess.run(
                ['python', '-c', 'import torch; import json; print(json.dumps({"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A", "torch": torch.__version__, "cuda": torch.version.cuda, "vram_total_mb": torch.cuda.get_device_properties(0).total_mem // (1024*1024) if torch.cuda.is_available() else 0}))'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logs.update(json.loads(result.stdout.strip()))
        except:
            pass
        try:
            result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, timeout=5)
            logs["disk"] = result.stdout
        except:
            pass
        return logs

    task_id = f"/tmp/task_{uuid.uuid4()}"
    # Unique filename for ComfyUI input directory
    input_filename = f"scail_input_{uuid.uuid4().hex[:8]}.jpg"
    comfyui_input_path = f"/ComfyUI/input/{input_filename}"

    try:
        # ==========================================
        # Image input (dynamic - required)
        # ComfyUI LoadImage expects filename in /ComfyUI/input/
        # ==========================================
        image_ready = False

        if "image_url" in job_input:
            download_image(job_input["image_url"], comfyui_input_path)
            image_ready = True
        elif "image_path" in job_input:
            shutil.copy2(job_input["image_path"], comfyui_input_path)
            image_ready = True
        elif "image_base64" in job_input:
            with open(comfyui_input_path, 'wb') as f:
                f.write(base64.b64decode(job_input["image_base64"]))
            image_ready = True

        if not image_ready:
            raise Exception("Image input required. Provide image_url, image_path, or image_base64")

        # ==========================================
        # Video: FIXED to default_video.mp4
        # ==========================================
        video_path = "default_video.mp4"  # pre-copied to /ComfyUI/input/

        # ==========================================
        # Load SCAIL workflow
        # ==========================================
        prompt = load_workflow('/SCAIL_api.json')

        # SageAttention 2++ enabled (pre-installed in image)
        prompt["22"]["inputs"]["attention_mode"] = job_input.get("attention_mode", "sageattn_qk_int8_pv_fp8_cuda")

        # CRITICAL: Force CPU for onnxruntime (SM120/Blackwell not supported by onnxruntime-gpu 1.22)
        prompt["364"]["inputs"]["onnx_device"] = "CPUExecutionProvider"
        # Force CPU for taichi pose rendering (SM120 compatibility)
        prompt["362"]["inputs"]["render_device"] = "cpu"

        # Fix model names to match network volume files
        prompt["38"]["inputs"]["model_name"] = "wan_2.1_vae.safetensors"
        prompt["364"]["inputs"]["vitpose_model"] = "vitpose_h_wholebody_model.onnx"

        # ==========================================
        # Node injection
        # ==========================================
        # Input image (node 106: LoadImage - expects filename in /ComfyUI/input/)
        prompt["106"]["inputs"]["image"] = input_filename

        # Fixed video (node 130: VHS_LoadVideo)
        prompt["130"]["inputs"]["video"] = video_path

        # Text prompts (node 368: WanVideoTextEncodeCached)
        prompt["368"]["inputs"]["positive_prompt"] = job_input.get(
            "prompt", "the human starts to dance"
        )
        if "negative_prompt" in job_input:
            prompt["368"]["inputs"]["negative_prompt"] = job_input["negative_prompt"]

        # Dimensions (node 203: width, node 204: height)
        prompt["203"]["inputs"]["value"] = int(job_input.get("width", 416))
        prompt["204"]["inputs"]["value"] = int(job_input.get("height", 672))

        # Context frames/overlap (node 355: WanVideoContextOptions)
        prompt["355"]["inputs"]["context_frames"] = int(job_input.get("context_frames", 81))
        prompt["355"]["inputs"]["context_overlap"] = int(job_input.get("context_overlap", 16))

        # CFG (node 238: FloatConstant)
        prompt["238"]["inputs"]["value"] = float(job_input.get("cfg", 1.0))

        # Seed (node 348: WanVideoSamplerv2)
        prompt["348"]["inputs"]["seed"] = int(job_input.get("seed", 779298828917358))

        # Steps (node 349: WanVideoSchedulerv2)
        prompt["349"]["inputs"]["steps"] = int(job_input.get("steps", 6))

        # FPS (node 130: input fps, node 139: output fps)
        fps = int(job_input.get("fps", 24))
        prompt["130"]["inputs"]["force_rate"] = 0
        prompt["139"]["inputs"]["frame_rate"] = fps

        # ==========================================
        # WebSocket connection & execution
        # Match Wan_Animate proven pattern (180s HTTP + 180s WS)
        # ==========================================
        ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
        logger.info(f"Connecting to WebSocket: {ws_url}")

        # HTTP connectivity check (max 180s - entrypoint already waited)
        http_url = f"http://{server_address}:8188/"
        max_http_attempts = 180
        for attempt in range(max_http_attempts):
            try:
                urllib.request.urlopen(http_url, timeout=5)
                logger.info(f"HTTP connection OK (attempt {attempt+1})")
                break
            except Exception:
                if attempt == max_http_attempts - 1:
                    log_tail = ""
                    try:
                        with open('/tmp/comfyui.log', 'r') as f:
                            log_tail = f.read()[-2000:]
                    except:
                        log_tail = "Could not read ComfyUI log"
                    raise Exception(f"ComfyUI not reachable after {max_http_attempts}s. Log: {log_tail}")
                time.sleep(1)

        # WebSocket connection (max 180s)
        ws = websocket.WebSocket()
        max_ws_attempts = 36
        for attempt in range(max_ws_attempts):
            try:
                ws.connect(ws_url)
                logger.info(f"WebSocket connected (attempt {attempt+1})")
                break
            except Exception:
                if attempt == max_ws_attempts - 1:
                    raise Exception("WebSocket connection timeout (180s)")
                time.sleep(5)

        # Execute workflow
        videos = get_videos(ws, prompt)
        ws.close()

        # Return video from node 139 (Wan Animating final result)
        # Node 137 = pose skeleton (영상.1), Node 139 = final result (영상.2)
        target_node = "139"
        if target_node in videos and videos[target_node]:
            return {"video": videos[target_node][0]}

        # Fallback: log available nodes for debugging
        available_nodes = [nid for nid in videos if videos[nid]]
        raise Exception(f"No video output from target node {target_node}. Available nodes with output: {available_nodes}")

    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        raise
    finally:
        if os.path.exists(task_id):
            shutil.rmtree(task_id, ignore_errors=True)
        if os.path.exists(comfyui_input_path):
            os.remove(comfyui_input_path)


runpod.serverless.start({"handler": handler})
