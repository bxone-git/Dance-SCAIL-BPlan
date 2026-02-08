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
        response_data = json.loads(urllib.request.urlopen(req).read())
        logger.info(f"Queue prompt response: {json.dumps(response_data)[:2000]}")
        return response_data
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
    executed_nodes = set()

    ws.settimeout(2400)
    last_node = "unknown"

    try:
        while True:
            try:
                out = ws.recv()
            except websocket.WebSocketTimeoutException:
                log_tail = get_comfyui_log_tail()
                raise Exception(f"Video generation timeout (40min). Last node: {last_node}. ComfyUI log: {log_tail}")

            if isinstance(out, str):
                message = json.loads(out)
                msg_type = message.get('type', '')
                if msg_type == 'executing':
                    data = message['data']
                    current_node = data.get('node', None)
                    if current_node:
                        last_node = current_node
                        executed_nodes.add(current_node)
                        logger.info(f"Executing node: {current_node}")
                    if current_node is None and data['prompt_id'] == prompt_id:
                        break
                elif msg_type == 'execution_error':
                    error_data = message.get('data', {})
                    log_tail = get_comfyui_log_tail()
                    raise Exception(f"ComfyUI execution error on node {error_data.get('node_id','?')}: {error_data.get('exception_message','unknown')}. Log: {log_tail}")
                elif msg_type == 'progress':
                    progress_data = message.get('data', {})
                    logger.info(f"Progress node {progress_data.get('node', '?')}: {progress_data.get('value', 0)}/{progress_data.get('max', 0)}")
            else:
                continue
    except websocket.WebSocketTimeoutException:
        log_tail = get_comfyui_log_tail()
        raise Exception(f"WebSocket timeout. Last node: {last_node}. Log: {log_tail}")

    history = get_history(prompt_id)[prompt_id]
    logger.info(f"Executed nodes during WS monitoring: {executed_nodes}")
    logger.info(f"History output nodes: {list(history['outputs'].keys())}")
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        logger.info(f"Node {node_id} output keys: {list(node_output.keys())}")
        if 'gifs' in node_output:
            logger.info(f"Node {node_id} has {len(node_output['gifs'])} video(s)")

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

    return output_videos, executed_nodes


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

    # Diagnostic mode: check ComfyUI node types and model files
    if job_input.get("action") == "check_nodes":
        result = {}
        # Check which node types are registered in ComfyUI
        try:
            url = f"http://{server_address}:8188/object_info"
            with urllib.request.urlopen(url, timeout=30) as response:
                object_info = json.loads(response.read())
            # Check for Wan-specific node types
            wan_nodes = ["WanVideoModelLoader", "WanVideoSamplerv2", "WanVideoDecode",
                        "WanVideoVAELoader", "WanVideoSetBlockSwap", "WanVideoSetLoRAs",
                        "WanVideoBlockSwap", "WanVideoLoraSelect", "WanVideoEmptyEmbeds",
                        "WanVideoClipVisionEncode", "WanVideoAddSCAILPoseEmbeds",
                        "WanVideoAddSCAILReferenceEmbeds", "WanVideoTextEncodeCached",
                        "WanVideoSamplerExtraArgs", "WanVideoContextOptions",
                        "WanVideoSchedulerv2", "CLIPVisionLoader", "GetImageSizeAndCount",
                        "VHS_VideoCombine"]
            result["node_types"] = {}
            for nt in wan_nodes:
                result["node_types"][nt] = nt in object_info
            # Count total registered node types
            result["total_node_types"] = len(object_info)
        except Exception as e:
            result["node_types_error"] = str(e)

        # Check model files on network volume
        try:
            nv = os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume")
            model_dirs = ["checkpoints", "vae", "clip_vision", "loras", "onnx"]
            result["model_files"] = {}
            for d in model_dirs:
                dirpath = os.path.join(nv, "ComfyUI", "models", d)
                if os.path.isdir(dirpath):
                    result["model_files"][d] = os.listdir(dirpath)
                else:
                    result["model_files"][d] = f"DIR NOT FOUND: {dirpath}"
        except Exception as e:
            result["model_files_error"] = str(e)

        # Also check /ComfyUI/models/ directly
        try:
            result["comfyui_model_dirs"] = {}
            for d in os.listdir("/ComfyUI/models"):
                dirpath = os.path.join("/ComfyUI/models", d)
                if os.path.isdir(dirpath):
                    files = os.listdir(dirpath)
                    if files:
                        result["comfyui_model_dirs"][d] = files
        except Exception as e:
            result["comfyui_models_error"] = str(e)

        # Check extra_model_paths.yaml if it exists
        for yaml_path in ["/ComfyUI/extra_model_paths.yaml", "/ComfyUI/extra_model_paths.yml"]:
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    result["extra_model_paths"] = f.read()
                break

        return result

    # Diagnostic mode: validate prompt without executing
    if job_input.get("action") == "validate_prompt":
        result = {}
        try:
            prompt = load_workflow('/SCAIL_api.json')

            # Apply same overrides as production
            prompt["22"]["inputs"]["attention_mode"] = "sageattn"
            prompt["364"]["inputs"]["onnx_device"] = "CPUExecutionProvider"
            prompt["362"]["inputs"]["render_device"] = "cpu"
            prompt["364"]["inputs"]["vitpose_model"] = "vitpose_h_wholebody_model.onnx"
            prompt["38"]["inputs"]["model_name"] = "wan_2.1_vae.safetensors"
            prompt["106"]["inputs"]["image"] = "example.jpg"
            prompt["130"]["inputs"]["video"] = "default_video.mp4"
            prompt["203"]["inputs"]["value"] = 416
            prompt["204"]["inputs"]["value"] = 672
            prompt["355"]["inputs"]["context_frames"] = 81
            prompt["355"]["inputs"]["context_overlap"] = 16
            prompt["238"]["inputs"]["value"] = 1.0
            prompt["348"]["inputs"]["seed"] = 779298828917358
            prompt["349"]["inputs"]["steps"] = 6
            prompt["130"]["inputs"]["force_rate"] = 0
            prompt["139"]["inputs"]["frame_rate"] = 24
            if "audio" in prompt["139"]["inputs"]:
                del prompt["139"]["inputs"]["audio"]
            prompt["139"]["inputs"]["trim_to_audio"] = False

            # Log what we're sending for node 139
            result["node_139_inputs"] = prompt["139"]["inputs"]
            result["node_38_inputs"] = prompt["38"]["inputs"]
            result["node_22_inputs"] = prompt["22"]["inputs"]

            # Try submitting to ComfyUI and capture FULL response
            url = f"http://{server_address}:8188/prompt"
            p = {"prompt": prompt, "client_id": "diag_" + str(uuid.uuid4())[:8]}
            data = json.dumps(p).encode('utf-8')
            req = urllib.request.Request(url, data=data)
            try:
                resp = urllib.request.urlopen(req)
                resp_data = json.loads(resp.read())
                result["prompt_accepted"] = True
                result["prompt_response"] = resp_data
                # Check which nodes are in the execution plan
                if "node_errors" in resp_data:
                    result["node_errors"] = resp_data["node_errors"]
                if "error" in resp_data:
                    result["prompt_error"] = resp_data["error"]
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8', errors='replace')
                result["prompt_accepted"] = False
                result["http_error_code"] = e.code
                result["error_body"] = error_body[:4000]
                # Parse the error body for node_errors
                try:
                    error_json = json.loads(error_body)
                    if "node_errors" in error_json:
                        result["node_errors"] = error_json["node_errors"]
                    if "error" in error_json:
                        result["prompt_error"] = error_json["error"]
                except:
                    pass

            # Also check if default_video.mp4 and example files exist
            result["input_files"] = {
                "default_video": os.path.exists("/ComfyUI/input/default_video.mp4"),
                "comfyui_input_dir": os.listdir("/ComfyUI/input/") if os.path.isdir("/ComfyUI/input/") else "NOT FOUND"
            }

        except Exception as e:
            result["error"] = str(e)
            import traceback
            result["traceback"] = traceback.format_exc()

        return result

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
        prompt["22"]["inputs"]["attention_mode"] = job_input.get("attention_mode", "sageattn")

        # CRITICAL: Force CPU for onnxruntime (SM120/Blackwell not supported by onnxruntime-gpu 1.22)
        prompt["364"]["inputs"]["onnx_device"] = "CPUExecutionProvider"
        # Force CPU for taichi pose rendering (SM120 compatibility)
        prompt["362"]["inputs"]["render_device"] = "cpu"

        # Fix model names to match network volume files
        prompt["364"]["inputs"]["vitpose_model"] = "vitpose_h_wholebody_model.onnx"
        # VAE model: workflow expects "Wan2.1_VAE.pth" but volume has "wan_2.1_vae.safetensors"
        prompt["38"]["inputs"]["model_name"] = "wan_2.1_vae.safetensors"

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

        # Remove audio input to prevent failure when default_video.mp4 has no audio track
        if "audio" in prompt["139"]["inputs"]:
            del prompt["139"]["inputs"]["audio"]
        # Ensure trim_to_audio is false when audio is removed
        prompt["139"]["inputs"]["trim_to_audio"] = False

        # ==========================================
        # Debug: Log critical Wan pipeline nodes before submission
        # ==========================================
        logger.info(f"Node 139 inputs: {json.dumps(prompt['139']['inputs'])}")
        logger.info(f"Node 139 class_type: {prompt['139'].get('class_type')}")
        logger.info(f"Node 348 inputs keys: {list(prompt['348']['inputs'].keys())}")
        logger.info(f"Node 28 inputs keys: {list(prompt['28']['inputs'].keys())}")
        logger.info(f"Node 22 inputs keys: {list(prompt['22']['inputs'].keys())}")
        logger.info(f"Node 38 inputs: {json.dumps(prompt['38']['inputs'])}")

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
        videos, executed_nodes = get_videos(ws, prompt)
        ws.close()

        # Validate Wan pipeline critical nodes executed
        wan_critical_nodes = {"22", "38", "348", "28", "139"}
        missing_wan = wan_critical_nodes - executed_nodes
        if missing_wan:
            logger.warning(f"Wan pipeline nodes did NOT execute: {missing_wan}")

        # Return video from node 139 (Wan Animate final result)
        if "139" in videos and videos["139"]:
            logger.info("Returning Wan Animate video from node 139")
            return {"video": videos["139"][0]}
        elif "137" in videos and videos["137"]:
            # Wan pipeline failed but pose succeeded — this is an ERROR
            logger.error("Node 139 (Wan Animate) has no output! Only skeleton (137) available.")
            logger.error(f"Executed nodes: {executed_nodes}")
            logger.error(f"History nodes with video: {[nid for nid in videos if videos[nid]]}")
            raise Exception(
                f"Wan Animate pipeline failed - only skeleton video produced. "
                f"Missing Wan nodes: {missing_wan}. "
                f"Executed nodes: {executed_nodes}. Check VAE model and GPU memory."
            )
        else:
            raise Exception(f"No video output from any node. Nodes in history: {list(videos.keys())}")

    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        raise
    finally:
        if os.path.exists(task_id):
            shutil.rmtree(task_id, ignore_errors=True)
        if os.path.exists(comfyui_input_path):
            os.remove(comfyui_input_path)


runpod.serverless.start({"handler": handler})
