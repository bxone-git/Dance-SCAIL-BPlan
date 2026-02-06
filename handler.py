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
    return json.loads(urllib.request.urlopen(req).read())


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def get_videos(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_videos = {}

    ws.settimeout(900)

    try:
        while True:
            try:
                out = ws.recv()
            except websocket.WebSocketTimeoutException:
                raise Exception("Video generation timeout (10min)")

            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break
            else:
                continue
    except websocket.WebSocketTimeoutException:
        raise Exception("WebSocket receive timeout (10min)")

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

    task_id = f"/tmp/task_{uuid.uuid4()}"

    try:
        # ==========================================
        # Image input (dynamic - required)
        # ==========================================
        image_path = None

        if "image_url" in job_input:
            os.makedirs(task_id, exist_ok=True)
            image_path = download_image(
                job_input["image_url"],
                os.path.abspath(os.path.join(task_id, "input_image.jpg"))
            )
        elif "image_path" in job_input:
            image_path = job_input["image_path"]
        elif "image_base64" in job_input:
            os.makedirs(task_id, exist_ok=True)
            file_path = os.path.abspath(os.path.join(task_id, "input_image.jpg"))
            with open(file_path, 'wb') as f:
                f.write(base64.b64decode(job_input["image_base64"]))
            image_path = file_path

        if image_path is None:
            raise Exception("Image input required. Provide image_url, image_path, or image_base64")

        # ==========================================
        # Video: FIXED to default_video.mp4
        # ==========================================
        video_path = "default_video.mp4"  # pre-copied to /ComfyUI/input/

        # ==========================================
        # Load SCAIL workflow
        # ==========================================
        prompt = load_workflow('/SCAIL_api.json')

        # SageAttention enabled (pre-installed in image)
        prompt["22"]["inputs"]["attention_mode"] = "sageattn"

        # ==========================================
        # Node injection
        # ==========================================
        # Input image (node 106: LoadImage)
        prompt["106"]["inputs"]["image"] = image_path

        # Fixed video (node 130: VHS_LoadVideo)
        prompt["130"]["inputs"]["video"] = video_path

        # Text prompts (node 368: WanVideoTextEncodeCached)
        prompt["368"]["inputs"]["positive_prompt"] = job_input.get(
            "prompt", "the human starts to dance"
        )
        if "negative_prompt" in job_input:
            prompt["368"]["inputs"]["negative_prompt"] = job_input["negative_prompt"]

        # Dimensions (node 203: width, node 204: height)
        prompt["203"]["inputs"]["value"] = int(job_input.get("width", 1280))
        prompt["204"]["inputs"]["value"] = int(job_input.get("height", 736))

        # CFG (node 238: FloatConstant)
        prompt["238"]["inputs"]["value"] = float(job_input.get("cfg", 1.0))

        # Seed (node 348: WanVideoSamplerv2)
        prompt["348"]["inputs"]["seed"] = int(job_input.get("seed", 779298828917358))

        # Steps (node 349: WanVideoSchedulerv2)
        prompt["349"]["inputs"]["steps"] = int(job_input.get("steps", 6))

        # FPS (node 130: input fps, node 139: output fps)
        fps = int(job_input.get("fps", 16))
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

        # Return first video found
        for node_id in videos:
            if videos[node_id]:
                return {"video": videos[node_id][0]}

        raise Exception("No video output produced")

    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        raise
    finally:
        if os.path.exists(task_id):
            shutil.rmtree(task_id, ignore_errors=True)


runpod.serverless.start({"handler": handler})
