#!/bin/bash
# Dance SCAIL BPlan - Endpoint Test Script
# Run this to test the production endpoint when GPUs are available
#
# Usage:
#   ./test_endpoint.sh              # Diagnostic test (read logs)
#   ./test_endpoint.sh full         # Full SCAIL workflow test with sample image
#   ./test_endpoint.sh status JOB_ID  # Check status of a running job

API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY environment variable}"
ENDPOINT="${RUNPOD_ENDPOINT_ID:-5xaui82bi1ejsq}"
BASE_URL="https://api.runpod.ai/v2/$ENDPOINT"

# Step 1: Check health
echo "=== Endpoint Health ==="
curl -s "$BASE_URL/health" -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
echo ""

MODE="${1:-diagnostic}"

if [ "$MODE" = "status" ]; then
    JOB_ID="$2"
    if [ -z "$JOB_ID" ]; then
        echo "Usage: $0 status <JOB_ID>"
        exit 1
    fi
    echo "=== Job Status: $JOB_ID ==="
    RESULT=$(curl -s "$BASE_URL/status/$JOB_ID" -H "Authorization: Bearer $API_KEY")
    echo "$RESULT" | python3 -m json.tool

    # If completed with video, save it
    STATUS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null)
    if [ "$STATUS" = "COMPLETED" ]; then
        echo ""
        echo "=== Saving video output ==="
        echo "$RESULT" | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
video_b64 = data.get('output', {}).get('video', '')
if video_b64:
    with open('test_output.mp4', 'wb') as f:
        f.write(base64.b64decode(video_b64))
    print('Video saved to test_output.mp4')
else:
    print('No video in output')
    print('Output:', json.dumps(data.get('output', {}), indent=2)[:2000])
"
    fi
    exit 0
fi

if [ "$MODE" = "diagnostic" ]; then
    echo "=== Submitting Diagnostic Job ==="
    RESULT=$(curl -s -X POST "$BASE_URL/run" \
        -H "Authorization: Bearer $API_KEY" \
        -H 'Content-Type: application/json' \
        -d '{"input": {"action": "read_log"}}')
    echo "$RESULT" | python3 -m json.tool
    JOB_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)

elif [ "$MODE" = "full" ]; then
    echo "=== Submitting Full SCAIL Workflow Job ==="
    # Uses a sample image URL - replace with your own
    RESULT=$(curl -s -X POST "$BASE_URL/run" \
        -H "Authorization: Bearer $API_KEY" \
        -H 'Content-Type: application/json' \
        -d '{
            "input": {
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/640px-Camponotus_flavomarginatus_ant.jpg",
                "prompt": "the human starts to dance",
                "width": 1280,
                "height": 736,
                "steps": 6,
                "cfg": 1.0,
                "fps": 16,
                "seed": 779298828917358
            }
        }')
    echo "$RESULT" | python3 -m json.tool
    JOB_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
fi

echo ""
echo "Job ID: $JOB_ID"
echo ""
echo "=== Polling (every 30s, max 20min) ==="
for i in $(seq 1 40); do
    HEALTH=$(curl -s "$BASE_URL/health" -H "Authorization: Bearer $API_KEY")
    STATUS=$(curl -s "$BASE_URL/status/$JOB_ID" -H "Authorization: Bearer $API_KEY")
    JOB_STATUS=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null)
    WORKERS=$(echo "$HEALTH" | python3 -c "import sys,json; w=json.load(sys.stdin).get('workers',{}); print(f'i={w.get(\"initializing\",0)} rdy={w.get(\"ready\",0)} run={w.get(\"running\",0)}')" 2>/dev/null)
    echo "[$i] $(date +%H:%M:%S) | workers: $WORKERS | job: $JOB_STATUS"

    if [ "$JOB_STATUS" = "COMPLETED" ] || [ "$JOB_STATUS" = "FAILED" ]; then
        echo ""
        echo "=== FINAL RESULT ==="
        echo "$STATUS" | python3 -m json.tool

        if [ "$JOB_STATUS" = "COMPLETED" ] && [ "$MODE" = "full" ]; then
            echo "$STATUS" | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
video_b64 = data.get('output', {}).get('video', '')
if video_b64:
    with open('test_output.mp4', 'wb') as f:
        f.write(base64.b64decode(video_b64))
    print('Video saved to test_output.mp4')
"
        fi
        break
    fi
    sleep 30
done
