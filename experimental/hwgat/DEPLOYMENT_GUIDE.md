# HWGAT Sign Language Recognition Microservice Deployment Guide

## 1. Overview & Architecture

The **HWGAT (Hierarchical Windowed Graph Attention Network)** is a state-of-the-art neural architecture for Indian Sign Language (ISL) recognition across **2,002 vocabulary classes**.

### Memory Isolation & Architecture Strategy

Render Free instances enforce a strict **512 MB cgroups memory limit**.
- Monolithic execution (hosting Flask web application, WebRTC signaling, MongoDB ODM, NLP safety moderation, MediaPipe Holistic, and PyTorch HWGAT inside a single process) consumes **~636 MB RSS**, triggering an Out-Of-Memory (OOM) kernel termination.
- **Solution**: The main InclusiveMatch AI web app runs safely on Render Free with `ENABLE_SIGN_AI=0` (consuming only **~73 MB - 188 MB RAM**), completely decoupling PyTorch and MediaPipe from the main web process.
- The HWGAT microservice runs as an independent, scalable service on a separate host with $\ge 1\text{ GB RAM}$, connected via `HWGAT_SERVICE_URL`.

---

## 2. Hardware & Resource Requirements

| Resource | Minimum Requirement | Recommended Production |
| :--- | :--- | :--- |
| **RAM** | **1.0 GB** | **2.0 GB - 4.0 GB** |
| **vCPU** | 1 Core | 2 Cores |
| **Disk** | 500 MB (Model + Libs) | 2 GB |
| **Inference Target** | CPU (quantized / single-thread) | CUDA GPU / Apple Silicon / Multi-core CPU |

> [!IMPORTANT]
> Do not attempt to run the HWGAT PyTorch model in the same process as the web app on any instance with less than 1 GB RAM.

---

## 3. Installation & Environment Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Pretrained weights: `experimental/hwgat/input/model_best_loss.pt` or `models/hwgat_best.pt`
- Vocabulary mapping: `experimental/hwgat/input/class_map_FDMSE.csv`

### Dependencies

Install the dedicated microservice dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install mediapipe opencv-python-headless scipy numpy flask flask-cors requests gunicorn
```

---

## 4. Running the Microservice

### Development / Local Testing

From the repository root:

```bash
python experimental/hwgat/hwgat_service.py
```
*The service binds to `http://0.0.0.0:5055` by default.*

### Production Deployment (Render, Railway, AWS, Docker)

Launch with Gunicorn (single master worker with 4 threads for concurrency):

```bash
cd experimental/hwgat
gunicorn --bind 0.0.0.0:5055 --workers 1 --threads 4 --timeout 120 hwgat_service:app
```

---

## 5. Connecting InclusiveMatch AI Web Application

In the InclusiveMatch AI application environment (e.g. Render Dashboard or `.env`):

```env
# Disable heavy in-process AI to protect Render 512MB RAM ceiling
ENABLE_SIGN_AI=0

# URL of the external HWGAT Microservice
HWGAT_SERVICE_URL=https://your-hwgat-service.onrender.com
```

When `HWGAT_SERVICE_URL` is configured:
1. `HWGATClient` health-checks the microservice at `GET /health`.
2. Real-time video frames and landmarks captured during WebRTC calls are automatically offloaded to the external service.
3. If the external microservice is offline or unreachable, InclusiveMatch AI degrades gracefully without crashing or dropping WebRTC calls.

---

## 6. Microservice API Specifications

### `GET /health`
Verifies service readiness and model parameter loading.

**Response (200 OK):**
```json
{
  "status": "ready",
  "service": "HWGAT Sign Recognition Microservice",
  "architecture": "HWGATE (Hierarchical Windowed Graph Attention Network)",
  "classes_count": 2002,
  "device": "cpu",
  "active_sessions": 2
}
```

### `POST /recognize_frame`
Processes a single image frame (JPEG/PNG) from an ongoing call.

- **Content-Type**: `multipart/form-data`
- **Fields**:
  - `frame`: Binary image file (JPEG or PNG)
  - `user_id`: Unique identifier for the user session (string)
  - `threshold`: Minimum confidence cutoff (float, default `0.25`)

**Response (200 OK):**
```json
{
  "text": "Hello",
  "confidence": 88.4,
  "message": "Recognized: Hello",
  "ready": true,
  "arch": "HWGAT",
  "signing_state": "SIGNING",
  "is_signing": true,
  "turn_completed": false,
  "phrase": "Hello",
  "finalized_phrase": "",
  "new_token": "Hello",
  "top_k": [
    {"rank": 1, "class_id": 142, "label": "Hello", "confidence": 88.4},
    {"rank": 2, "class_id": 89, "label": "Namaste", "confidence": 4.1}
  ]
}
```

### `POST /recognize_landmarks`
Processes a pre-extracted landmark sequence directly.

- **Content-Type**: `application/json`
- **Body**:
```json
{
  "landmarks": [[[0.5, 0.33, 0.0, 1.0], ...], ...],
  "user_id": "user_123",
  "mode": "training_aligned",
  "threshold": 0.25
}
```

**Response (200 OK):**
```json
{
  "text": "Thank You",
  "confidence": 91.2,
  "ready": true,
  "arch": "HWGAT",
  "mode": "training_aligned"
}
```

### `POST /reset_user`
Flushes session sliding buffers when a participant hangs up or disconnects.

- **Content-Type**: `application/x-www-form-urlencoded`
- **Field**: `user_id`

**Response (200 OK):**
```json
{
  "ok": true,
  "message": "User user_123 session reset successfully"
}
```

---

## 7. WebRTC Subtitle Delivery Contract

1. In `templates/video_call.html`, camera frames are rendered onto a 320x240 canvas and posted every 900ms.
2. `routes/video_routes.py` passes the frame to `HWGATClient`.
3. If recognized with $\ge 25\%$ confidence, Socket.IO emits `sign_caption` with payload:
   ```json
   {
     "user_id": "...",
     "text": "Hello",
     "confidence": 88.4,
     "phrase": "Hello friend"
   }
   ```
4. The subtitle bar displays synchronized live captions on both sides of the call.
