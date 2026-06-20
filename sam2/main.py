import os
import io
import time
import logging
import urllib.request
import requests
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sam2-service")

app = FastAPI(
    title="VeraLabel SAM 2 ML Service",
    description="Stateless SAM 2 service supporting boxes, points, or combined refinement prompts.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration and environment variables
FASTAPI_ML_MOCK = os.getenv("FASTAPI_ML_MOCK", "false").lower() == "true"

def get_device():
    device_env = os.getenv("SAM_DEVICE")
    if device_env:
        return device_env
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

DEVICE = get_device()

# Model configuration path defaults (Using SAM 2.0 to match the configs in the repository)
MODEL_CFG = "sam2_hiera_t.yaml"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "sam2_hiera_tiny.pt")

# Telemetry State
metrics = {
    "total_embeddings_generated": 0,
    "total_embedding_time_sec": 0.0,
    "error_count": 0,
    "uptime_start": time.time()
}

predictor = None

def download_checkpoint():
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    if not os.path.exists(CHECKPOINT_PATH):
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
        logger.info(f"Downloading SAM 2.0 tiny checkpoint from {url}...")
        try:
            urllib.request.urlretrieve(url, CHECKPOINT_PATH)
            logger.info("Download completed successfully.")
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            raise e

def load_sam2_model():
    global predictor
    if FASTAPI_ML_MOCK:
        logger.info("FastAPI ML Service running in MOCK mode. SAM 2 model will not be loaded.")
        return

    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        download_checkpoint()
        logger.info(f"Loading SAM 2 model with config {MODEL_CFG} on {DEVICE}...")
        sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=DEVICE)
        predictor = SAM2ImagePredictor(sam2_model)
        logger.info("SAM 2 model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load SAM 2 model. Falling back to MOCK mode.", exc_info=True)
        predictor = None

@app.on_event("startup")
def startup_event():
    load_sam2_model()

# Schemas
class SettingsUpdate(BaseModel):
    mock_mode: Optional[bool] = None
    device: Optional[str] = None

class GenerateEmbeddingRequest(BaseModel):
    image_url: HttpUrl
    upload_url: HttpUrl

@app.post("/api/v1/generate-embedding", status_code=status.HTTP_200_OK)
async def generate_embedding(req: GenerateEmbeddingRequest):
    logger.info(f"Generating embedding for image URL: {req.image_url}")
    start_time = time.time()

    try:
        # Download image bytes
        response = requests.get(str(req.image_url), timeout=120)
        response.raise_for_status()
        image_bytes = response.content

        # Decode image using OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # HWC BGR
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file or format.")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle Mock or Real inference
        if predictor is None or FASTAPI_ML_MOCK:
            logger.info("Running in mock mode - generating dummy embedding array.")
            emb_buffer = io.BytesIO()
            np.savez_compressed(
                emb_buffer,
                image_embed=np.zeros((1, 256, 64, 64), dtype=np.float32),
                high_res_feat_0=np.zeros((1, 32, 256, 256), dtype=np.float32),
                high_res_feat_1=np.zeros((1, 64, 128, 128), dtype=np.float32),
                orig_h=float(img.shape[0]),
                orig_w=float(img.shape[1])
            )
            embedding_bytes = emb_buffer.getvalue()
        else:
            import torch
            
            autocast_device = "cuda" if "cuda" in DEVICE else "cpu"
            autocast_dtype = torch.bfloat16 if autocast_device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            
            with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
                predictor.set_image(img_rgb)
                features = predictor._features
                orig_h, orig_w = img.shape[:2]
                
                emb_buffer = io.BytesIO()
                np.savez_compressed(
                    emb_buffer,
              image_embed=features["image_embed"].float().cpu().numpy(),
                    high_res_feat_0=features["high_res_feats"][0].float().cpu().numpy(),
                    high_res_feat_1=features["high_res_feats"][1].float().cpu().numpy(),
                    orig_h=float(orig_h),
                    orig_w=float(orig_w)
                )
                embedding_bytes = emb_buffer.getvalue()

        # Upload embedding to Cloudflare R2
        upload_resp = requests.put(
            str(req.upload_url),
            data=embedding_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=120
        )
        upload_resp.raise_for_status()

        elapsed = time.time() - start_time
        metrics["total_embeddings_generated"] += 1
        metrics["total_embedding_time_sec"] += elapsed
        logger.info(f"Successfully generated and uploaded embedding in {elapsed:.2f} seconds.")
        return {"status": "success", "message": "Embedding generated and uploaded.", "elapsed_seconds": elapsed}

    except Exception as e:
        import traceback
        try:
            with open(r"c:\Users\Administrator\veralabel_cv_model\sam2\error.log", "a") as f:
                f.write(f"Error for {req.image_url}:\n{traceback.format_exc()}\n\n")
        except Exception:
            pass
        metrics["error_count"] += 1
        logger.error(f"Error during embedding generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

@app.get("/api/v1/telemetry")
def get_telemetry():
    uptime = time.time() - metrics["uptime_start"]
    avg_emb = metrics["total_embedding_time_sec"] / metrics["total_embeddings_generated"] if metrics["total_embeddings_generated"] > 0 else 0.0
    
    gpu_mem = 0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2) # MB
    except:
        pass
        
    return {
        "status": "Operational" if predictor is not None else "Standby (Mock)",
        "uptime_seconds": uptime,
        "total_embeddings_generated": metrics["total_embeddings_generated"],
        "avg_embedding_latency_sec": avg_emb,
        "error_count": metrics["error_count"],
        "gpu_memory_mb": round(gpu_mem, 2),
        "device": DEVICE,
        "mock_mode": FASTAPI_ML_MOCK,
        "checkpoint": MODEL_CFG
    }

@app.get("/api/v1/settings")
def get_settings():
    return {
        "mock_mode": FASTAPI_ML_MOCK,
        "device": DEVICE,
        "model_cfg": MODEL_CFG,
        "checkpoint_path": CHECKPOINT_PATH
    }

@app.post("/api/v1/settings")
def update_settings(update: SettingsUpdate):
    global FASTAPI_ML_MOCK, DEVICE
    needs_reload = False
    
    if update.mock_mode is not None and update.mock_mode != FASTAPI_ML_MOCK:
        FASTAPI_ML_MOCK = update.mock_mode
        needs_reload = True
        
    if update.device is not None and update.device != DEVICE:
        DEVICE = update.device
        needs_reload = True
        
    if needs_reload:
        logger.info(f"Settings updated: mock_mode={FASTAPI_ML_MOCK}, device={DEVICE}. Reloading model...")
        load_sam2_model()
        
    return {"status": "success", "message": "Settings updated"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mock_mode": predictor is None or FASTAPI_ML_MOCK,
        "device": DEVICE
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
