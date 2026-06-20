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
from pipeline import start_pipeline, enqueue_task

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
    
    def get_predictor():
        return predictor
        
    def get_config():
        return {
            "mock_mode": FASTAPI_ML_MOCK,
            "device": DEVICE
        }
        
    start_pipeline(metrics, get_predictor, get_config)

# Schemas
class SettingsUpdate(BaseModel):
    mock_mode: Optional[bool] = None
    device: Optional[str] = None

class GenerateEmbeddingRequest(BaseModel):
    image_url: HttpUrl
    upload_url: HttpUrl

@app.post("/api/v1/generate-embedding", status_code=status.HTTP_202_ACCEPTED)
async def generate_embedding(req: GenerateEmbeddingRequest):
    logger.info(f"Queuing embedding request for image URL: {req.image_url}")
    try:
        await enqueue_task(str(req.image_url), str(req.upload_url))
        return {"status": "success", "message": "Embedding request queued successfully."}
    except Exception as e:
        metrics["error_count"] += 1
        logger.error(f"Failed to queue embedding request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue embedding request: {str(e)}")

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
