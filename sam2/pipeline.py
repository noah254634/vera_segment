import asyncio
import io
import logging
import time
import cv2
import numpy as np
import requests

logger = logging.getLogger("sam2-service.pipeline")

# Queues for 3-stage async/sync execution pipeline
download_queue = asyncio.Queue()
gpu_queue = asyncio.Queue(maxsize=4)
uploader_queue = asyncio.Queue()

# Injected configurations
_metrics = None
_get_predictor = None
_get_config = None

# Shared session for HTTP connection pooling (keeps TCP/TLS connections alive)
http_session = requests.Session()

async def enqueue_task(image_url: str, upload_url: str):
    await download_queue.put({
        "image_url": image_url,
        "upload_url": upload_url
    })

async def downloader_loop():
    while True:
        task = await download_queue.get()
        try:
            image_url = task["image_url"]
            upload_url = task["upload_url"]
            
            response = await asyncio.to_thread(http_session.get, image_url, timeout=120)
            response.raise_for_status()
            image_bytes = response.content
            
            def decode_img(b):
                nparr = np.frombuffer(b, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return None, 0, 0
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img_rgb, img.shape[0], img.shape[1]
                
            img_rgb, h, w = await asyncio.to_thread(decode_img, image_bytes)
            if img_rgb is None:
                logger.error(f"Failed to decode image asset: {image_url}")
                continue
                
            await gpu_queue.put({
                "img_rgb": img_rgb,
                "upload_url": upload_url,
                "orig_h": h,
                "orig_w": w,
                "image_url": image_url
            })
        except Exception as e:
            if _metrics:
                _metrics["error_count"] += 1
            logger.error(f"Downloader task error for {task.get('image_url')}: {e}")
        finally:
            download_queue.task_done()

async def gpu_loop():
    while True:
        task = await gpu_queue.get()
        try:
            img_rgb = task["img_rgb"]
            upload_url = task["upload_url"]
            orig_h = task["orig_h"]
            orig_w = task["orig_w"]
            image_url = task["image_url"]
            
            predictor = _get_predictor() if _get_predictor else None
            config = _get_config() if _get_config else {}
            mock_mode = config.get("mock_mode", False)
            device = config.get("device", "cpu")
            
            start_time = time.time()
            if predictor is None or mock_mode:
                def gen_mock():
                    emb_buffer = io.BytesIO()
                    np.savez_compressed(
                        emb_buffer,
                        image_embed=np.zeros((1, 256, 64, 64), dtype=np.float32),
                        high_res_feat_0=np.zeros((1, 32, 256, 256), dtype=np.float32),
                        high_res_feat_1=np.zeros((1, 64, 128, 128), dtype=np.float32),
                        orig_h=float(orig_h),
                        orig_w=float(orig_w)
                    )
                    return emb_buffer.getvalue()
                embedding_bytes = await asyncio.to_thread(gen_mock)
            else:
                import torch
                
                def run_inference():
                    autocast_device = "cuda" if "cuda" in device else "cpu"
                    autocast_dtype = torch.bfloat16 if autocast_device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
                    
                    with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
                        predictor.set_image(img_rgb)
                        features = predictor._features
                        
                        emb_buffer = io.BytesIO()
                        np.savez_compressed(
                            emb_buffer,
                            image_embed=features["image_embed"].float().cpu().numpy(),
                            high_res_feat_0=features["high_res_feats"][0].float().cpu().numpy(),
                            high_res_feat_1=features["high_res_feats"][1].float().cpu().numpy(),
                            orig_h=float(orig_h),
                            orig_w=float(orig_w)
                        )
                        return emb_buffer.getvalue()
                        
                embedding_bytes = await asyncio.to_thread(run_inference)
                
            duration = time.time() - start_time
            if _metrics:
                _metrics["total_embedding_time_sec"] += duration
                
            await uploader_queue.put({
                "embedding_bytes": embedding_bytes,
                "upload_url": upload_url,
                "image_url": image_url
            })
        except Exception as e:
            if _metrics:
                _metrics["error_count"] += 1
            logger.error(f"GPU inference pipeline task error: {e}")
        finally:
            gpu_queue.task_done()

async def uploader_loop():
    while True:
        task = await uploader_queue.get()
        try:
            embedding_bytes = task["embedding_bytes"]
            upload_url = task["upload_url"]
            image_url = task["image_url"]
            
            def upload():
                resp = http_session.put(
                    upload_url,
                    data=embedding_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=120
                )
                resp.raise_for_status()
            await asyncio.to_thread(upload)
            
            if _metrics:
                _metrics["total_embeddings_generated"] += 1
            logger.info(f"Background pipeline successfully processed and uploaded embedding for {image_url}")
        except Exception as e:
            if _metrics:
                _metrics["error_count"] += 1
            logger.error(f"Uploader task error for {task.get('image_url')}: {e}")
        finally:
            uploader_queue.task_done()

def start_pipeline(app_metrics: dict, get_predictor_fn, get_config_fn):
    global _metrics, _get_predictor, _get_config
    _metrics = app_metrics
    _get_predictor = get_predictor_fn
    _get_config = get_config_fn
    
    # Spawn pipeline loops
    asyncio.create_task(gpu_loop())
    for _ in range(3):
        asyncio.create_task(downloader_loop())
        asyncio.create_task(uploader_loop())
    logger.info("Modular background task processing pipeline started.")
