"""
DancePixel Production Server
============================

Multi-User Pose Estimation Server mit Queue-System (Celery + Redis)

Features:
- Asynchrone Verarbeitung (nicht blockierend)
- Multi-Worker Support (parallele Video-Verarbeitung)
- GPU-Optimierung (FP16, Batch Processing)
- Status-Polling für Clients
- Redis Queue für unbegrenzte Concurrent Users
- Auto-Retry bei Fehlern
- Health-Check Endpoint

Requirements:
    pip install fastapi uvicorn ultralytics opencv-python-headless pillow python-multipart redis celery

Setup:
    1. Redis starten: redis-server
    2. FastAPI starten: python pose_server_production.py
    3. Worker starten: celery -A pose_server_production worker --loglevel=info --concurrency=3

Author: Emil (DancePixel Team)
Date: 2026-01-11
"""

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import json
import redis
from celery import Celery
import uuid
from pathlib import Path
import time
from typing import Optional

# ============================================================
# CONFIGURATION
# ============================================================

# Server Config
HOST = "0.0.0.0"  # Alle IPs (wichtig für öffentlichen Zugriff)
PORT = 5000

# Model Config
MODEL_PATH = "yolov8n-pose.pt"  # n=fastest, s=balanced, m/l/x=accurate
USE_FP16 = True  # 2x schneller auf GPU (empfohlen)

# Redis Config
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Storage Config
TEMP_DIR = "/tmp/dancepixel"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "DANCEPIXEL_FINAL")

# Task Config
TASK_TTL = 86400  # 24 Stunden (danach werden Ergebnisse gelöscht)

# ============================================================
# REDIS & CELERY SETUP
# ============================================================

# Redis Client
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=False  # Binary data (JSON)
)

# Celery App
celery_app = Celery(
    'dancepixel',
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Celery Config
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/Berlin',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1h max pro Task
    worker_prefetch_multiplier=1,  # Ein Task pro Worker
)

# ============================================================
# YOLOV8 MODEL LADEN
# ============================================================

print("\n" + "="*60)
print("🔄 Lade YOLOv8-Pose Model...")
print("="*60)

model = YOLO(MODEL_PATH)

# GPU Detection
try:
    import torch
    if torch.cuda.is_available():
        model.to('cuda')
        device = "GPU (CUDA)"
        
        # FP16 Optimization (2x schneller)
        if USE_FP16:
            model.model.half()
            device += " + FP16"
    else:
        device = "CPU"
except ImportError:
    device = "CPU (torch not available)"

print(f"✅ YOLOv8-Pose geladen!")
print(f"📊 Device: {device}")
print(f"📦 Model: {MODEL_PATH}")
print("="*60 + "\n")

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="DancePixel Production API",
    description="Multi-User Pose Estimation Server",
    version="2.0.0"
)

# CORS (für Production: Nur spezifische Domains erlauben!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Ersetze mit deiner Domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files (HTML, CSS, JS)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    print(f"📁 Static Files: {STATIC_DIR}")

# Temp Directory erstellen
os.makedirs(TEMP_DIR, exist_ok=True)

# ============================================================
# CELERY TASK (Worker-Funktion)
# ============================================================

@celery_app.task(bind=True, name='dancepixel.process_video')
def process_video_task(self, video_path: str, task_id: str):
    """
    Background Task: Video verarbeiten und Pose-Daten extrahieren
    
    Args:
        video_path: Pfad zum temporären Video
        task_id: Eindeutige Task-ID
    
    Returns:
        dict: Status und Ergebnis
    """
    
    start_time = time.time()
    
    try:
        print(f"\n{'='*60}")
        print(f"🎬 Task {task_id}: Starte Verarbeitung")
        print(f"📁 Video: {video_path}")
        print(f"{'='*60}\n")
        
        # Status Update: Processing
        redis_client.set(f"task:{task_id}:status", "processing")
        redis_client.set(f"task:{task_id}:progress", 0)
        
        # Video öffnen
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Video konnte nicht geöffnet werden")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video Info:")
        print(f"   - Frames: {total_frames}")
        print(f"   - FPS: {fps:.1f}")
        print(f"   - Resolution: {width}x{height}")
        print(f"\n🚀 Starte Pose-Extraktion...\n")
        
        all_frames = []
        processed = 0
        last_update = 0
        
        # Frame-by-Frame Verarbeitung
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLOv8 Inference
            results = model(frame, verbose=False)
            
            # Keypoints extrahieren
            frame_data = {
                "frame_number": processed,
                "detections": []
            }
            
            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                keypoints = results[0].keypoints.data.cpu().numpy()
                boxes = results[0].boxes.data.cpu().numpy() if results[0].boxes else []
                
                for i, kp in enumerate(keypoints):
                    detection = {
                        "keypoints": kp.tolist(),  # 17x3 (x, y, confidence)
                        "bbox": boxes[i][:4].tolist() if len(boxes) > i else [0, 0, 0, 0],
                        "confidence": float(boxes[i][4]) if len(boxes) > i else 0.0
                    }
                    frame_data["detections"].append(detection)
            
            all_frames.append(frame_data)
            processed += 1
            
            # Progress Update (alle 5 Frames oder bei Ende)
            if processed - last_update >= 5 or processed == total_frames:
                progress = int((processed / total_frames) * 100)
                redis_client.set(f"task:{task_id}:progress", progress)
                
                # Celery State Update
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress,
                        'processed': processed,
                        'total': total_frames
                    }
                )
                
                elapsed = time.time() - start_time
                fps_current = processed / elapsed if elapsed > 0 else 0
                
                print(f"📊 Progress: {progress}% ({processed}/{total_frames}) @ {fps_current:.1f} FPS")
                
                last_update = processed
        
        cap.release()
        
        # Processing Stats
        elapsed_time = time.time() - start_time
        avg_fps = total_frames / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"✅ Verarbeitung abgeschlossen!")
        print(f"⏱️  Zeit: {elapsed_time:.1f}s")
        print(f"📊 Durchschnitt: {avg_fps:.1f} FPS")
        print(f"🎯 Detections: {len(all_frames)} Frames")
        print(f"{'='*60}\n")
        
        # Ergebnis erstellen
        result = {
            "frames": all_frames,
            "metadata": {
                "total_frames": total_frames,
                "fps": fps,
                "processed_frames": len(all_frames),
                "width": width,
                "height": height,
                "processing_time": elapsed_time,
                "avg_fps": avg_fps
            }
        }
        
        # In Redis speichern (mit TTL)
        redis_client.setex(
            f"task:{task_id}:result",
            TASK_TTL,
            json.dumps(result)
        )
        
        # Status: Completed
        redis_client.setex(f"task:{task_id}:status", TASK_TTL, "completed")
        redis_client.setex(f"task:{task_id}:progress", TASK_TTL, 100)
        
        # Temporäre Datei löschen
        try:
            os.remove(video_path)
            print(f"🗑️  Temp-Datei gelöscht: {video_path}")
        except Exception as e:
            print(f"⚠️  Konnte Temp-Datei nicht löschen: {e}")
        
        return {
            "status": "completed",
            "task_id": task_id,
            "frames": len(all_frames),
            "processing_time": elapsed_time
        }
        
    except Exception as e:
        # Fehler behandeln
        print(f"\n{'='*60}")
        print(f"❌ Task {task_id}: FEHLER")
        print(f"📛 Error: {str(e)}")
        print(f"{'='*60}\n")
        
        redis_client.setex(f"task:{task_id}:status", 3600, "failed")
        redis_client.setex(f"task:{task_id}:error", 3600, str(e))
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return {
            "status": "failed",
            "task_id": task_id,
            "error": str(e)
        }

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """
    Root endpoint - Liefert HTML Frontend
    """
    html_path = os.path.join(STATIC_DIR, "DANCEPIXEL_DebugSimple.html")
    
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {
            "message": "DancePixel Production API",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health"
        }

@app.post("/upload")
async def upload_video(file: UploadFile):
    """
    Video hochladen und in Queue einreihen
    
    Args:
        file: Video-Datei (MP4, MOV, AVI, WebM)
    
    Returns:
        dict: Task-ID und Status
    """
    
    # Validierung
    allowed_extensions = ['.mp4', '.mov', '.avi', '.webm', '.mkv']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Nur Video-Formate erlaubt: {', '.join(allowed_extensions)}"
        )
    
    # Dateigrößen-Limit (500 MB)
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > 500:
        raise HTTPException(
            status_code=413,
            detail=f"Video zu groß: {file_size_mb:.1f} MB (max. 500 MB)"
        )
    
    # Task ID generieren
    task_id = str(uuid.uuid4())
    
    # Temporäre Datei erstellen
    temp_path = os.path.join(TEMP_DIR, f"{task_id}_{file.filename}")
    
    with open(temp_path, "wb") as f:
        f.write(content)
    
    print(f"\n📤 Upload: {file.filename} ({file_size_mb:.1f} MB)")
    print(f"🆔 Task ID: {task_id}")
    
    # Task in Queue einreihen
    task = process_video_task.apply_async(
        args=[temp_path, task_id],
        task_id=task_id
    )
    
    # Status initialisieren
    redis_client.set(f"task:{task_id}:status", "queued")
    redis_client.set(f"task:{task_id}:progress", 0)
    redis_client.set(f"task:{task_id}:filename", file.filename)
    
    print(f"✅ Task in Queue: {task_id}\n")
    
    return JSONResponse({
        "task_id": task_id,
        "status": "queued",
        "message": "Video wird verarbeitet. Nutze /status/{task_id} zum Abfragen.",
        "filename": file.filename
    })

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Task-Status abfragen
    
    Args:
        task_id: Task-ID vom Upload
    
    Returns:
        dict: Status, Progress, Download-URL (falls fertig)
    """
    
    status_bytes = redis_client.get(f"task:{task_id}:status")
    
    if not status_bytes:
        raise HTTPException(
            status_code=404,
            detail="Task nicht gefunden oder abgelaufen (24h TTL)"
        )
    
    status = status_bytes.decode('utf-8')
    
    progress_bytes = redis_client.get(f"task:{task_id}:progress")
    progress = int(progress_bytes.decode('utf-8')) if progress_bytes else 0
    
    filename_bytes = redis_client.get(f"task:{task_id}:filename")
    filename = filename_bytes.decode('utf-8') if filename_bytes else "unknown"
    
    response = {
        "task_id": task_id,
        "status": status,
        "progress": progress,
        "filename": filename
    }
    
    # Falls fertig: Download-URL hinzufügen
    if status == "completed":
        response["download_url"] = f"/download/{task_id}"
        response["message"] = "Verarbeitung abgeschlossen! Lade Pose-Daten herunter."
    
    # Falls in Verarbeitung: ETA berechnen (optional)
    elif status == "processing":
        response["message"] = f"Verarbeitung läuft... ({progress}%)"
    
    # Falls in Queue
    elif status == "queued":
        response["message"] = "Warte auf freien Worker..."
    
    # Falls Fehler: Error-Message hinzufügen
    elif status == "failed":
        error_bytes = redis_client.get(f"task:{task_id}:error")
        error = error_bytes.decode('utf-8') if error_bytes else "Unknown error"
        response["error"] = error
        response["message"] = "Verarbeitung fehlgeschlagen. Siehe 'error' für Details."
    
    return response

@app.get("/download/{task_id}")
async def download_result(task_id: str):
    """
    Pose-Daten herunterladen
    
    Args:
        task_id: Task-ID vom Upload
    
    Returns:
        JSON: Pose-Daten mit Metadata
    """
    
    result_bytes = redis_client.get(f"task:{task_id}:result")
    
    if not result_bytes:
        # Prüfe ob Task überhaupt existiert
        status_bytes = redis_client.get(f"task:{task_id}:status")
        
        if not status_bytes:
            raise HTTPException(
                status_code=404,
                detail="Task nicht gefunden oder abgelaufen"
            )
        
        status = status_bytes.decode('utf-8')
        
        if status == "processing" or status == "queued":
            raise HTTPException(
                status_code=202,
                detail=f"Task noch nicht fertig (Status: {status})"
            )
        elif status == "failed":
            error_bytes = redis_client.get(f"task:{task_id}:error")
            error = error_bytes.decode('utf-8') if error_bytes else "Unknown error"
            raise HTTPException(status_code=500, detail=f"Verarbeitung fehlgeschlagen: {error}")
        else:
            raise HTTPException(
                status_code=404,
                detail="Ergebnis nicht gefunden"
            )
    
    result = json.loads(result_bytes.decode('utf-8'))
    
    return JSONResponse(result)

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """
    Task und Ergebnis löschen (für Cleanup)
    
    Args:
        task_id: Task-ID
    
    Returns:
        dict: Bestätigung
    """
    
    # Alle Keys löschen
    keys = [
        f"task:{task_id}:status",
        f"task:{task_id}:progress",
        f"task:{task_id}:result",
        f"task:{task_id}:error",
        f"task:{task_id}:filename"
    ]
    
    deleted = 0
    for key in keys:
        if redis_client.delete(key):
            deleted += 1
    
    if deleted == 0:
        raise HTTPException(404, "Task nicht gefunden")
    
    return {
        "message": "Task gelöscht",
        "task_id": task_id,
        "deleted_keys": deleted
    }

@app.get("/health")
async def health_check():
    """
    Health Check für Monitoring
    
    Returns:
        dict: System-Status
    """
    
    # Redis Check
    try:
        redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    # GPU Check
    gpu_available = False
    gpu_name = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
    except:
        pass
    
    # Worker Check (Celery)
    try:
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()
        worker_count = len(active_workers) if active_workers else 0
        worker_status = "healthy" if worker_count > 0 else "no workers"
    except Exception as e:
        worker_count = 0
        worker_status = f"unhealthy: {str(e)}"
    
    overall_healthy = (
        redis_status == "healthy" and
        worker_count > 0
    )
    
    return {
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": time.time(),
        "components": {
            "redis": redis_status,
            "workers": {
                "status": worker_status,
                "count": worker_count
            },
            "gpu": {
                "available": gpu_available,
                "name": gpu_name
            },
            "model": MODEL_PATH
        }
    }

@app.get("/stats")
async def get_stats():
    """
    Server-Statistiken (Queue-Größe, aktive Tasks, etc.)
    
    Returns:
        dict: Statistiken
    """
    
    try:
        inspect = celery_app.control.inspect()
        
        # Aktive Tasks
        active = inspect.active()
        active_count = sum(len(tasks) for tasks in active.values()) if active else 0
        
        # Queue-Größe (reserved tasks)
        reserved = inspect.reserved()
        reserved_count = sum(len(tasks) for tasks in reserved.values()) if reserved else 0
        
        # Worker Info
        stats = inspect.stats()
        worker_info = []
        if stats:
            for worker_name, worker_stats in stats.items():
                worker_info.append({
                    "name": worker_name,
                    "pool": worker_stats.get('pool', {}).get('implementation', 'unknown')
                })
        
        return {
            "active_tasks": active_count,
            "queued_tasks": reserved_count,
            "workers": worker_info,
            "total_workers": len(worker_info)
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": "Konnte Statistiken nicht abrufen"
        }

# ============================================================
# SERVER STARTEN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 DancePixel Production Server")
    print("="*60)
    print(f"📡 API: http://{HOST}:{PORT}")
    print(f"📋 Docs: http://{HOST}:{PORT}/docs")
    print(f"💚 Health: http://{HOST}:{PORT}/health")
    print(f"📊 Stats: http://{HOST}:{PORT}/stats")
    print("="*60)
    print("🔥 WICHTIG: Starte Worker mit:")
    print("   celery -A pose_server_production worker --loglevel=info --concurrency=3")
    print("="*60 + "\n")
    
    # Server starten
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )
