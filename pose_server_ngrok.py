"""
🚀 Professional Pose Estimation Backend mit ngrok
YOLOv8-Pose + FastAPI + ngrok Tunnel

Dieser Server:
- Hostet die HTML-Datei (DANCEPIXEL_DebugSimple.html)
- Bietet die Pose Detection API
- Startet automatisch einen öffentlichen ngrok-Tunnel
- Zeigt die öffentliche URL an

Setup:
1. ngrok Auth Token konfigurieren (einmalig):
   ngrok config add-authtoken YOUR_TOKEN

2. Python dependencies installieren:
   pip install pyngrok

3. Server starten:
   python pose_server_ngrok.py
"""

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
import json
import asyncio
import tempfile
import os
from io import BytesIO
from PIL import Image
import time
from pathlib import Path
import uuid
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import threading

# Ultralytics YOLO
from ultralytics import YOLO

# ngrok für public URL
from pyngrok import ngrok

app = FastAPI(title="Professional Pose Estimation API (ngrok)")

# CORS für Browser-Zugriff
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files für Fonts (falls vorhanden)
try:
    font_dir = Path(__file__).parent / "DEM-MO typeface"
    if font_dir.exists():
        app.mount("/DEM-MO typeface", StaticFiles(directory=str(font_dir)), name="fonts")
        print("✅ Font-Verzeichnis gemountet")
except Exception as e:
    print(f"ℹ️  Fonts nicht verfügbar: {e}")

# ==========================================
# Model laden
# ==========================================
print("🔄 Lade YOLOv8-Pose Model...")
model = YOLO('yolov8x-pose.pt')  # Automatischer Download beim ersten Mal
print("✅ YOLOv8-Pose geladen!")

# ==========================================
# THREAD POOL für parallele Verarbeitung
# ==========================================
# Anzahl der Worker = CPU Cores (oder weniger falls GPU limitiert)
MAX_WORKERS = min(4, os.cpu_count() or 2)  # Max 4 parallele Videos
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="VideoWorker")
print(f"🔧 Thread Pool initialisiert mit {MAX_WORKERS} Workers")

# Thread-safe model access (YOLO ist nicht thread-safe by default)
model_lock = threading.Lock()

# ==========================================
# VIDEO QUEUE SYSTEM (Multi-User Support)
# ==========================================
# In-Memory Storage für Video-Jobs (pro User-Session)
video_jobs: Dict[str, Dict[str, any]] = {}
# Format: {
#   "job_id": {
#       "status": "processing" | "completed" | "failed",
#       "filename": "video.mp4",
#       "user_session": "session_uuid",
#       "progress": 0-100,
#       "result": {...},
#       "error": "...",
#       "created_at": datetime,
#       "updated_at": datetime
#   }
# }

def create_video_job(filename: str, user_session: str) -> str:
    """Erstelle einen neuen Video-Job"""
    job_id = str(uuid.uuid4())
    video_jobs[job_id] = {
        "status": "queued",
        "filename": filename,
        "user_session": user_session,
        "progress": 0,
        "result": None,
        "error": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    return job_id

def get_user_jobs(user_session: str) -> List[Dict]:
    """Hole alle Jobs eines Users"""
    user_jobs = []
    for job_id, job_data in video_jobs.items():
        if job_data["user_session"] == user_session:
            user_jobs.append({
                "job_id": job_id,
                **job_data
            })
    return sorted(user_jobs, key=lambda x: x["created_at"], reverse=True)

def update_job_status(job_id: str, status: str, progress: int = None, result: dict = None, error: str = None):
    """Update Job Status"""
    if job_id in video_jobs:
        video_jobs[job_id]["status"] = status
        video_jobs[job_id]["updated_at"] = datetime.now()
        if progress is not None:
            video_jobs[job_id]["progress"] = progress
        if result is not None:
            video_jobs[job_id]["result"] = result
        if error is not None:
            video_jobs[job_id]["error"] = error

# Optional: RTMPose
try:
    # from mmpose.apis import MMPoseInferencer
    # rtmpose_model = MMPoseInferencer('rtmpose-m')
    # print("✅ RTMPose geladen!")
    rtmpose_available = False
except:
    rtmpose_available = False
    print("ℹ️ RTMPose nicht verfügbar (optional)")

# ==========================================
# Keypoint Namen (COCO Format)
# ==========================================
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

SKELETON_CONNECTIONS = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # Kopf
    [5, 7], [7, 9], [6, 8], [8, 10],  # Arme
    [5, 6], [5, 11], [6, 12], [11, 12],  # Torso
    [11, 13], [13, 15], [12, 14], [14, 16]  # Beine
]

# ==========================================
# Pose Detection
# ==========================================
def detect_pose_yolo(image_np):
    """
    YOLOv8-Pose Detection (Thread-Safe)
    
    Returns:
        list: [{
            'keypoints': [[x, y, conf], ...],
            'bbox': [x, y, w, h],
            'confidence': float
        }]
    """
    # Thread-safe model access
    with model_lock:
        results = model(image_np, verbose=False)
    
    poses = []
    for result in results:
        if result.keypoints is None:
            continue
            
        for i in range(len(result.keypoints)):
            # Keypoints: [17, 3] - (x, y, confidence)
            kpts = result.keypoints[i].data.cpu().numpy()[0]
            
            # Bounding Box
            if result.boxes is not None and len(result.boxes) > i:
                box = result.boxes[i].xyxy.cpu().numpy()[0]
                conf = float(result.boxes[i].conf.cpu().numpy()[0])
            else:
                # Fallback: Berechne Bbox aus Keypoints
                visible_kpts = kpts[kpts[:, 2] > 0.5]
                if len(visible_kpts) > 0:
                    x_min, y_min = visible_kpts[:, :2].min(axis=0)
                    x_max, y_max = visible_kpts[:, :2].max(axis=0)
                    box = [x_min, y_min, x_max, y_max]
                    conf = 0.9
                else:
                    continue
            
            poses.append({
                'keypoints': kpts.tolist(),
                'bbox': [float(box[0]), float(box[1]), 
                        float(box[2] - box[0]), float(box[3] - box[1])],
                'confidence': conf
            })
    
    return poses

# ==========================================
# HTML Hosting (Multi-File Support)
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    """Serve die Standard-HTML-Datei"""
    return await serve_html_file("dancepixel")

@app.get("/{filename}", response_class=HTMLResponse)
async def serve_html_file(filename: str):
    """Serve HTML-Dateien mit Fallunterscheidung"""
    
    # Mapping von URL-Namen zu Dateinamen
    file_mapping = {
        "dancepixel": "DANCEPIXEL.html",
        "dancepixel_debugsimple": "DANCEPIXEL_DebugSimple.html",
        "dancepixel_debugsimple_ppbtn": "DANCEPIXEL_DebugSimple_PPBtn.html"
    }
    
    # Fallback auf alte Datei falls nicht im Mapping
    filename_lower = filename.lower()
    target_file = file_mapping.get(filename_lower)
    
    if target_file:
        html_path = Path(__file__).parent / target_file
        if html_path.exists():
            print(f"✅ Serving: {target_file}")
            return FileResponse(html_path)
    
    # Fallback: Versuche verfügbare Dateien zu finden
    available_files = []
    for url_name, html_file in file_mapping.items():
        file_path = Path(__file__).parent / html_file
        if file_path.exists():
            available_files.append(f'<li><a href="/{url_name}">{html_file}</a></li>')
    
    return HTMLResponse(f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
                h1 {{ color: #333; }}
                ul {{ line-height: 2; }}
                a {{ color: #007bff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                .error {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <h1>🎨 DANCEPIXEL Server</h1>
            <p class="error">⚠️ Datei "{filename}" nicht gefunden!</p>
            <h2>Verfügbare Versionen:</h2>
            <ul>
                {''.join(available_files) if available_files else '<li>Keine HTML-Dateien gefunden</li>'}
            </ul>
            <p><a href="/docs">📖 Zur API Dokumentation</a></p>
        </body>
    </html>
    """)

# ==========================================
# WebSocket für Real-time
# ==========================================
@app.websocket("/ws/pose")
async def websocket_pose(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client verbunden")
    
    try:
        while True:
            # Empfange Base64-codiertes Bild
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'frame':
                # Decode Base64 Bild
                img_data = base64.b64decode(message['image'].split(',')[1])
                img = Image.open(BytesIO(img_data))
                img_np = np.array(img)
                
                # Convert RGB to BGR für OpenCV
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Pose Detection
                poses = detect_pose_yolo(img_np)
                
                # Sende zurück
                await websocket.send_json({
                    'type': 'pose',
                    'poses': poses,
                    'timestamp': time.time()
                })
    except Exception as e:
        print(f"❌ WebSocket Fehler: {e}")
    finally:
        print("🔌 Client getrennt")

# ==========================================
# Video Upload & Processing
# ==========================================
@app.post("/upload")
async def upload_video(file: UploadFile = File(...), user_session: str = None):
    """
    Video hochladen und in Queue stellen
    Returns: job_id für Status-Polling
    
    Args:
        file: Video file
        user_session: Query parameter for session ID
    """
    # Generate user session if not provided
    if not user_session:
        user_session = str(uuid.uuid4())
    
    print(f"📁 Video Upload: {file.filename} (User: {user_session})")
    
    # WICHTIG: Datei JETZT lesen, bevor der Request endet
    try:
        file_content = await file.read()
        print(f"📦 File read: {len(file_content)} bytes")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # Erstelle Job
    job_id = create_video_job(file.filename, user_session)
    print(f"✅ Job created: {job_id}")
    
    # Submit to thread pool für PARALLELE Verarbeitung
    executor.submit(process_video_sync, job_id, file.filename, file_content, user_session)
    print(f"🚀 Job submitted to thread pool (active workers: {len(executor._threads)})")
    
    return {
        "success": True,
        "job_id": job_id,
        "user_session": user_session,
        "message": f"Video {file.filename} in queue"
    }

def process_video_sync(job_id: str, filename: str, file_content: bytes, user_session: str):
    """Process video synchron in Thread (für ThreadPoolExecutor)"""
    temp_input = None
    thread_name = threading.current_thread().name
    
    try:
        # Update Status: Processing
        update_job_status(job_id, "processing", progress=0)
        print(f"🎬 [{thread_name}] Start processing: {job_id}")
        
        # Temporäre Datei erstellen
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Video speichern (direkt aus bytes)
        with open(temp_input, 'wb') as f:
            f.write(file_content)
        
        # Video mit OpenCV öffnen
        cap = cv2.VideoCapture(temp_input)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 [{thread_name}] Video: {width}x{height} @ {fps} FPS, {frame_count} frames")
        
        # Pose Detection für alle Frames
        all_poses = []
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Pose Detection
            poses = detect_pose_yolo(frame)
            all_poses.append(poses)
            
            frame_idx += 1
            
            # Progress Update
            progress = int((frame_idx / frame_count) * 100)
            if frame_idx % 30 == 0:
                update_job_status(job_id, "processing", progress=progress)
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"📊 [{thread_name}] Job {job_id}: {progress}% @ {current_fps:.1f} FPS")
        
        cap.release()
        
        elapsed = time.time() - start_time
        print(f"✅ [{thread_name}] Job {job_id} completed in {elapsed:.1f}s")
        
        # Speichere Ergebnis
        result = {
            "metadata": {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": frame_count,
                "processing_time": round(elapsed, 2),
                "filename": filename
            },
            "poses": all_poses
        }
        
        update_job_status(job_id, "completed", progress=100, result=result)
        
    except Exception as e:
        print(f"❌ [{thread_name}] Job {job_id} failed: {e}")
        update_job_status(job_id, "failed", error=str(e))
    finally:
        # Cleanup
        if temp_input and os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except:
                pass

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get Job Status"""
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = video_jobs[job_id]
    
    # Don't send full pose data in status check (too large)
    response = {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
        "progress": job["progress"],
        "error": job["error"]
    }
    
    # Only include full result when completed
    if job["status"] == "completed":
        response["result"] = job["result"]
    
    return response

@app.get("/user-jobs/{user_session}")
async def get_user_jobs_endpoint(user_session: str):
    """Get all jobs for a user session"""
    jobs = get_user_jobs(user_session)
    
    # Return list without full pose data (just metadata)
    jobs_summary = []
    for job in jobs:
        jobs_summary.append({
            "job_id": job["job_id"],
            "status": job["status"],
            "filename": job["filename"],
            "progress": job["progress"],
            "error": job["error"],
            "created_at": job["created_at"].isoformat(),
            "metadata": job["result"]["metadata"] if job["result"] else None
        })
    
    return {
        "user_session": user_session,
        "jobs": jobs_summary
    }

# ==========================================
# Info Endpoint
# ==========================================
@app.get("/info")
async def info():
    return {
        "name": "Professional Pose Estimation API (ngrok)",
        "version": "2.0.0",
        "features": ["Multi-Video Upload", "Async Processing", "User Sessions"],
        "models": {
            "yolov8": "Available",
            "rtmpose": "Available" if rtmpose_available else "Not installed"
        },
        "endpoints": {
            "root": "/ (HTML Interface)",
            "websocket": "/ws/pose",
            "upload": "/upload (POST with video file)",
            "convert": "/convert-to-mov (POST with WebM file)",
            "docs": "/docs"
        }
    }

# ==========================================
# Video Converter Endpoint
# ==========================================
@app.post("/convert-to-mov")
async def convert_to_mov(file: UploadFile = File(...)):
    """Konvertiert WebM zu MOV (für Mac/Safari) - verwendet OpenCV"""
    temp_input = None
    temp_output = None
    
    try:
        print(f"\n🎬 Konvertiere Video: {file.filename}")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_input:
            content = await file.read()
            tmp_input.write(content)
            temp_input = tmp_input.name
        
        # Output path
        output_filename = file.filename.replace('.webm', '.mov').replace('.WEBM', '.mov')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mov') as tmp_output:
            temp_output = tmp_output.name
        
        print(f"📹 Input: {file.filename}")
        print(f"🔄 WebM → MOV (H.264 mit OpenCV)")
        
        # Open input video
        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            raise Exception("Konnte Video nicht öffnen")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Create output video writer
        # FourCC für MOV: 'mp4v' oder 'avc1' (H.264)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception("Konnte Output-Video nicht erstellen")
        
        # Convert frame by frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            # Progress logging
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"📊 Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        out.release()
        
        print(f"✅ Konvertierung erfolgreich: {output_filename} ({frame_count} frames)")
        
        # Return converted file
        return FileResponse(
            temp_output,
            media_type="video/quicktime",
            filename=output_filename,
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}"
            }
        )
        
    except Exception as e:
        print(f"❌ Fehler bei Konvertierung: {e}")
        # Cleanup on error
        if temp_input and os.path.exists(temp_input):
            os.remove(temp_input)
        if temp_output and os.path.exists(temp_output):
            os.remove(temp_output)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
    finally:
        # Note: temp files are cleaned up after FileResponse is sent
        # FileResponse handles temp_output cleanup automatically
        if temp_input and os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except:
                pass

# ==========================================
# Server starten mit ngrok
# ==========================================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 Professional Pose Estimation Server (ngrok)")
    print("="*60)
    
    # ngrok Tunnel starten
    try:
        print("🔄 Starte ngrok Tunnel...")
        tunnel = ngrok.connect(5000, bind_tls=True)
        public_url = tunnel.public_url
        print("✅ ngrok Tunnel aktiv!")
        print(f"🌐 Öffentliche URL: {public_url}")
        print(f"📱 Teile diese URL mit anderen!")
        print("="*60)
        print(f"📡 Lokal - WebSocket: ws://localhost:5000/ws/pose")
        print(f"📤 Lokal - Upload: http://localhost:5000/upload")
        print(f"🌐 Public - HTML: {public_url}")
        print(f"🌐 Public - Upload: {public_url}/upload")
        print(f"📋 API Docs: {public_url}/docs")
        print("="*60)
        print("⚠️  Drücke CTRL+C zum Beenden")
        print("="*60 + "\n")
    except Exception as e:
        print(f"❌ ngrok Fehler: {e}")
        print("ℹ️  Stelle sicher, dass:")
        print("   1. ngrok installiert ist: pip install pyngrok")
        print("   2. Auth Token konfiguriert: ngrok config add-authtoken YOUR_TOKEN")
        print("\n🔄 Starte Server OHNE ngrok (nur lokal)...\n")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server beendet")
        try:
            ngrok.kill()
            print("✅ ngrok Tunnel geschlossen")
        except:
            pass
