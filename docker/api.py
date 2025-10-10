#!/usr/bin/env python3
"""
WhisperX Web API
A simple FastAPI wrapper for WhisperX transcription service.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List
import json
import logging

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import whisperx
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="WhisperX API",
    description="Automatic Speech Recognition with word-level timestamps and speaker diarization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web interface
if Path("web").exists():
    app.mount("/static", StaticFiles(directory="web"), name="static")

# Global configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BATCH_SIZE = 16 if DEVICE == "cuda" else 8
OUTPUT_DIR = Path("/output")
MODELS_CACHE = {}

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

class TranscriptionRequest(BaseModel):
    model: str = "large-v2"
    language: Optional[str] = None
    task: str = "transcribe"
    diarize: bool = False
    align: bool = True
    batch_size: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    highlight_words: bool = False
    output_format: str = "json"

class TranscriptionResponse(BaseModel):
    task_id: str
    status: str
    segments: Optional[List] = None
    language: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[dict] = None

# Global task storage (in production, use Redis or database)
TASKS = {}

def get_or_load_model(model_name: str, task_id: str = None):
    """Load and cache WhisperX model with progress tracking"""
    if model_name not in MODELS_CACHE:
        logger.info(f"Loading model: {model_name}")
        
        # Update task status if provided
        if task_id and task_id in TASKS:
            TASKS[task_id]["status"] = "downloading_model"
            TASKS[task_id]["progress"] = {"stage": "model_download", "percent": 0}
        
        try:
            model = whisperx.load_model(
                model_name, 
                DEVICE, 
                compute_type=COMPUTE_TYPE
            )
            MODELS_CACHE[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")
            
            # Update progress
            if task_id and task_id in TASKS:
                TASKS[task_id]["progress"] = {"stage": "model_loaded", "percent": 100}
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            if task_id and task_id in TASKS:
                TASKS[task_id]["status"] = "failed"
                TASKS[task_id]["error"] = f"Model loading failed: {str(e)}"
            raise
            
    return MODELS_CACHE[model_name]

def transcribe_audio_file(
    audio_path: str,
    request: TranscriptionRequest,
    task_id: str
):
    """Background task to transcribe audio file"""
    try:
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["progress"] = {"stage": "initializing", "percent": 0}
        logger.info(f"Starting transcription for task {task_id}")
        
        # Load model with progress tracking
        model = get_or_load_model(request.model, task_id)
        
        # Update progress
        TASKS[task_id]["progress"] = {"stage": "loading_audio", "percent": 20}
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Update progress
        TASKS[task_id]["progress"] = {"stage": "transcribing", "percent": 30}
        
        # Transcribe
        batch_size = request.batch_size or BATCH_SIZE
        result = model.transcribe(audio, batch_size=batch_size)
        
        TASKS[task_id]["language"] = result.get("language")
        TASKS[task_id]["progress"] = {"stage": "transcription_complete", "percent": 60}
        
        # Alignment
        if request.align and not request.language:
            TASKS[task_id]["progress"] = {"stage": "aligning", "percent": 70}
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], 
                device=DEVICE
            )
            result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                DEVICE, 
                return_char_alignments=False
            )
            TASKS[task_id]["progress"] = {"stage": "alignment_complete", "percent": 80}
        
        # Speaker diarization
        if request.diarize:
            TASKS[task_id]["progress"] = {"stage": "diarizing", "percent": 85}
            # Use token from form if provided, otherwise fall back to environment
            hf_token = TASKS[task_id].get("hf_token") or os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN required for diarization. Please provide token in the form or set HF_TOKEN environment variable.")
            
            diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=hf_token, 
                device=DEVICE
            )
            
            diarize_segments = diarize_model(
                audio,
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers
            )
            
            result = whisperx.assign_word_speakers(diarize_segments, result)
            TASKS[task_id]["progress"] = {"stage": "diarization_complete", "percent": 95}
        
        # Finalize
        TASKS[task_id]["progress"] = {"stage": "finalizing", "percent": 98}
        
        # Update task with results
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["segments"] = result.get("segments", [])
        TASKS[task_id]["progress"] = {"stage": "complete", "percent": 100}
        
        # Save output file
        output_file = OUTPUT_DIR / f"{task_id}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Transcription completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Error in transcription task {task_id}: {str(e)}")
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)
        TASKS[task_id]["progress"] = {"stage": "failed", "percent": 0}

@app.get("/")
async def root():
    """Serve web interface or API info"""
    web_file = Path("web/index.html")
    if web_file.exists():
        with open(web_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return {
            "message": "WhisperX API is running",
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "endpoints": {
                "transcribe": "/transcribe",
                "status": "/transcribe/{task_id}",
                "download": "/transcribe/{task_id}/download",
                "health": "/health",
                "models": "/models",
                "languages": "/languages"
            }
        }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": list(MODELS_CACHE.keys())
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    model: str = Form("large-v2"),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
    diarize: bool = Form(False),
    align: bool = Form(True),
    batch_size: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    highlight_words: bool = Form(False),
    output_format: str = Form("json"),
    hf_token: Optional[str] = Form(None)
):
    """Transcribe audio file with WhisperX"""
    
    # Validate file type
    if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded file
    try:
        temp_dir = Path(tempfile.mkdtemp())
        audio_path = temp_dir / audio_file.filename
        
        with open(audio_path, "wb") as buffer:
            buffer.write(await audio_file.read())
        
        # Create transcription request
        request = TranscriptionRequest(
            model=model,
            language=language,
            task=task,
            diarize=diarize,
            align=align,
            batch_size=batch_size,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            highlight_words=highlight_words,
            output_format=output_format
        )
        
        # Initialize task
        TASKS[task_id] = {
            "status": "queued",
            "filename": audio_file.filename,
            "request": request.dict(),
            "hf_token": hf_token  # Store HF token for this task
        }
        
        # Start background transcription
        background_tasks.add_task(
            transcribe_audio_file,
            str(audio_path),
            request,
            task_id
        )
        
        return TranscriptionResponse(
            task_id=task_id,
            status="queued"
        )
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcribe/{task_id}", response_model=TranscriptionResponse)
async def get_transcription_status(task_id: str):
    """Get transcription task status and results"""
    
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = TASKS[task_id]
    
    return TranscriptionResponse(
        task_id=task_id,
        status=task["status"],
        segments=task.get("segments"),
        language=task.get("language"),
        error=task.get("error"),
        progress=task.get("progress")
    )

@app.get("/transcribe/{task_id}/download")
async def download_transcription(task_id: str):
    """Download transcription results as JSON file"""
    
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = TASKS[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Transcription not completed")
    
    output_file = OUTPUT_DIR / f"{task_id}.json"
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_file,
        filename=f"transcription_{task_id}.json",
        media_type="application/json"
    )

@app.get("/models")
async def list_models():
    """List available WhisperX models"""
    models = [
        "tiny", "tiny.en", "base", "base.en", "small", "small.en", 
        "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3"
    ]
    return {"models": models}

@app.get("/models/list")
async def list_available_models():
    """List models that are currently downloaded and available"""
    available_models = list(MODELS_CACHE.keys())
    return {"available_models": available_models}

@app.get("/languages")
async def list_languages():
    """List supported languages"""
    return {"languages": whisperx.utils.LANGUAGES}

@app.post("/models/download")
async def download_model_endpoint(request: dict, background_tasks: BackgroundTasks):
    """Download a model with progress tracking"""
    model_name = request.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    
    allowed_models = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]
    if model_name not in allowed_models:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    if model_name in MODELS_CACHE:
        return {"status": "already_downloaded", "model": model_name}
    
    # Create a download task
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "downloading",
        "model": model_name,
        "progress": 0,
        "message": "Starting download..."
    }
    
    # Start background download
    background_tasks.add_task(download_model_task, model_name, task_id)
    
    return {"status": "downloading", "task_id": task_id, "model": model_name}

@app.get("/models/download/progress/{task_id}")
async def get_download_progress(task_id: str):
    """Get download progress for a specific task"""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = TASKS[task_id]
    return {
        "status": task.get("status", "unknown"),
        "progress": task.get("progress", 0),
        "message": task.get("message", ""),
        "error": task.get("error")
    }

@app.post("/download-model/{model_name}")
async def download_model(model_name: str, background_tasks: BackgroundTasks):
    """Pre-download a model (legacy endpoint)"""
    allowed_models = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]
    if model_name not in allowed_models:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    if model_name in MODELS_CACHE:
        return {"status": "already_downloaded", "model": model_name}
    
    # Create a download task
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "downloading_model",
        "model": model_name,
        "progress": {"stage": "downloading_model", "percent": 0}
    }
    
    # Start background download
    background_tasks.add_task(download_model_task, model_name, task_id)
    
    return {"status": "downloading", "task_id": task_id, "model": model_name}

def download_model_task(model_name: str, task_id: str):
    """Background task to download model with detailed progress"""
    try:
        logger.info(f"Starting download for model: {model_name}")
        
        # Update progress
        TASKS[task_id]["status"] = "downloading"
        TASKS[task_id]["progress"] = 10
        TASKS[task_id]["message"] = f"Initializing download for {model_name}..."
        
        # Download the model (this will trigger the actual download)
        TASKS[task_id]["progress"] = 30
        TASKS[task_id]["message"] = f"Downloading {model_name} model files..."
        
        model = get_or_load_model(model_name, task_id)
        
        # Update progress
        TASKS[task_id]["progress"] = 90
        TASKS[task_id]["message"] = f"Finalizing {model_name}..."
        
        # Mark as completed
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["message"] = f"Model {model_name} downloaded successfully!"
        
        logger.info(f"Model {model_name} downloaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {str(e)}")
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)
        TASKS[task_id]["message"] = f"Download failed: {str(e)}"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WhisperX Web API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api:app" if not args.reload else "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )