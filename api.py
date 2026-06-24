from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from backend.ml.train_regression import train_all_models, train_model
import os

app = FastAPI(title="ML Training API")

@app.get("/")
async def root():
    return {
        "message": "ML Training API is running",
        "docs": "/docs",
        "health": "/health"
    }

class TrainRequest(BaseModel):
    filename: str
    model_name: str 
    train_size: float = 0.75
    model_type: str = 'linear'

class TrainAllRequest(BaseModel):
    filename: str
    train_size: float = 0.75

@app.post("/train")
async def train_single_model(req: TrainRequest, background_tasks: BackgroundTasks):

    path = Path("data") / req.filename 
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Data file '{req.filename}' not found")
    
    valid_types = ['linear', 'ridge', 'decision_tree']
    if req.model_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_type. Choose from: {valid_types}"
        )
    background_tasks.add_task(
        train_model, 
        req.filename, 
        req.model_name, 
        req.train_size, 
        req.model_type
    )
    
    return {
        "message": f"Model '{req.model_name}' training started",
        "model_type": req.model_type,
        "filename": req.filename
    }

@app.post("/train-all")
async def train_all(req: TrainAllRequest, background_tasks: BackgroundTasks):
    path = Path("data") / req.filename 
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Data file '{req.filename}' not found")
    
    background_tasks.add_task(
        train_all_models, 
        req.filename, 
        None, 
        req.train_size, 
        None
    )
    
    return {
        "message": "All models training started",
        "filename": req.filename
    }

@app.get("/models")
async def list_models():
    models_path = Path("models")
    if not models_path.exists():
        return {"models": []}
    
    models = [f.name for f in models_path.glob("*.pkl") if not f.name.endswith("_scaler.pkl")]
    return {"models": models}

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running"}