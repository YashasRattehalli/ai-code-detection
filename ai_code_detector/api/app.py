"""
FastAPI application for AI-generated code detection.

This module provides an API endpoint to detect if a code snippet was likely
generated by AI or written by a human.
"""

import json
import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException

# Import schema definitions
from ai_code_detector.api.schema import CodeRequest, CodeResponse

# Import project modules
from ai_code_detector.config import FEATURE_COLUMNS, FILE_PATHS, LOGGING_CONFIG, MODEL_CONFIGS
from ai_code_detector.inference_pipeline import InferencePipeline

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Code Detector API",
    description="API for detecting AI-generated code",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "model": "AI Code Detector"}

@app.post("/detect", response_model=CodeResponse)
async def detect_code(request: CodeRequest):
    """
    Detect if code was generated by AI.
    
    Args:
        request: CodeRequest with code
        
    Returns:
        CodeResponse with detection results
    """
    try:
        # Log the code length for debugging
        logger.debug(f"Received code with length: {len(request.code)}")
        
        # Make prediction using the inference pipeline
        pipeline = InferencePipeline(
            model_type="xgboost",
            threshold=0.5,
            model_config=MODEL_CONFIGS["xgboost"],
            model_path=FILE_PATHS["model"],
            importance_path=FILE_PATHS["feature_importance"],
            feature_columns=FEATURE_COLUMNS
        )
        result = pipeline.predict_single(
            code=request.code,
        )
        
        return CodeResponse(
            probability=result["probability"],
            is_ai_generated=result["is_ai_generated"],
            language=result["language"]
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing code: {str(e)}")

@app.get("/model-info")
async def model_info():
    """
    Get information about the model.
    
    Returns:
        Dictionary with model metadata
    """
    # Try to load feature importance
    try:
        with open(FILE_PATHS["feature_importance"], 'r') as f:
            feature_importance = json.load(f)
    except Exception:
        feature_importance = None
    
    return {
        "model_type": "XGBoost",
        "feature_columns": FEATURE_COLUMNS,
        "feature_importance": feature_importance,
        "encoder": MODEL_CONFIGS["xgboost"]["encoder"]["model_name"],
        "max_sequence_length": MODEL_CONFIGS["xgboost"]["encoder"]["max_length"]
    }

if __name__ == "__main__":

    # Run the server with host and port from environment variables or defaults
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app:app", host=host, port=port, reload=False) 