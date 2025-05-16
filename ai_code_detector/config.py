"""
Configuration settings for the AI code detection project.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    "xgboost": {
        # Encoder settings
        "encoder": {
            "model_name": "microsoft/unixcoder-base",
            "max_length": 512,
            "batch_size": 16,
            "language_prefixes": {
                "python": "<python> ",
                "java": "<java> ",
                "cpp": "<cpp> "
            }
        },
        
        # XGBoost parameters
        "params": {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8, 
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "tree_method": "hist",
            "predictor": "auto",
            "n_jobs": -1,
            "random_state": 42,
            "scale_pos_weight": 1.0,
            "base_score": 0.5  # Required for binary classification
        },
        
        # Training settings
        "training": {
            "num_boost_round": 2000,
            "early_stopping_rounds": 20,
            "validation_split": 0.2,
            "batch_size": 16,
            "balance_ratio": 5.0  # Max ratio between classes (0 = no balancing)
        },
        
        # Hyperparameter tuning settings
        "hyperparameter_tuning": {
            "param_distributions": {
                "max_depth": (3, 10),
                "learning_rate": (0.01, 0.2),
                "subsample": (0.6, 1.0),
                "colsample_bytree": (0.6, 1.0),
                "min_child_weight": (1, 6),
                "scale_pos_weight": (0.8, 1.2)
            },
            "n_iter": 25,
            "cv": 3
        }
    }
}

# File paths
FILE_PATHS = {
    "embeddings": os.path.join(DATA_DIR, "code_embeddings.pkl"),
    "model": os.path.join(MODELS_DIR, "xgboost_model.pkl"),
    "feature_importance": os.path.join(MODELS_DIR, "xgboost_importance.json")
}

# Feature columns used in the model
FEATURE_COLUMNS = [
    'avgFunctionLength', 
    'avgIdentifierLength', 
    'avgLineLength', 
    'emptyLinesDensity', 
    'functionDefinitionDensity', 
    'maintainabilityIndex', 
    'maxDecisionTokens', 
    'whiteSpaceRatio'
]

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
} 