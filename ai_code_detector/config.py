"""
Configuration settings for the AI code detection project.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

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
    },
    "embedding_classifier": {
        # Encoder settings
        "encoder": {
            "model_name": "microsoft/unixcoder-base",
            "max_length": 512,
            "batch_size": 16
        },
        
        # Model parameters
        "params": {
            "model_name": "microsoft/unixcoder-base",
            "num_classes": 2,
            "dropout_rate": 0.1
        },
        
        # Training settings
        "training": {
            "batch_size": 16,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "warmup_steps": 0,
            "weight_decay": 0.01,
            "eval_steps": 100,
            "validation_split": 0.2
        }
    }
}

# File paths
INFERENCE_FILE_PATHS = {
    "embeddings": os.path.join(EMBEDDINGS_DIR, "unixcoder_embeddings.pkl"),
    "xgboost": os.path.join(MODELS_DIR, "xgboost_model.pkl"),
    "xgboost_model_info": os.path.join(MODELS_DIR, "xgboost_model_info.json"),
    "unixcoder": os.path.join(MODELS_DIR, "unixcoder_model.pth"),
    "embedding_classifier": os.path.join(MODELS_DIR, "embedding_classifier.pth")
}


# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
} 