"""
Models for AI code detection.

This package contains model implementations for detecting AI-generated code.
Currently supported models:
- XGBoost with embeddings
- UnixCoder classifier for deep learning
- Generic embedding classifier
"""

from ai_code_detector.models.code_embedder import CodeEmbeddingEncoder
from ai_code_detector.models.embedding_classifier import (
    EmbeddingClassifier,
    EmbeddingDataset,
)
from ai_code_detector.models.unixcoder import UnixCoderDataset, UnixCoderModel
from ai_code_detector.models.xgboost_classifier import XGBoostClassifier
from ai_code_detector.trainer import Trainer

__all__ = [
    # XGBoost ML model
    'XGBoostClassifier',
    
    # Deep Learning models
    'UnixCoderModel',
    'UnixCoderDataset',
    'EmbeddingClassifier',
    'EmbeddingDataset',
    
    # Utilities
    'CodeEmbeddingEncoder',
    'Trainer'
] 
