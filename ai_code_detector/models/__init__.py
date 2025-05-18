"""
Models for AI code detection.

This package contains model implementations for detecting AI-generated code.
Currently supported models:
- XGBoost with UnixCoder embeddings
- UnixCoder classifier with classification head
"""

from ai_code_detector.models.feature_extractor import FeatureExtractor
from ai_code_detector.models.unixcoder_classifier import CodeDataset, UnixCoderClassifier, UnixCoderClassifierTrainer
from ai_code_detector.models.unixcoder_embedder import UnixCoderEncoder
from ai_code_detector.models.xgboost_classifier import XGBoostClassifier

__all__ = [
    'XGBoostClassifier', 
    'UnixCoderEncoder', 
    'FeatureExtractor',
    'UnixCoderClassifier',
    'UnixCoderClassifierTrainer',
    'CodeDataset'
] 
