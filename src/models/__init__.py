"""
Models for AI code detection.

This package contains model implementations for detecting AI-generated code.
Currently supported models:
- XGBoost with UnixCoder embeddings
"""

from models.xgboost_model import XGBoostClassifier, UnixCoderEncoder

__all__ = ['XGBoostClassifier', 'UnixCoderEncoder'] 