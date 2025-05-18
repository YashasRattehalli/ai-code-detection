"""
Classifier adapter for different model backends.

This module provides a common interface for different classifier models
used in the AI code detection system.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from ai_code_detector.models.unixcoder_classifier import UnixCoderClassifierTrainer
from ai_code_detector.models.xgboost_classifier import XGBoostClassifier

# Set up logging
logger = logging.getLogger(__name__)

class ClassifierAdapter:
    """Adapter class that provides a common interface for different model backends."""
    
    def __init__(
        self, 
        model_type: str = "xgboost",
        model_config: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the classifier adapter.
        
        Args:
            model_type: Type of classifier model to use ("xgboost" or "unixcoder")
            model_config: Configuration dictionary for the model
            feature_columns: List of feature column names to use
        """
        self.model_type = model_type
        self.model_config = model_config or {}
        self.feature_columns = feature_columns or []
        self.model = None
        
        # Initialize the underlying classifier
        if model_type == "xgboost":
            self.model = XGBoostClassifier(
                params=self.model_config.get("params", None),
                feature_columns=self.feature_columns
            )
        elif model_type == "unixcoder":
            # For UnixCoder, we don't initialize the full model here
            # It will be loaded via load_model
            self.model = None
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @property
    def params(self) -> Dict[str, Any]:
        """Access the underlying model's parameters."""
        if self.model is None:
            return {}
        elif hasattr(self.model, 'params'):
            return self.model.params
        else:
            return {}
    
    def load_model(
        self, 
        model_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary with model info (parameters and metrics) if available
        """
        model_info = None
        
        if self.model_type == "xgboost":
            if self.model is None:
                self.model = XGBoostClassifier(
                    params=self.model_config.get("params", None),
                    feature_columns=self.feature_columns
                )
            model_info = self.model.load_model(model_path)
        
        elif self.model_type == "unixcoder":
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            try:
                self.model = UnixCoderClassifierTrainer.load_model(model_path, device)
                logger.info(f"Successfully loaded UnixCoder model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading UnixCoder model: {e}")
                raise
                
        return model_info
    
    def predict(self, X: np.ndarray, languages: Optional[List[str]] = None) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            X: Feature matrix or code samples
            languages: Optional list of programming languages (for unixcoder)
            
        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been loaded")
        
        if self.model_type == "xgboost":
            return self.model.predict(X)
        
        elif self.model_type == "unixcoder":
            # For UnixCoder, we need to extract the code samples from X
            # Since we're at inference time, we need to determine what X contains
            # If X is already embedded and has feature columns, we need to extract just code
            if isinstance(X, np.ndarray) and len(X.shape) == 2:
                # If this is a 2D array, assume it's already processed with embeddings
                # UnixCoder doesn't use this directly, but we need original code
                # This is a limitation - we'd need to pass code_samples separately
                logger.warning("UnixCoder classifier cannot use pre-embedded features directly.")
                # For a real implementation, we'd need to track the original code samples
                # We're returning empty probabilities for now
                return np.zeros(len(X))
            
            # If X is a list of code samples (expected for UnixCoder direct prediction)
            if hasattr(self.model, 'predict'):
                try:
                    _, probabilities = self.model.predict(X, languages)
                    return np.array(probabilities)
                except TypeError:
                    # If the model doesn't accept languages parameter
                    _, probabilities = self.model.predict(X)
                    return np.array(probabilities)
            
            # Fallback - should not happen
            logger.error("UnixCoder model does not have predict method")
            return np.zeros(len(X))
            
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'target_binary') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for model training/prediction.
        
        Args:
            df: DataFrame with features and embeddings
            target_column: Column name for the target variable
            
        Returns:
            X: Feature matrix
            y: Target labels
        """
        if self.model is None or self.model_type != "xgboost":
            raise ValueError("Feature preparation is only supported for XGBoost model")
        
        return self.model.prepare_features(df, target_column)
        
    def train(self, **kwargs) -> Dict[str, float]:
        """
        Train the model with the provided parameters.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None or self.model_type != "xgboost":
            raise ValueError("Training is only supported for XGBoost model")
            
        return self.model.train(**kwargs)
    
    def save_model(self, model_path: str, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Save the trained model to a file.
        
        Args:
            model_path: Path to save the model
            metrics: Optional dictionary with model evaluation metrics to save
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        if self.model_type == "xgboost":
            self.model.save_model(model_path, metrics=metrics)
        else:
            raise ValueError(f"Saving {self.model_type} model is not implemented") 