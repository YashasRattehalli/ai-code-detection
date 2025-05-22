"""
XGBoost-based code detection model with UnixCoder embeddings.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, cast

import numpy as np
import xgboost as xgb  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Set up logging
logger = logging.getLogger(__name__)



class XGBoostClassifier:
    """
    XGBoost classifier for code detection with advanced features.
    
    This classifier combines code embeddings with additional features
    to detect AI-generated code.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the classifier with optional parameters.
        
        Args:
            params: XGBoost parameters dictionary
        """
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'tree_method': 'hist',
            'predictor': 'auto',
            'n_jobs': -1,
            'random_state': 42,
            'scale_pos_weight': 1.0,
            'base_score': 0.5  # Default value for binary classification
        }
        
        # Ensure base_score is set for binary classification
        if 'base_score' not in self.params:
            self.params['base_score'] = 0.5
            
        self.model: Optional[xgb.Booster] = None
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        logger.info("Model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        return metrics
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              num_boost_round: int = 2000,
              early_stopping_rounds: Optional[int] = 20) -> Dict[str, float]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Number of rounds for early stopping, None to disable
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create DMatrix objects for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set up watchlist for monitoring
        watchlist = [(dtrain, 'train'), (dval, 'val')]
        
        # Train model with optional early stopping
        logger.info("Training XGBoost model...")
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        # Make predictions on validation set
        assert self.model is not None, "Model training failed"
        y_pred_proba = self.model.predict(dval)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate comprehensive metrics
        metrics = self.evaluate_model(y_val, y_pred, y_pred_proba)
        
        return metrics
    

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        # Create DMatrix for prediction
        dmatrix = xgb.DMatrix(X)
        
        # Make predictions
        return cast(xgb.Booster, self.model).predict(dmatrix)
    
    def save_model(self, model_path: str, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Save the trained model to a file along with parameters and metrics.
        
        Args:
            model_path: Path to save the model
            metrics: Optional dictionary with model evaluation metrics to save
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if not model_path:
            raise ValueError("model_path must be provided and cannot be None or empty")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        logger.info(f"Saving model to {model_path}...")
        cast(xgb.Booster, self.model).save_model(model_path)
        
        # Save model parameters and metrics
        model_info_path = f"{os.path.splitext(model_path)[0]}_info.json"
        model_info = {
            'parameters': self.params
        }
        
        # Add metrics if available
        if metrics:
            model_info['metrics'] = metrics
        
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model parameters and metrics saved to {model_info_path}")
    
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a model from a file.
        
        Args:
            model_path: Path to the model file
            feature_importance_path: Deprecated, kept for backward compatibility
            
        Returns:
            Dictionary with model info (parameters and metrics) if available
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        logger.info(f"Loading model from {model_path}...")
        self.model = xgb.Booster()
        cast(xgb.Booster, self.model).load_model(model_path)
        
        # Try to load model info if available
        model_info_path = f"{os.path.splitext(model_path)[0]}_info.json"
        model_info = {}
        
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                
            # Update parameters if available
            if 'parameters' in model_info:
                self.params = model_info['parameters']
                
            logger.info(f"Loaded model info from {model_info_path}")
            
        return model_info