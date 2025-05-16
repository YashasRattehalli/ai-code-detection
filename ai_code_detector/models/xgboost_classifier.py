"""
XGBoost-based code detection model with UnixCoder embeddings.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd  # type: ignore
import xgboost as xgb  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    confusion_matrix,
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
    
    def __init__(self, params: Optional[Dict[str, Any]] = None, feature_columns: Optional[List[str]] = None):
        """
        Initialize the classifier with optional parameters.
        
        Args:
            params: XGBoost parameters dictionary
            feature_columns: List of feature column names
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
            
        self.feature_columns = feature_columns
        self.model: Optional[xgb.Booster] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        
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
        # Check if required columns exist
        required_columns = ['embedding']
        if self.feature_columns:
            required_columns.extend(self.feature_columns)
            
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Combine embeddings with other features
        X_embeddings = np.stack(df['embedding'].values)
        
        # If we have additional features, include them
        if self.feature_columns:
            X_features = df[self.feature_columns].values
            X = np.hstack((X_embeddings, X_features))
        else:
            X = X_embeddings
        
        # Get target if available, otherwise return dummy
        if target_column in df.columns:
            y = df[target_column].values
        else:
            y = np.zeros(len(df))  # Dummy values for inference
        
        return X, y
    
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
              early_stopping_rounds: int = 20) -> Dict[str, float]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Number of rounds for early stopping
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create DMatrix objects for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set up watchlist for early stopping
        watchlist = [(dtrain, 'train'), (dval, 'val')]
        
        # Train model with early stopping
        logger.info("Training XGBoost model...")
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        # Store feature importances
        if self.model is not None:
            try:
                # Get importance scores
                importance_scores = self.model.get_score(importance_type='gain')
                
                # Normalize to get relative importance - handle empty dictionary
                if importance_scores:
                    # Convert all values to float and calculate total
                    float_scores = {k: float(v) for k, v in importance_scores.items()}
                    total_importance = sum(float_scores.values())
                    
                    # Create normalized dictionary
                    if total_importance > 0:
                        self.feature_importance = {
                            k: float_scores[k] / total_importance 
                            for k in float_scores
                        }
                    else:
                        self.feature_importance = float_scores
                    
                    # Log top 10 important features
                    sorted_features = sorted(
                        self.feature_importance.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    logger.info("Top feature importances:")
                    for feat, imp in sorted_features[:10]:
                        logger.info(f"  {feat}: {imp:.4f}")
            except Exception as e:
                logger.warning(f"Error processing feature importance: {e}")
        
        # Make predictions on validation set
        assert self.model is not None, "Model training failed"
        y_pred_proba = self.model.predict(dval)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate comprehensive metrics
        metrics = self.evaluate_model(y_val, y_pred, y_pred_proba)
        
        # Display confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        logger.info(f"Confusion matrix:\n{cm}")
        
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
    
    def save_model(self, model_path: str, feature_importance_path: Optional[str] = None) -> None:
        """
        Save the trained model to a file.
        
        Args:
            model_path: Path to save the model
            feature_importance_path: Path to save feature importance
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        logger.info(f"Saving model to {model_path}...")
        cast(xgb.Booster, self.model).save_model(model_path)
        
        # Save feature importance if available
        if self.feature_importance:
            if feature_importance_path is None:
                feature_importance_path = f"{os.path.splitext(model_path)[0]}_importance.json"
                
            with open(feature_importance_path, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
    
    def load_model(self, model_path: str, feature_importance_path: Optional[str] = None) -> None:
        """
        Load a model from a file.
        
        Args:
            model_path: Path to the model file
            feature_importance_path: Path to the feature importance file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        logger.info(f"Loading model from {model_path}...")
        self.model = xgb.Booster()
        cast(xgb.Booster, self.model).load_model(model_path)
        
        # Try to load feature importance if available
        if feature_importance_path is None:
            feature_importance_path = f"{os.path.splitext(model_path)[0]}_importance.json"
            
        if os.path.exists(feature_importance_path):
            with open(feature_importance_path, 'r') as f:
                self.feature_importance = json.load(f)