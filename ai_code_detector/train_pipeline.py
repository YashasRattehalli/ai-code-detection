#!/usr/bin/env python3
"""
Training pipeline for AI-generated code detection.

This script handles the end-to-end training process for the code detection model:
1. Data loading and preparation
2. Feature extraction 
3. Embedding generation
4. Model training and evaluation
5. Model saving
"""

import argparse
import ast
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Import project modules
from ai_code_detector.config import FEATURE_COLUMNS, FILE_PATHS, LOGGING_CONFIG, MODEL_CONFIGS
from ai_code_detector.core import CodeDetector

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Class for loading and preparing datasets for AI code detection.
    
    This class handles loading data from CSV files, processing features,
    and sampling balanced datasets.
    """
    
    def __init__(
        self,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the DataLoader.
        
        Args:
            feature_columns: List of feature column names
        """
        self.feature_columns = feature_columns
    
    def sample_balanced_dataset(
        self,
        df: pd.DataFrame,
        total_samples: int
    ) -> pd.DataFrame:
        """
        Create a balanced dataset by sampling from each class.
        
        Args:
            df: Input DataFrame
            total_samples: Total number of samples in the output DataFrame
            
        Returns:
            Balanced DataFrame
        """
        logger.info(f"Balancing dataset to {total_samples} total samples...")
        
        # Get the count of each class
        class_0 = df[df['target_binary'] == 0]
        class_1 = df[df['target_binary'] == 1]
        
        # Determine number of samples per class
        samples_per_class = total_samples // 2
        
        # Sample from each class (or take all if fewer than needed)
        if len(class_0) > samples_per_class:
            class_0 = class_0.sample(samples_per_class, random_state=42)
        if len(class_1) > samples_per_class:
            class_1 = class_1.sample(samples_per_class, random_state=42)
        
        # Combine the balanced dataset
        balanced_df = pd.concat([class_0, class_1])
        
        # Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Balanced dataset created with {len(balanced_df)} samples")
        
        return balanced_df
    
    def load_dataset(
        self,
        dataset_path: str,
        balance_dataset: bool = False,
        total_samples: int = 10000
    ) -> pd.DataFrame:
        """
        Load and prepare the dataset.
        
        Args:
            dataset_path: Path to the dataset CSV file
            balance_dataset: Whether to balance the dataset
            total_samples: Number of samples to use if balancing
            
        Returns:
            DataFrame with processed data
        """
        start_time = time.time()
        logger.info(f"Loading dataset from {dataset_path}...")
        
        # Use optimized CSV reading with only necessary columns
        df = pd.read_csv(
            dataset_path, 
            usecols=['code', 'language', 'target', 'features'],
            dtype={'code': str, 'language': str, 'target': str, 'features': str}
        )
        
        logger.info(f"Loaded {len(df)} records from dataset")
        
        # Map target labels to binary values
        logger.info("Processing target labels and features...")
        df['target_binary'] = df['target'].apply(lambda x: 1 if x == 'ai' else 0)
        
        # Log class distribution
        class_counts = df['target_binary'].value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        # Apply balanced sampling if requested
        if balance_dataset:
            df = self.sample_balanced_dataset(df, total_samples)
        
        # Parse features from JSON string with error handling
        def parse_features(feature_str):
            if pd.isna(feature_str) or feature_str == '':
                return {}
            try:
                return ast.literal_eval(feature_str)
            except (SyntaxError, ValueError) as e:
                logger.warning(f"Invalid feature string found: {feature_str[:50]}... - {str(e)}")
                return {}
        
        df['features'] = df['features'].apply(parse_features)
        
        # Extract individual features
        if self.feature_columns is not None:
            for feature in self.feature_columns:
                df[feature] = df['features'].apply(lambda x: x.get(feature, 0))
        
        elapsed_time = time.time() - start_time
        logger.info(f"Dataset loading completed in {elapsed_time:.2f} seconds")
        
        return df


class TrainingPipeline(CodeDetector):
    """
    Pipeline for training the AI code detection model.
    
    This class extends the base CodeDetector to provide functionality
    for training, hyperparameter tuning, and cross-validation.
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        model_config: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        embeddings_path: Optional[str] = None
    ):
        """
        Initialize the training pipeline.
        
        Args:
            model_type: Type of model to train ("xgboost" by default)
            model_config: Optional model configuration
            feature_columns: List of feature column names
            model_path: Path to save the trained model
            embeddings_path: Path to save/load embeddings
        """
        # Get configurations
        self.model_type = model_type
        
        # Use provided config or get from MODEL_CONFIGS
        if model_config is None:
            if model_type not in MODEL_CONFIGS:
                raise ValueError(f"Unsupported model type: {model_type}. "
                                 f"Available options: {list(MODEL_CONFIGS.keys())}")
            model_config = MODEL_CONFIGS[model_type]
        
        # Use provided paths or get from FILE_PATHS
        if model_path is None:
            model_path = FILE_PATHS["model"]
        if embeddings_path is None:
            embeddings_path = FILE_PATHS["embeddings"]

        
        # Initialize base class
        super().__init__(
            model_config=model_config,
            feature_columns=feature_columns,
            model_path=model_path,
            embeddings_path=embeddings_path
        )
        
        # Initialize data loader
        self.data_loader = DataLoader(feature_columns=feature_columns)
    
    def process_dataset(
        self,
        dataset_path: str,
        save_embeddings: bool = True,
        load_embeddings: bool = True,
        balance_dataset: bool = True,
        total_samples: int = 10000
    ) -> pd.DataFrame:
        """
        Process the dataset and create embeddings.
        
        Args:
            dataset_path: Path to the dataset CSV file
            save_embeddings: Whether to save generated embeddings
            load_embeddings: Whether to try loading cached embeddings
            balance_dataset: Whether to balance the dataset
            total_samples: Number of samples to use if balancing
            
        Returns:
            DataFrame with processed data including embeddings
        """
        start_time = time.time()
        
        # Load and prepare dataset
        df = self.data_loader.load_dataset(
            dataset_path=dataset_path,
            balance_dataset=balance_dataset,
            total_samples=total_samples
        )
        
        # Check if we can load from cache
        embeddings_array = None
        if load_embeddings:
            embeddings_array = self.load_embeddings()
            
            if embeddings_array is not None and len(embeddings_array) == len(df):
                logger.info("Using cached embeddings")
                df['embedding'] = list(embeddings_array)
                return df
            elif embeddings_array is not None:
                logger.warning("Cached embeddings don't match dataset size. Regenerating...")
        
        # Generate embeddings in optimized batches
        logger.info("Generating code embeddings...")
        codes = df['code'].tolist()
        langs = df['language'].tolist()
        
        # Get batch size from config or use default
        batch_size = self.model_config.get("encoder", {}).get("batch_size", 16)
        
        # Generate embeddings
        embeddings_array = self.generate_embeddings(codes, langs, batch_size)
        
        # Save embeddings to file if requested
        if save_embeddings:
            self.save_embeddings(embeddings_array)
        
        # Store embeddings in dataframe
        df['embedding'] = list(embeddings_array)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Dataset processing completed in {elapsed_time:.2f} seconds")
        
        return df
    
    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for the model using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with best parameters
        """
        logger.info("Tuning hyperparameters with Optuna...")
        
        # Check if we have hyperparameter tuning config
        if "hyperparameter_tuning" not in self.model_config:
            logger.warning("No hyperparameter tuning configuration found, skipping")
            return {}
            
        config = self.model_config["hyperparameter_tuning"]
        
        import optuna
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        
        def objective(trial):
            # Define the hyperparameters to optimize
            param = {
                'learning_rate': trial.suggest_float('learning_rate', 
                                                  config["param_distributions"]["learning_rate"][0],
                                                  config["param_distributions"]["learning_rate"][1]),
                'max_depth': trial.suggest_int('max_depth', 
                                            config["param_distributions"]["max_depth"][0],
                                            config["param_distributions"]["max_depth"][1]),
                'min_child_weight': trial.suggest_int('min_child_weight', 
                                                    config["param_distributions"]["min_child_weight"][0],
                                                    config["param_distributions"]["min_child_weight"][1]),
                'subsample': trial.suggest_float('subsample', 
                                               config["param_distributions"]["subsample"][0],
                                               config["param_distributions"]["subsample"][1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 
                                                      config["param_distributions"]["colsample_bytree"][0],
                                                      config["param_distributions"]["colsample_bytree"][1]),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight',
                                                      config["param_distributions"]["scale_pos_weight"][0],
                                                      config["param_distributions"]["scale_pos_weight"][1])
            }
            
            # Create XGBoost classifier with trial parameters
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                tree_method='hist',
                n_jobs=-1,
                random_state=42,
                **param
            )
            
            # Evaluate using cross-validation
            score = cross_val_score(
                xgb_model, 
                X_train, 
                y_train, 
                cv=config["cv"],
                scoring='roc_auc', 
                n_jobs=-1
            ).mean()
            
            return score
        
        # Create Optuna study for maximizing score
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        
        # Optimize with specified number of trials
        study.optimize(objective, n_trials=config["n_iter"])
        
        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best score: {study.best_value}")
        
        return best_params
    
    def train_model(
        self,
        df: pd.DataFrame,
        tune_params: bool = True,
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the model using the processed dataset.
        
        Args:
            df: DataFrame with processed data
            tune_params: Whether to perform hyperparameter tuning
            n_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary with metrics and results
        """
        # Prepare features using the classifier helper
        X, y = self.classifier.prepare_features(df)
        
        # Set up stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Initialize metrics tracking
        all_metrics = []
        
        # Tune hyperparameters if requested (using the first fold)
        if tune_params:
            first_fold = next(skf.split(X, y))
            train_idx, _ = first_fold
            best_params = self.tune_hyperparameters(X[train_idx], y[train_idx])
            
            # Update the classifier parameters
            for param, value in best_params.items():
                self.classifier.params[param] = value
        
        # Perform k-fold cross-validation
        logger.info(f"Performing {n_folds}-fold stratified cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training fold {fold+1}/{n_folds}")
            
            # Get train and validation data for this fold
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train the model for this fold
            training_config = self.model_config.get("training", {})
            fold_metrics = self.classifier.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                num_boost_round=training_config.get("num_boost_round", 2000),
                early_stopping_rounds=training_config.get("early_stopping_rounds", 20)
            )
            
            # Add fold number to metrics
            fold_metrics['fold'] = fold + 1
            all_metrics.append(fold_metrics)
            
            # Log fold results
            logger.info(f"Fold {fold+1} results: {fold_metrics}")
        
        # Calculate average metrics across folds
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'fold':
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        logger.info(f"Average metrics across {n_folds} folds: {avg_metrics}")
        
        return {
            'fold_metrics': all_metrics,
            'average_metrics': avg_metrics
        }
    
    def train_final_model(
        self,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Train the final model on all data.
        
        Args:
            df: DataFrame with processed data
            
        Returns:
            Dictionary with model metrics
        """
        logger.info("Training final model on all data...")
        
        # Prepare features
        X, y = self.classifier.prepare_features(df)
        
        # Create a dummy validation set (we won't use early stopping)
        X_val = X[:100]  # Just a small subset for validation
        y_val = y[:100]
        
        # Train the model on all data
        training_config = self.model_config.get("training", {})
        metrics = self.classifier.train(
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            num_boost_round=training_config.get("num_boost_round", 2000),
            early_stopping_rounds=None  # No early stopping for final model
        )
        
        # Save the model with metrics
        self.classifier.save_model(
            model_path=self.model_path,
            metrics=metrics
        )
        
        logger.info(f"Final model saved to {self.model_path}")
        
        return metrics
    
    def train(
        self,
        dataset_path: str,
        tune_params: bool = True,
        n_folds: int = 5,
        balance_ratio: float = 0
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            dataset_path: Path to the dataset CSV file
            tune_params: Whether to perform hyperparameter tuning
            n_folds: Number of folds for cross-validation
            balance_ratio: Maximum ratio between classes (0 = no balancing)
            
        Returns:
            Dictionary with training results and metrics
        """
        start_time = time.time()
        
        # Override balance ratio if provided
        if balance_ratio > 0:
            self.model_config["training"]["balance_ratio"] = balance_ratio
        
        # Process dataset
        df = self.process_dataset(
            dataset_path=dataset_path,
            save_embeddings=True,
            load_embeddings=True,
            balance_dataset=True
        )
        
        # Train with cross-validation
        cv_results = self.train_model(
            df=df,
            tune_params=tune_params,
            n_folds=n_folds
        )
        
        # Train final model on all data
        final_metrics = self.train_final_model(df)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Training pipeline completed successfully in {elapsed_time:.2f} seconds!")
        
        # Return all metrics for potential logging/reporting
        return {
            'fold_metrics': cv_results['fold_metrics'],
            'average_metrics': cv_results['average_metrics'],
            'final_model_metrics': final_metrics
        }

def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='Train the AI-generated code detection model')
    parser.add_argument('--dataset', type=str, 
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv'), 
                        help='Path to the dataset CSV file')
    parser.add_argument('--model-type', type=str, default='xgboost', help='Type of model to train')
    parser.add_argument('--tune-params', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--balance-ratio', type=float, default=0, 
                        help='Maximum ratio between classes (0 = no balancing, 5 = at most 5:1 ratio)')
    parser.add_argument('--n-folds', type=int, default=5, 
                        help='Number of folds for stratified k-fold cross-validation')
    
    args = parser.parse_args()
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(
        model_type=args.model_type
    )
    
    # Run the training pipeline
    pipeline.train(
        dataset_path=args.dataset,
        tune_params=args.tune_params,
        n_folds=args.n_folds,
        balance_ratio=args.balance_ratio
    )

if __name__ == "__main__":
    main() 