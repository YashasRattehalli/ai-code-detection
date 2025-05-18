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
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd  # type: ignore
import xgboost as xgb  # type: ignore
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # type: ignore

# Import project modules
from ai_code_detector.config import FILE_PATHS, LOGGING_CONFIG, MODEL_CONFIGS
from ai_code_detector.core import CodeDetector

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


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
        # Ensure model_path is always a valid string, not None
        if model_path is None:
            model_path = FILE_PATHS.get("model", "")
            if not model_path:
                model_path = os.path.join("models", f"{model_type}_model.json")
        
        if embeddings_path is None:
            embeddings_path = FILE_PATHS.get("embeddings", "")
            if not embeddings_path:
                embeddings_path = os.path.join("models", "embeddings.pkl")

        # Initialize base class
        super().__init__(
            model_config=model_config,
            feature_columns=feature_columns,
            model_path=model_path,
            embeddings_path=embeddings_path
        )
    
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
        columns = {'code': str, 'language': str, 'target': str}
        if self.feature_columns:
            columns['features'] = str
            
        df = pd.read_csv(
            dataset_path, 
            usecols=columns.keys(),
            dtype=columns
        )
        
        logger.info(f"Loaded {len(df)} records from dataset")
        
        # Map target labels to binary values
        logger.info("Processing target labels...")
        df['target_binary'] = df['target'].apply(lambda x: 1 if x == 'ai' else 0)
        
        # Log class distribution
        class_counts = df['target_binary'].value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        # Apply balanced sampling if requested
        if balance_dataset:
            df = self.sample_balanced_dataset(df, total_samples)
        
        # Parse and extract features if needed
        if self.feature_columns and 'features' in df.columns:
            self._extract_features_from_column(df)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Dataset loading completed in {elapsed_time:.2f} seconds")
        
        return df
    
    def _extract_features_from_column(self, df: pd.DataFrame) -> None:
        """
        Extract features from the 'features' column and add them as separate columns.
        
        Args:
            df: DataFrame with a 'features' column containing feature dictionaries
        """
        import ast

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
        for feature in self.feature_columns:
            df[feature] = df['features'].apply(lambda x: x.get(feature, 0))
    
    def process_dataset(
        self,
        dataset_path: str,
        save_embeddings: bool = True,
        load_embeddings: bool = False,
        balance_dataset: bool = False,
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
        
        # Step 1: Load and prepare dataset
        df = self.load_dataset(
            dataset_path=dataset_path,
            balance_dataset=balance_dataset,
            total_samples=total_samples
        )
        
        # Step 2: Try to load cached embeddings if requested
        embeddings_array = None
        if load_embeddings and self.embeddings_path:
            embeddings_array = self.load_embeddings()
            if embeddings_array is not None and len(embeddings_array) == len(df):
                logger.info(f"Using {len(embeddings_array)} cached embeddings")
                df['embedding'] = list(embeddings_array)
                return df
            else:
                logger.info("Cached embeddings not available or size mismatch")
        
        else:
            # Step 3: Generate new embeddings
            logger.info(f"Generating embeddings for {len(df)} code samples...")
            batch_size = self.model_config.get("encoder", {}).get("batch_size", 16)
            embeddings_array = self.generate_embeddings(
                code_samples=df['code'].tolist(),
                languages=df['language'].tolist(),
                batch_size=batch_size
        )
        
        # Step 4: Save embeddings if requested
        if save_embeddings and self.embeddings_path:
            self.save_embeddings(embeddings_array)
            logger.info(f"Saved {len(embeddings_array)} embeddings to {self.embeddings_path}")
        
        # Step 5: Add embeddings to dataframe
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
        
        def objective(trial):
            # Define the hyperparameters to optimize based on configuration
            param = {}
            for param_name, param_range in config["param_distributions"].items():
                if param_name in ('max_depth', 'min_child_weight'):
                    param[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    param[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            
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
        Train the model using the processed dataset with cross-validation.
        
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
        
        # Tune hyperparameters if requested (using the first fold)
        if tune_params:
            train_idx, _ = next(skf.split(X, y))
            best_params = self.tune_hyperparameters(X[train_idx], y[train_idx])
            
            # Update the classifier parameters
            for param, value in best_params.items():
                self.classifier.params[param] = value
        
        # Perform k-fold cross-validation
        logger.info(f"Performing {n_folds}-fold stratified cross-validation...")
        
        # Training configuration
        training_config = self.model_config.get("training", {})
        num_boost_round = training_config.get("num_boost_round", 2000)
        early_stopping_rounds = training_config.get("early_stopping_rounds", 20)
        
        # Track metrics for each fold
        fold_metrics = []
        
        # Train and evaluate on each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            fold_num = fold + 1
            logger.info(f"Training fold {fold_num}/{n_folds}")
            
            # Split data for this fold
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model on this fold
            metrics = self.classifier.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds
            )
            
            # Record fold results
            metrics['fold'] = fold_num
            fold_metrics.append(metrics)
            logger.info(f"Fold {fold_num} results: {metrics}")
        
        # Calculate average metrics across folds
        avg_metrics = self._calculate_average_metrics(fold_metrics)
        logger.info(f"Average metrics across {n_folds} folds: {avg_metrics}")
        
        return {
            'fold_metrics': fold_metrics,
            'average_metrics': avg_metrics
        }
    
    def _calculate_average_metrics(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average metrics across all folds.
        
        Args:
            fold_metrics: List of metric dictionaries from each fold
            
        Returns:
            Dictionary with averaged metrics
        """
        if not fold_metrics:
            return {}
        
        # Initialize dictionary for average metrics
        avg_metrics = {}
        
        # Get all metric keys except 'fold'
        metric_keys = [key for key in fold_metrics[0].keys() if key != 'fold']
        
        # Calculate average for each metric
        for key in metric_keys:
            avg_metrics[key] = sum(m[key] for m in fold_metrics) / len(fold_metrics)
        
        return avg_metrics
    
    def train_final_model(
        self,
        df: pd.DataFrame,
        test_size: float = 0.1
    ) -> Dict[str, float]:
        """
        Train the final model using train/test split.
        
        Args:
            df: DataFrame with processed data
            test_size: Proportion of data to use for validation
            
        Returns:
            Dictionary with model metrics
        """
        logger.info(f"Training final model with {test_size:.0%} test split...")
        
        # Prepare features using the classifier
        X, y = self.classifier.prepare_features(df)
        
        # Create a train/test split with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")
        
        # Get training configuration
        training_config = self.model_config.get("training", {})
        
        # Train the model
        metrics = self.classifier.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_boost_round=training_config.get("num_boost_round", 2000),
            early_stopping_rounds=training_config.get("early_stopping_rounds", 50)
        )
        
        # Save the model
        if self.model_path:
            try:
                self.classifier.save_model(model_path=self.model_path, metrics=metrics)
                logger.info(f"Final model saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
        else:
            logger.warning("No model path specified, model not saved")
        
        return metrics
    
    def sample_balanced_dataset(self, df: pd.DataFrame, total_samples: int) -> pd.DataFrame:
        """
        Create a balanced sample from the dataset.
        
        Args:
            df: DataFrame to sample from
            total_samples: Total number of samples to include
            
        Returns:
            Balanced DataFrame
        """
        # Get counts by class
        class_counts = df['target_binary'].value_counts()
        logger.info(f"Original class distribution: {class_counts.to_dict()}")
        
        # Calculate samples per class
        samples_per_class = total_samples // len(class_counts)
        
        # Sample each class
        sampled_dfs = []
        for class_value, count in class_counts.items():
            class_df = df[df['target_binary'] == class_value]
            if len(class_df) > samples_per_class:
                sampled_df = class_df.sample(samples_per_class, random_state=42)
            else:
                sampled_df = class_df  # Take all samples if fewer than needed
            sampled_dfs.append(sampled_df)
        
        # Combine and shuffle
        balanced_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42)
        
        # Log new distribution
        new_counts = balanced_df['target_binary'].value_counts()
        logger.info(f"Balanced class distribution: {new_counts.to_dict()}")
        
        return balanced_df
    
    def train(
        self,
        dataset_path: str,
        tune_params: bool = True,
        n_folds: int = 5,
        balance_ratio: float = 0,
        save_embeddings: bool = True,
        load_embeddings: bool = True,
        test_size: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            dataset_path: Path to the dataset CSV file
            tune_params: Whether to perform hyperparameter tuning
            n_folds: Number of folds for cross-validation
            balance_ratio: Maximum ratio between classes (0 = no balancing)
            save_embeddings: Whether to save generated embeddings
            load_embeddings: Whether to try loading cached embeddings
            test_size: Proportion of data to use for final validation
            
        Returns:
            Dictionary with training results and metrics
        """
        start_time = time.time()
        
        # Apply balance ratio if provided
        if balance_ratio > 0:
            logger.info(f"Setting balance ratio to {balance_ratio}")
            self.model_config["training"]["balance_ratio"] = balance_ratio
        
        # Step 1: Process dataset and generate embeddings
        df = self.process_dataset(
            dataset_path=dataset_path,
            save_embeddings=save_embeddings,
            load_embeddings=load_embeddings,
            balance_dataset=False
        )
        
        # Step 2: Train with cross-validation
        logger.info("Starting model training with cross-validation...")
        cv_results = self.train_model(
            df=df,
            tune_params=tune_params,
            n_folds=n_folds
        )
        
        # Step 3: Train final model on all data with test/val split
        logger.info("Training final model...")
        final_metrics = self.train_final_model(df, test_size=test_size)
        
        # Summarize results
        total_time = time.time() - start_time
        logger.info(f"Training pipeline completed in {total_time:.2f} seconds")
        logger.info(f"Final model performance: {final_metrics}")
        
        # Return all results
        return {
            'fold_metrics': cv_results['fold_metrics'],
            'average_metrics': cv_results['average_metrics'],
            'final_model_metrics': final_metrics,
            'training_time_seconds': total_time
        }

def main():
    """Main function to run the training pipeline."""
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Train the AI-generated code detection model')
    
    # Dataset and model configuration
    parser.add_argument('--dataset', type=str, 
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv'), 
                        help='Path to the dataset CSV file')
    parser.add_argument('--model-type', type=str, 
                        default='xgboost', 
                        help='Type of model to train (xgboost or unixcoder)')
    
    # Training options
    parser.add_argument('--tune-params', action='store_true', 
                        help='Perform hyperparameter tuning')
    parser.add_argument('--n-folds', type=int, default=5, 
                        help='Number of folds for cross-validation')
    parser.add_argument('--balance-ratio', type=float, default=0, 
                        help='Maximum ratio between classes (0=no balancing)')
    parser.add_argument('--test-size', type=float, default=0.1,
                        help='Proportion of data to use for final validation (default: 10%)')
    
    # Embedding options
    parser.add_argument('--save-embeddings', action='store_true',
                        help='Save embeddings to cache')
    parser.add_argument('--load-embeddings', action='store_true',
                        help='Load embeddings from cache')
    
    args = parser.parse_args()
    
    # Initialize and run training pipeline
    pipeline = TrainingPipeline(model_type=args.model_type)
    
    pipeline.train(
        dataset_path=args.dataset,
        tune_params=args.tune_params,
        n_folds=args.n_folds,
        balance_ratio=args.balance_ratio,
        save_embeddings=not args.save_embeddings,
        load_existing_embeddings=not args.load_embeddings,
        test_size=args.test_size
    )

if __name__ == "__main__":
    main() 