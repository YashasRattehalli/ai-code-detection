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
import pickle
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Import project modules
from config import FEATURE_COLUMNS, FILE_PATHS, LOGGING_CONFIG, MODEL_CONFIGS
from models.xgboost_model import UnixCoderEncoder, XGBoostClassifier

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

def process_dataset(
    dataset_path: str, 
    encoder: UnixCoderEncoder, 
    feature_columns: List[str],
    embeddings_path: str,
    save_embeddings: bool = True, 
    load_embeddings: bool = True,  # New parameter to control loading
    batch_size: int = 16,
    balance_ratio: float = 0  # 0 means no balancing, >0 defines max ratio
) -> pd.DataFrame:
    """
    Process the dataset and create embeddings with optimized memory usage.
    
    Args:
        dataset_path: Path to the dataset CSV file
        encoder: UnixCoder encoder instance
        feature_columns: List of feature column names to extract
        embeddings_path: Path to save/load embeddings
        save_embeddings: Whether to save embeddings to file
        load_embeddings: Whether to try loading pre-computed embeddings
        batch_size: Number of samples to process at once
        balance_ratio: Maximum ratio for balancing the dataset
        
    Returns:
        DataFrame with processed data including embeddings
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
    
    # No longer sampling a subset - using all data
    logger.info(f"Using all {len(df)} rows from dataset")
    
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
    for feature in feature_columns:
        df[feature] = df['features'].apply(lambda x: x.get(feature, 0))
    
    # Check if we can load from cache
    if load_embeddings and os.path.exists(embeddings_path):
        logger.info(f"Loading pre-computed embeddings from {embeddings_path}...")
        try:
            with open(embeddings_path, 'rb') as f:
                embeddings_array = pickle.load(f)
                
            if len(embeddings_array) == len(df):
                logger.info("Using cached embeddings")
                df['embedding'] = list(embeddings_array)
                return df
            else:
                logger.warning("Cached embeddings don't match dataset size. Regenerating...")
        except Exception as e:
            logger.warning(f"Error loading cached embeddings: {e}. Regenerating...")
    
    # Generate embeddings in optimized batches
    logger.info("Generating code embeddings...")
    codes = df['code'].tolist()
    langs = df['language'].tolist()
    
    # More efficient batch encoding
    embeddings_array = encoder.batch_encode(codes, langs, batch_size=batch_size)
    
    # Save embeddings to file if requested
    if save_embeddings:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        
        logger.info(f"Saving embeddings to {embeddings_path}...")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings_array, f)
    
    # Store embeddings in dataframe
    df['embedding'] = list(embeddings_array)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Dataset processing completed in {elapsed_time:.2f} seconds")
    
    return df

def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tune hyperparameters for the XGBoost model using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary for hyperparameter tuning
        
    Returns:
        Dictionary with best parameters
    """
    logger.info("Tuning hyperparameters with Optuna...")
    
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

def train_pipeline(
    dataset_path: str, 
    model_config: Dict[str, Any],
    feature_columns: List[str],
    embeddings_path: str,
    model_path: str,
    importance_path: str,
    load_embeddings: bool = True,
    tune_params: bool = True,
    n_folds: int = 5
) -> None:
    """
    Execute the complete training pipeline with optimizations.
    
    Args:
        dataset_path: Path to the dataset CSV file
        model_config: Model configuration dictionary
        feature_columns: Feature columns to use
        embeddings_path: Path to save/load embeddings
        model_path: Path to save model
        importance_path: Path to save feature importance
        load_embeddings: Whether to load pre-computed embeddings
        tune_params: Whether to perform hyperparameter tuning
        n_folds: Number of folds for stratified k-fold cross-validation
    """
    start_time = time.time()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Initialize the UnixCoder encoder from config
    encoder_config = model_config["encoder"]
    encoder = UnixCoderEncoder(
        model_name=encoder_config["model_name"],
        max_length=encoder_config["max_length"],
        language_prefixes=encoder_config["language_prefixes"]
    )
    
    # Process dataset and get embeddings
    df = process_dataset(
        dataset_path=dataset_path,
        encoder=encoder,
        feature_columns=feature_columns,
        embeddings_path=embeddings_path,
        save_embeddings=True,  # Always save embeddings when they're computed
        load_embeddings=load_embeddings,
        batch_size=model_config["training"]["batch_size"],
        balance_ratio=model_config["training"]["balance_ratio"]
    )
    
    # Initialize classifier with parameters from config
    classifier = XGBoostClassifier(
        params=model_config["params"],
        feature_columns=feature_columns
    )
    
    # Prepare features
    X, y = classifier.prepare_features(df)
    
    # Set up stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize metrics tracking
    all_metrics = []
    
    # Tune hyperparameters if requested (using the first fold)
    if tune_params:
        first_fold = next(skf.split(X, y))
        train_idx, val_idx = first_fold
        best_params = tune_hyperparameters(
            X[train_idx], y[train_idx], 
            model_config["hyperparameter_tuning"]
        )
        # Update the classifier parameters
        for param, value in best_params.items():
            classifier.params[param] = value
    
    # Perform k-fold cross-validation
    logger.info(f"Performing {n_folds}-fold stratified cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Training fold {fold+1}/{n_folds}")
        
        # Get train and validation data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train the model for this fold
        training_config = model_config["training"]
        fold_metrics = classifier.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_boost_round=training_config["num_boost_round"],
            early_stopping_rounds=training_config["early_stopping_rounds"]
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
    
    # Train final model on all data
    logger.info("Training final model on all data...")
    final_metrics = classifier.train(
        X_train=X,
        y_train=y,
        X_val=None,
        y_val=None,
        num_boost_round=training_config["num_boost_round"],
        early_stopping_rounds=None  # No early stopping for final model
    )
    
    # Save the final model
    classifier.save_model(model_path, importance_path)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Training pipeline completed successfully in {elapsed_time:.2f} seconds!")
    
    # Return metrics for potential logging/reporting
    return {
        'fold_metrics': all_metrics,
        'average_metrics': avg_metrics,
        'final_model_metrics': final_metrics
    }

def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='Train the AI-generated code detection model')
    parser.add_argument('--dataset', type=str, 
                        default='/Users/kantharaju/Projects/Personal/CodeDetector/data/dataset.csv', 
                        help='Path to the dataset CSV file')
    parser.add_argument('--model-type', type=str, default='xgboost', help='Type of model to train')
    parser.add_argument('--load-embeddings', action='store_true', help='Load pre-computed embeddings if available')
    parser.add_argument('--tune-params', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--balance-ratio', type=float, default=0, 
                        help='Maximum ratio between classes (0 = no balancing, 5 = at most 5:1 ratio)')
    parser.add_argument('--n-folds', type=int, default=5, 
                        help='Number of folds for stratified k-fold cross-validation')
    
    args = parser.parse_args()
    
    # Check if model type is supported
    if args.model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {args.model_type}. Available options: {list(MODEL_CONFIGS.keys())}")
    
    # Get configuration for the specified model
    model_config = MODEL_CONFIGS[args.model_type]
    
    # Override balance ratio if provided
    if args.balance_ratio > 0:
        model_config["training"]["balance_ratio"] = args.balance_ratio
    
    # Run the training pipeline
    train_pipeline(
        dataset_path=args.dataset,
        model_config=model_config,
        feature_columns=FEATURE_COLUMNS,
        embeddings_path=FILE_PATHS["embeddings"],
        model_path=FILE_PATHS["model"],
        importance_path=FILE_PATHS["feature_importance"],
        load_embeddings=args.load_embeddings,
        tune_params=args.tune_params,
        n_folds=args.n_folds
    )

if __name__ == "__main__":
    main() 