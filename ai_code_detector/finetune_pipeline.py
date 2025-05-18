#!/usr/bin/env python3
"""
Fine-tuning pipeline for the UnixCoder classifier model.

This script provides a complete pipeline for fine-tuning the UnixCoder
model for detecting AI-generated code:
1. Data loading and preparation
2. Model initialization
3. Fine-tuning
4. Evaluation
5. Model saving
"""

import argparse
import logging
import os
import time
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from ai_code_detector.config import FILE_PATHS, LOGGING_CONFIG
from ai_code_detector.models.unixcoder_classifier import CodeDataset, UnixCoderClassifier, UnixCoderClassifierTrainer

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class FineTuningPipeline:
    """
    Pipeline for fine-tuning the UnixCoder classifier model.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        max_length: int = 1024,
        cache_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the fine-tuning pipeline.
        
        Args:
            model_name: Name or path of the UnixCoder model
            max_length: Maximum sequence length for tokenization
            cache_dir: Optional directory to cache the downloaded model
            output_dir: Directory to save trained models
        """
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.output_dir = output_dir or FILE_PATHS.get("model_dir", "./models")
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    def load_data(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        balance_dataset: bool = False,
        max_samples: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare the dataset.
        
        Args:
            dataset_path: Path to the dataset CSV file
            test_size: Fraction of data to use for testing
            val_size: Fraction of data to use for validation
            balance_dataset: Whether to balance the dataset
            max_samples: Maximum number of samples to use (for debugging/testing)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        start_time = time.time()
        logger.info(f"Loading dataset from {dataset_path}...")
        
        # Read only necessary columns
        df = pd.read_csv(
            dataset_path,
            usecols=['code', 'language', 'target'],
            dtype={'code': str, 'language': str, 'target': str}
        )
        
        logger.info(f"Loaded {len(df)} records from dataset")
        
        # Create binary labels
        df['label'] = df['target'].apply(lambda x: 1 if x == 'ai' else 0)
        
        # Log class distribution
        class_counts = df['label'].value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        # Balance the dataset if requested
        if balance_dataset:
            logger.info("Balancing dataset...")
            # Get counts
            class_0 = df[df['label'] == 0]
            class_1 = df[df['label'] == 1]
            
            # Get the minimum count (or max_samples/2 if specified)
            min_count = min(len(class_0), len(class_1))
            if max_samples and max_samples < min_count * 2:
                min_count = max_samples // 2
            
            # Sample from each class
            class_0 = class_0.sample(min_count, random_state=42)
            class_1 = class_1.sample(min_count, random_state=42)
            
            # Combine and shuffle
            df = pd.concat([class_0, class_1]).sample(frac=1, random_state=42)
            
            logger.info(f"Balanced dataset size: {len(df)}")
        elif max_samples and len(df) > max_samples:
            # If not balancing but have max_samples, take a stratified sample
            df = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(int(max_samples * len(x) / len(df)), random_state=42)
            )
            
            logger.info(f"Sampled dataset size: {len(df)}")
        
        # Split into train, validation, and test sets
        train_df, temp_df = train_test_split(
            df, test_size=test_size + val_size, stratify=df['label'], random_state=42
        )
        
        # Calculate relative sizes for val and test
        relative_val_size = val_size / (test_size + val_size)
        
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - relative_val_size), stratify=temp_df['label'], random_state=42
        )
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Log time taken
        elapsed_time = time.time() - start_time
        logger.info(f"Data loading completed in {elapsed_time:.2f} seconds")
        
        return train_df, val_df, test_df
    
    def create_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[CodeDataset, CodeDataset, CodeDataset]:
        """
        Create PyTorch datasets from DataFrames.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Creating PyTorch datasets...")
        
        # Create training dataset
        train_dataset = CodeDataset(
            code_samples=train_df['code'].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            languages=train_df['language'].tolist()
        )
        
        # Create validation dataset
        val_dataset = CodeDataset(
            code_samples=val_df['code'].tolist(),
            labels=val_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            languages=val_df['language'].tolist()
        )
        
        # Create test dataset
        test_dataset = CodeDataset(
            code_samples=test_df['code'].tolist(),
            labels=test_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            languages=test_df['language'].tolist()
        )
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, "
                    f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def finetune(
        self,
        train_dataset: CodeDataset,
        val_dataset: CodeDataset,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        eval_steps: int = 100,
        run_name: Optional[str] = None,
        model_dir: Optional[str] = None
    ) -> Tuple[UnixCoderClassifierTrainer, Dict]:
        """
        Fine-tune the UnixCoder classifier model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            warmup_steps: Warmup steps for learning rate scheduler
            weight_decay: Weight decay for regularization
            eval_steps: Number of steps between evaluations
            run_name: Optional name for this training run
            model_dir: Optional path to save the model (overrides automatic directory creation)
            
        Returns:
            Tuple of (trainer, metrics)
        """
        # Create model output directory if not provided
        if not model_dir:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            run_id = run_name or f"unixcoder-classifier-{timestamp}"
            model_dir = os.path.join(self.output_dir, run_id)
            os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Fine-tuning model, saving to {model_dir}...")
        
        # Initialize the classifier model
        model = UnixCoderClassifier(
            model_name=self.model_name,
            num_classes=2,
            dropout_rate=0.1,
            cache_dir=self.cache_dir
        )
        
        # Initialize the trainer
        trainer = UnixCoderClassifierTrainer(
            model=model,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Train the model
        logger.info(f"Starting training for {num_epochs} epochs with batch size {batch_size}...")
        metrics = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            eval_steps=eval_steps,
            save_dir=model_dir
        )
        
        return trainer, metrics
    
    def evaluate(
        self,
        trainer: UnixCoderClassifierTrainer,
        test_dataset: CodeDataset,
        batch_size: int = 16
    ) -> Dict[str, float]:
        """
        Evaluate the fine-tuned model on the test set.
        
        Args:
            trainer: The model trainer
            test_dataset: Test dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Run evaluation
        test_metrics = trainer.evaluate(test_dataset, batch_size=batch_size)
        
        # Log results
        logger.info(f"Test Results:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        logger.info(f"  AUC: {test_metrics['auc']:.4f}")
        
        return test_metrics
    
    def run_pipeline(
        self,
        dataset_path: str,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        max_samples: Optional[int] = None,
        run_name: Optional[str] = None
    ) -> Tuple[UnixCoderClassifierTrainer, Dict, Dict]:
        """
        Run the complete fine-tuning pipeline.
        
        Args:
            dataset_path: Path to the dataset CSV file
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            warmup_steps: Warmup steps for learning rate scheduler
            weight_decay: Weight decay for regularization
            max_samples: Maximum number of samples to use
            run_name: Optional name for this training run
            
        Returns:
            Tuple of (trainer, training_metrics, test_metrics)
        """
        logger.info("Starting UnixCoder classifier fine-tuning pipeline...")
        start_time = time.time()
        
        # Load and prepare data
        train_df, val_df, test_df = self.load_data(
            dataset_path=dataset_path,
            max_samples=max_samples
        )
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self.create_datasets(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df
        )
        
        # Create model output directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = run_name or f"unixcoder-classifier-{timestamp}"
        model_dir = os.path.join(self.output_dir, run_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Fine-tune model
        trainer, train_metrics = self.finetune(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            model_dir=model_dir
        )
        
        # Evaluate on test set
        test_metrics = self.evaluate(
            trainer=trainer,
            test_dataset=test_dataset,
            batch_size=batch_size
        )
        
        # Save metrics to the final model directory using the trainer's save_model method
        final_model_dir = os.path.join(model_dir, "final")
        if os.path.exists(final_model_dir):
            trainer.save_model(
                final_model_dir, 
                train_metrics=self._extract_final_metrics(train_metrics) if train_metrics else None, 
                test_metrics=test_metrics
            )
        
        # Log total time
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        
        return trainer, train_metrics, test_metrics

    def _extract_final_metrics(self, metrics_dict: Dict) -> Dict[str, float]:
        """
        Extract the most recent/final values from training metrics.
        
        Args:
            metrics_dict: Dictionary with metrics lists
            
        Returns:
            Dictionary with final metric values
        """
        final_metrics = {}
        for key, value in metrics_dict.items():
            if isinstance(value, list) and value:
                final_metrics[key] = value[-1]  # Take the last value
            else:
                final_metrics[key] = value
        return final_metrics


def main():
    """
    Main function to run the fine-tuning pipeline from command line.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune UnixCoder for AI-generated code detection"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_head.csv'),
        help="Path to the dataset CSV file"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/unixcoder-base",
        help="Name or path of the UnixCoder model to use"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the fine-tuned model"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache the pretrained model"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimizer"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup steps for learning rate scheduler"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for debugging)"
    )
    
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this training run"
    )
    
    args = parser.parse_args()
    
    # Initialize and run the pipeline
    pipeline = FineTuningPipeline(
        model_name=args.model_name,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir
    )
    
    pipeline.run_pipeline(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_samples=args.max_samples,
        run_name=args.run_name
    )


if __name__ == "__main__":
    main() 