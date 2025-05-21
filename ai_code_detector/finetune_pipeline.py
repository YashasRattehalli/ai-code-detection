#!/usr/bin/env python3
"""
Fine-tuning pipeline for the UnixCoder classifier model.

This script provides a complete pipeline for fine-tuning the UnixCoder
model for detecting AI-generated code.
"""


import argparse
import logging
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from ai_code_detector.config import FILE_PATHS, LOGGING_CONFIG
from ai_code_detector.models.unixcoder_classifier import CodeDataset, UnixCoderClassifier, UnixCoderClassifierTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class FineTuningPipeline:
    """Pipeline for fine-tuning the UnixCoder classifier model."""
    
    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        max_length: int = 1024,
        cache_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """Initialize the fine-tuning pipeline."""
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.output_dir = output_dir or FILE_PATHS.get("model_dir", "./models")
        
        logger.info(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    def load_data(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        balance: bool = False,
        max_samples: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and prepare the dataset."""
        logger.info(f"Loading dataset from {dataset_path}...")
        
        # Read only necessary columns
        df = pd.read_csv(
            dataset_path,
            usecols=['code', 'language', 'target'],
            dtype={'code': str, 'language': str, 'target': str}
        )
        
        # Create binary labels
        df['label'] = df['target'].apply(lambda x: 1 if x == 'ai' else 0)
        
        # Log class distribution
        class_counts = df['label'].value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        if self.model_name == "microsoft/unixcoder-base":
            df["embedding"] = self.load_embeddings(FILE_PATHS.get("embeddings"))
        # Balance the dataset if requested
        if balance:
            class_0 = df[df['label'] == 0]
            class_1 = df[df['label'] == 1]
            
            min_count = min(len(class_0), len(class_1))
            if max_samples and max_samples < min_count * 2:
                min_count = max_samples // 2
            
            df = pd.concat([
                class_0.sample(min_count, random_state=42),
                class_1.sample(min_count, random_state=42)
            ]).sample(frac=1, random_state=42)
            
            logger.info(f"Balanced dataset size: {len(df)}")
        elif max_samples and len(df) > max_samples:
            # Take a stratified sample
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
        return train_df, val_df, test_df
    
    def load_embeddings(self, embeddings_path: Optional[str] = None) -> Optional[List[np.ndarray]]:
        """
        Load pre-computed embeddings from a file.
        
        Args:
            embeddings_path: Path to the embeddings file
            
        Returns:
            Array of embeddings or None if loading fails
        """
        if not embeddings_path:
            embeddings_path = FILE_PATHS.get("embeddings")
            
        if not embeddings_path:
            logger.warning("No embeddings path provided or found in FILE_PATHS")
            return None
            
        try:
            if os.path.exists(embeddings_path):
                logger.info(f"Loading pre-computed embeddings from {embeddings_path}")
                with open(embeddings_path, 'rb') as f:
                    embeddings = pickle.load(f)
                    # Return full embeddings in production, not just first 100
                    return list(embeddings)
            else:
                logger.warning(f"Embeddings file not found at {embeddings_path}")
        except Exception as e:
            logger.warning(f"Error loading embeddings: {str(e)}")
            
        return None
    
    def create_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[CodeDataset, CodeDataset, CodeDataset]:
        """Create PyTorch datasets from DataFrames."""
        logger.info("Creating PyTorch datasets...")
        
        # Verify embeddings exist
        if 'embedding' not in train_df.columns or train_df['embedding'].isnull().any():
            raise ValueError("Embeddings are missing in the training dataframe")
        
        # Create datasets
        train_dataset = CodeDataset(
            embeddings=train_df['embedding'].tolist(),
            labels=train_df['label'].tolist()
        )
        
        val_dataset = CodeDataset(
            embeddings=val_df['embedding'].tolist(),
            labels=val_df['label'].tolist()
        )
        
        test_dataset = CodeDataset(
            embeddings=test_df['embedding'].tolist(),
            labels=test_df['label'].tolist()
        )
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset
    
    def run_pipeline(
        self,
        dataset_path: str,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        max_samples: Optional[int] = None,
        run_name: Optional[str] = None,
        balance_dataset: bool = False,
    ) -> Tuple[UnixCoderClassifierTrainer, Dict, Dict]:
        """Run the complete fine-tuning pipeline."""
        logger.info("Starting UnixCoder classifier fine-tuning pipeline...")
        start_time = time.time()
        
        # Create model output directory
        run_id = run_name or "unixcoder-classifier"
        model_dir = os.path.join(self.output_dir, run_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. Load and prepare data
        train_df, val_df, test_df = self.load_data(
            dataset_path=dataset_path,
            max_samples=max_samples,
            balance=balance_dataset
        )
        
        # Check if embeddings are loaded
        if 'embedding' not in train_df.columns or train_df['embedding'].isnull().any():
            raise ValueError("Embeddings are required for this model but are missing in the dataset")
        
        # Get embedding dimension from first sample
        embedding_dim = train_df['embedding'].iloc[0].shape[0]
        logger.info(f"Using embeddings with dimension: {embedding_dim}")
        
        # 2. Create datasets
        train_dataset, val_dataset, test_dataset = self.create_datasets(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df
        )
        
        # 3. Initialize model and trainer
        model = UnixCoderClassifier(
            embedding_dim=embedding_dim,
            num_classes=2,
            dropout_rate=0.1
        )
        
        trainer = UnixCoderClassifierTrainer(
            model=model
        )
        
        # 4. Fine-tune model
        logger.info(f"Fine-tuning model, saving to {model_dir}...")
        train_metrics = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            weight_decay=weight_decay,
            save_dir=model_dir
        )
        
        # 5. Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_metrics = trainer.evaluate(test_dataset, batch_size=batch_size)
        
        # Log results
        logger.info(f"Test Results: Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        # Log total time
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        
        return trainer, train_metrics, test_metrics


def main():
    """Main function to run the fine-tuning pipeline from command line."""
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
    
    parser.add_argument(
        "--balance_dataset",
        action="store_true",
        help="Whether to balance the dataset"
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
        weight_decay=args.weight_decay,
        max_samples=args.max_samples,
        run_name=args.run_name,
        balance_dataset=args.balance_dataset
    )


if __name__ == "__main__":
    main() 
