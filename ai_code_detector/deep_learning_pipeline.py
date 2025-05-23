#!/usr/bin/env python3
"""
Unified Deep Learning Pipeline for AI Code Detection.

This script provides a complete pipeline for training deep learning models
for detecting AI-generated code, supporting both UnixCoder and generic
embedding-based classifiers.
"""

import argparse
import logging
import os
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ai_code_detector.config import EMBEDDINGS_DIR, LOGGING_CONFIG, MODELS_DIR, PROJECT_ROOT
from ai_code_detector.models.code_embedder import CodeEmbeddingEncoder
from ai_code_detector.models.embedding_classifier import EmbeddingClassifier, EmbeddingDataset
from ai_code_detector.models.unixcoder import UnixCoderDataset, UnixCoderModel
from ai_code_detector.trainer import Trainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class DeepLearningPipeline:
    """Unified pipeline for training deep learning models."""
    
    def __init__(
        self,
        model_name: str = "unixcoder"
    ):
        """
        Initialize the deep learning pipeline.
        
        Args:
            model_name: Name of the base model to use
        """
        self.model_name = model_name
        
        # Initialize encoder
        self.encoder = CodeEmbeddingEncoder(
            cache_dir=EMBEDDINGS_DIR
        )
        
        logger.info(f"Initialized {model_name} pipeline with model: {model_name}")
        logger.info(f"Encoder embedding dimension: {self.encoder.embedding_dim}")
    
    def load_data(
        self,
        dataset_path: str,
        test_size: float = 0.1,
        val_size: float = 0.1,
        need_embeddings: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and prepare the dataset."""
        logger.info(f"Loading dataset from {dataset_path}...")
        
        # Read only necessary columns
        df = pd.read_csv(
            dataset_path,
            usecols=['code', 'language', 'target'],
            dtype={'code': str, 'language': str, 'target': str}
        )
        
        # Clean the data - remove rows with missing code
        initial_size = len(df)
        df = df.dropna(subset=['code'])  # Remove rows where code is NaN
        df = df[df['code'].str.strip() != '']  # Remove rows where code is empty string
        cleaned_size = len(df)
        
        if initial_size != cleaned_size:
            logger.info(f"Removed {initial_size - cleaned_size} rows with missing/empty code. "
                       f"Dataset size: {initial_size} -> {cleaned_size}")
        
        # Create binary labels
        df['label'] = df['target'].apply(lambda x: 1 if x == 'ai' else 0)
        
        # Log class distribution
        class_counts = df['label'].value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        # Handle embeddings - either load from cache or generate new ones
        embeddings_loaded = False
        
        embeddings_path = os.path.join(EMBEDDINGS_DIR, self.model_name)
        # Try to load cached embeddings first
        if need_embeddings and os.path.exists(embeddings_path):
            try:
                logger.info(f"Loading cached embeddings from {embeddings_path}")
                embedding_data = CodeEmbeddingEncoder.load_embeddings(embeddings_path)
                cached_embeddings = embedding_data['embeddings']
                
                # Verify the cached embeddings match our dataset size
                if len(cached_embeddings) == len(df):
                    df["embedding"] = cached_embeddings.tolist()
                    embeddings_loaded = True
                    logger.info(f"Successfully loaded {len(cached_embeddings)} cached embeddings")
                else:
                    logger.warning(f"Cached embeddings size ({len(cached_embeddings)}) doesn't match dataset size ({len(df)})")
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
        
        # Generate embeddings if not loaded from cache
        if need_embeddings and not embeddings_loaded:
            logger.info("Generating embeddings for the dataset...")
            embeddings = self.encoder.batch_encode(
                texts=df['code'].tolist(),
                batch_size=8,
                show_progress=True
            )
            df["embedding"] = embeddings.tolist()
            
            logger.info(f"Saving embeddings to cache: {embeddings_path}")
            self.encoder.save_embeddings(
                embeddings=embeddings,
                file_path=embeddings_path,
                metadata={
                    'dataset_info': 'Training dataset embeddings',
                    'num_samples': len(df)
                }
            )
        # Split into train, validation, and test sets
        train_df, temp_df = train_test_split(
            df, test_size=test_size + val_size, stratify=df['label'], random_state=42
        )
        
        # Calculate relative sizes for val and test
        relative_val_size = val_size / (test_size + val_size)
        
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - relative_val_size), stratify=temp_df['label'], random_state=42
        )
        
        logger.info(f"Split sizes - Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
                   f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
                   f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def create_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[Union[UnixCoderDataset, EmbeddingDataset], Union[UnixCoderDataset, EmbeddingDataset], Union[UnixCoderDataset, EmbeddingDataset]]:
        """Create PyTorch datasets for training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        train_labels = train_df['label'].tolist()
        val_labels = val_df['label'].tolist()
        test_labels = test_df['label'].tolist()
        
        if self.model_name == "unixcoder":
            train_texts = train_df['code'].tolist()
            val_texts = val_df['code'].tolist()
            test_texts = test_df['code'].tolist()
            train_dataset = UnixCoderDataset(texts=train_texts, labels=train_labels, device=device)
            val_dataset = UnixCoderDataset(texts=val_texts, labels=val_labels, device=device)
            test_dataset = UnixCoderDataset(texts=test_texts, labels=test_labels, device=device)
        else:  # embedding classifier
            train_embeddings = [np.array(emb) for emb in train_df['embedding']]
            val_embeddings = [np.array(emb) for emb in val_df['embedding']]
            test_embeddings = [np.array(emb) for emb in test_df['embedding']]
            train_dataset = EmbeddingDataset(train_embeddings, train_labels)
            val_dataset = EmbeddingDataset(val_embeddings, val_labels)
            test_dataset = EmbeddingDataset(test_embeddings, test_labels)
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def run_pipeline(
        self,
        dataset_path: str,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        run_name: Optional[str] = None,
        need_embeddings: bool = False
    ) -> Tuple[Trainer, Dict, Dict]:
        """Run the complete training pipeline."""
        logger.info(f"Starting {self.model_name} deep learning pipeline...")
        start_time = time.time()
        
        # Create model output directory
        run_id = run_name or f"{self.model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        model_dir = os.path.join(MODELS_DIR, run_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. Load and prepare data
        train_df, val_df, test_df = self.load_data(
            dataset_path=dataset_path,
            need_embeddings=need_embeddings
        )
    
        
        # 2. Create datasets
        train_dataset, val_dataset, test_dataset = self.create_datasets(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df
        )
        
        # 3. Initialize model and trainer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model_name == "unixcoder":
            model = UnixCoderModel()

        else:  # embedding classifier
            # Use hidden layers if provided, otherwise default architecture
            model = EmbeddingClassifier(
                embedding_dim=self.encoder.embedding_dim
            )
        
        trainer = Trainer(model=model, device=device)
        
        logger.info(f"Initialized {self.model_name} model and trainer on {device}")
        
        # 4. Train the model
        logger.info("Starting model training...")
        train_metrics = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            weight_decay=weight_decay,
            save_dir=model_dir
        )
        
        # 5. Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_dataset, batch_size)
        
        # Log final results
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Final test metrics: {test_metrics}")
        
        return trainer, train_metrics, test_metrics


def main():
    """Main function to run the deep learning pipeline."""
    parser = argparse.ArgumentParser(description='Train deep learning models for AI code detection')
    parser.add_argument('--dataset', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'dataset_val.csv'),
                        help='Path to the training dataset CSV file')
    parser.add_argument('--model-name', type=str, default='unixcoder',
                        help='Name of the base model to use')
    parser.add_argument('--output-dir', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate for training')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--run-name', type=str,
                        help='Name for this training run')
    
    args = parser.parse_args()
    
    # Initialize and run the pipeline
    pipeline = DeepLearningPipeline(
        model_name=args.model_name
    )
    
    try:
        trainer, train_metrics, test_metrics = pipeline.run_pipeline(
            dataset_path=args.dataset,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            run_name=args.run_name
        )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Train metrics: {train_metrics}")
        logger.info(f"Test metrics: {test_metrics}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 
