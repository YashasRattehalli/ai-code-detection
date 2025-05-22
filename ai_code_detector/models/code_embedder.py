"""
Generic embedding encoder for any Hugging Face model.

This module provides a universal embedding encoder that can work with any
Hugging Face transformer model and includes functionality to save/load embeddings.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Set up logging
logger = logging.getLogger(__name__)


class CodeEmbeddingEncoder:
    """
    Universal embedding encoder that works with any Hugging Face transformer model.
    
    This encoder can generate embeddings for source code using any pre-trained
    transformer model from Hugging Face and provides utilities to save/load embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        max_length: int = 1024,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        pooling_strategy: str = "cls",
        **kwargs
    ):
        """
        Initialize the encoder with the specified model.
        
        Args:
            model_name: Name or path of the Hugging Face model
            max_length: Maximum sequence length for tokenization
            cache_dir: Optional directory to cache the downloaded model
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            pooling_strategy: How to pool embeddings ('cls', 'mean', 'max')
            **kwargs: Additional parameters for model loading
        """
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.pooling_strategy = pooling_strategy
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                **kwargs
            )
            self.model = AutoModel.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                **kwargs
            ).to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
    
    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings."""
        return self.model.config.hidden_size
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text/code sample into embeddings.
        
        Args:
            text: Text/code to encode
            
        Returns:
            Numpy array containing the embedding
        """
        # Tokenize the text
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Move tensors to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Apply pooling strategy
            if self.pooling_strategy == "cls":
                # Use CLS token (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif self.pooling_strategy == "mean":
                # Mean pooling over all tokens (considering attention mask)
                hidden_states = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            elif self.pooling_strategy == "max":
                # Max pooling over all tokens
                hidden_states = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                hidden_states[mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                embeddings = torch.max(hidden_states, 1)[0].cpu().numpy()
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return embeddings[0]  # Return single embedding array
    
    def batch_encode(
        self, 
        texts: List[str], 
        batch_size: int = 8,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode multiple text/code samples in batches for efficiency.
        
        Args:
            texts: List of text/code samples to encode
            batch_size: Number of samples to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array containing all embeddings (shape: [num_samples, embedding_dim])
        """
        total_samples = len(texts)
        all_embeddings = []
        
        # Create progress bar if requested
        iterator = range(0, total_samples, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
        
        for i in iterator:
            end_idx = min(i + batch_size, total_samples)
            batch_texts = texts[i:end_idx]
            
            # Tokenize the entire batch
            encoded = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Move tensors to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Apply pooling strategy
                batch_embeddings = []
                if self.pooling_strategy == "cls":
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif self.pooling_strategy == "mean":
                    hidden_states = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                elif self.pooling_strategy == "max":
                    hidden_states = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    hidden_states[mask_expanded == 0] = -1e9
                    batch_embeddings = torch.max(hidden_states, 1)[0].cpu().numpy()
                
                all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        return np.vstack(all_embeddings)
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        file_path: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save embeddings to a file with optional metadata.
        
        Args:
            embeddings: Numpy array of embeddings to save
            file_path: Path where to save the embeddings
            metadata: Optional metadata to save alongside embeddings
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Prepare data to save
        data = {
            'embeddings': embeddings,
            'metadata': metadata or {},
            'model_info': {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'max_length': self.max_length,
                'pooling_strategy': self.pooling_strategy
            }
        }
        
        # Save to pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {embeddings.shape[0]} embeddings to {file_path}")
    
    @staticmethod
    def load_embeddings(file_path: str) -> Dict:
        """
        Load embeddings from a file.
        
        Args:
            file_path: Path to the embeddings file
            
        Returns:
            Dictionary containing embeddings, metadata, and model info
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embeddings file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded {data['embeddings'].shape[0]} embeddings from {file_path}")
        return data
    
    def encode_and_save(
        self,
        texts: List[str],
        save_path: str,
        batch_size: int = 8,
        metadata: Optional[Dict] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts and save embeddings to file in one step.
        
        Args:
            texts: List of text/code samples to encode
            save_path: Path where to save the embeddings
            batch_size: Number of samples to process at once
            metadata: Optional metadata to save alongside embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array containing all embeddings
        """
        # Generate embeddings
        embeddings = self.batch_encode(
            texts=texts,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        # Add additional metadata
        full_metadata = metadata or {}
        full_metadata.update({
            'num_samples': len(texts),
            'batch_size': batch_size,
        })
        
        # Save embeddings
        self.save_embeddings(embeddings, save_path, full_metadata)
        
        return embeddings
    
    def __repr__(self) -> str:
        return (f"GenericEmbeddingEncoder(model_name='{self.model_name}', "
                f"embedding_dim={self.embedding_dim}, "
                f"pooling_strategy='{self.pooling_strategy}', "
                f"device='{self.device}')") 