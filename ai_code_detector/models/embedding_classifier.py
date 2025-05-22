import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Set up logging
logger = logging.getLogger(__name__)

class EmbeddingDataset(Dataset):
    """
    Dataset for embedding-based classification tasks.
    Works with any type of precomputed embeddings.
    """
    def __init__(
        self,
        embeddings: List[np.ndarray],
        labels: Optional[List[int]] = None
    ):
        """
        Initialize dataset with embeddings and labels.
        
        Args:
            embeddings: List of precomputed embeddings
            labels: Optional list of labels (1 for AI, 0 for human)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-convert embeddings to tensors and move to device for efficiency
        self.embeddings = torch.stack([
            torch.tensor(emb, dtype=torch.float32) for emb in embeddings
        ]).to(self.device)
        
        self.labels: Optional[torch.Tensor] = None
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long).to(self.device)

        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        item = {
            "embedding": self.embeddings[idx]
        }
        # Add label if available
        if self.labels is not None:
            item['labels'] = self.labels[idx]
            
        return item


class EmbeddingClassifier(nn.Module):
    """
    Generic classifier model that takes precomputed embeddings as input.
    Can work with embeddings from any encoder (UnixCoder, CodeBERT, etc.).
    """
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        hidden_layers: Optional[List[int]] = None
    ):
        """
        Initialize the classifier model.
        Args:
            embedding_dim: Dimension of the input embeddings
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout probability for regularization
            hidden_layers: List of hidden layer sizes (if None, uses single layer)
        """
        super(EmbeddingClassifier, self).__init__()
        
        if hidden_layers is None:
            # Default single hidden layer architecture
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim // 2, num_classes)
            )
        else:
            # Custom architecture with specified hidden layers
            layers = []
            input_dim = embedding_dim
            
            for hidden_dim in hidden_layers:
                layers.extend([
                    nn.Dropout(dropout_rate),
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU()
                ])
                input_dim = hidden_dim
            
            # Final output layer
            layers.extend([
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim, num_classes)
            ])
            
            self.classifier = nn.Sequential(*layers)
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            embedding: Precomputed embedding tensor (batch_size, embedding_dim)
        Returns:
            Logits for each class
        """
        logits = self.classifier(embedding)
        return logits


