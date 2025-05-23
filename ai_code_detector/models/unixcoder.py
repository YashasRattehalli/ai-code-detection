import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

# Set up logging
logger = logging.getLogger(__name__)

class UnixCoderDataset(Dataset):
    """
    Dataset for code classification tasks using raw text input.
    """
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer_name: str = "microsoft/unixcoder-base",
        max_length: int = 1024,
        device: Optional[torch.device] = None
    ):
        """
        Initialize dataset with code samples and labels.
        
        Args:
            texts: List of code/text samples
            labels: Optional list of labels (1 for AI, 0 for human)
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length for tokenization
            device: Device to store tensors on for efficiency
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.texts = texts
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Handle labels with proper typing
        self.labels: Optional[torch.Tensor] = None
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long).to(self.device)

        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Handle missing/null values by converting to string
        if text is None or (isinstance(text, float) and text != text):  # Check for NaN
            text = ""  # Use empty string for missing values
        elif not isinstance(text, str):
            text = str(text)  # Convert to string if not already
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            "input_ids": encoding['input_ids'].squeeze(0).to(self.device),
            "attention_mask": encoding['attention_mask'].squeeze(0).to(self.device)
        }
        
        # Add label if available
        if self.labels is not None:
            item['labels'] = self.labels[idx]
            
        return item


class UnixCoderModel(nn.Module):
    """
    Full UnixCoder model with classification head for finetuning.
    """
    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        freeze_base_model: bool = False
    ):
        """
        Initialize the UnixCoder model with classification head.
        
        Args:
            model_name: Name of the pretrained UnixCoder model
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout probability for regularization
            freeze_base_model: Whether to freeze the base model parameters
        """
        super(UnixCoderModel, self).__init__()
        
        # Load the pretrained UnixCoder model
        self.unixcoder = AutoModel.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base_model:
            for param in self.unixcoder.parameters():
                param.requires_grad = False
        
        # Get the hidden size from the model config
        hidden_size = self.unixcoder.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Logits for each class (batch_size, num_classes)
        """
        # Get embeddings from UnixCoder
        outputs = self.unixcoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Pass through classifier
        logits = self.classifier(cls_embedding)
        
        return logits
