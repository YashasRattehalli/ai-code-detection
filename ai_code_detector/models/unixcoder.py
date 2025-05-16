import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Set up logging
logger = logging.getLogger(__name__)

class UnixCoderEncoder:
    """
    Class to create code embeddings using Microsoft's UnixCoder model.
    
    This encoder converts source code into embeddings that capture semantic
    meaning, which can be used for classification or other ML tasks.
    """
    
    def __init__(
        self, 
        model_name: str = "microsoft/unixcoder-base", 
        max_length: int = 1024,
        language_prefixes: Optional[Dict[str, str]] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the encoder with the specified model.
        
        Args:
            model_name: Name or path of the UnixCoder model
            max_length: Maximum sequence length for tokenization
            language_prefixes: Dictionary mapping language names to prefix tokens
            cache_dir: Optional directory to cache the downloaded model
        """
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model with cache directory if provided
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
        # Maximum sequence length
        self.max_length = max_length
        
        # Language-specific prefixes
        self.lang_prefixes = language_prefixes or {
            "python": "<python> ",
            "java": "<java> ",
            "cpp": "<cpp> "
        }
    
    def encode(self, code: str, language: Optional[str] = None) -> np.ndarray:
        """
        Encode source code into embeddings.
        
        Args:
            code: Source code to encode
            language: Programming language of the source code (optional)
            
        Returns:
            Numpy array containing code embeddings
        """
        # Preprocessing for different languages
        if language and language.lower() in self.lang_prefixes:
            code = f"{self.lang_prefixes[language.lower()]}{code}"
                
        # Tokenize the code with efficient padding and truncation
        encoded = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create tensors and move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get model output with no gradient computation
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Get embeddings from the last hidden state
            # Use the CLS token embedding as code representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return embeddings[0]  # Return the embedding array

    def batch_encode(self, codes: List[str], languages: Optional[List[str]] = None, batch_size: int = 8) -> np.ndarray:
        """
        Encode multiple code samples in batch for better efficiency.
        
        Args:
            codes: List of source code samples
            languages: List of corresponding programming languages (optional)
            batch_size: Number of samples to process at once
            
        Returns:
            Numpy array containing code embeddings
        """
        total_samples = len(codes)
        all_embeddings = []
        
        # Process in batches with progress reporting
        for i in tqdm(range(0, total_samples, batch_size), desc="Generating embeddings"):
            end_idx = min(i + batch_size, total_samples)
            batch_codes = codes[i:end_idx]
            
            # Handle languages if provided
            batch_preprocessed = batch_codes.copy()
            if languages:
                for j, (code, lang) in enumerate(zip(batch_codes, languages[i:end_idx])):
                    if lang and lang.lower() in self.lang_prefixes:
                        batch_preprocessed[j] = f"{self.lang_prefixes[lang.lower()]}{code}"
            
            # Tokenize the entire batch at once
            encoded = self.tokenizer(
                batch_preprocessed,
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
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(batch_embeddings)
                
        # Concatenate all batches
        return np.vstack(all_embeddings)