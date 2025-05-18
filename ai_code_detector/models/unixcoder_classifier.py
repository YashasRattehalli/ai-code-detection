import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Set up logging
logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    """
    Dataset for code classification tasks.
    """
    def __init__(
        self,
        code_samples: List[str],
        labels: Optional[List[int]] = None,
        tokenizer = None,
        max_length: int = 1024,
        languages: Optional[List[str]] = None,
        language_prefixes: Optional[Dict[str, str]] = None
    ):
        """
        Initialize dataset with code samples and labels.
        
        Args:
            code_samples: List of source code samples
            labels: Optional list of labels (1 for AI, 0 for human)
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            languages: Optional list of programming languages
            language_prefixes: Dictionary mapping language names to prefix tokens
        """
        self.code_samples = code_samples
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.languages = languages
        
        # Default language prefixes if none provided
        self.lang_prefixes = language_prefixes or {
            "python": "<python> ",
            "java": "<java> ",
            "cpp": "<cpp> ",
            "javascript": "<javascript> ",
            "go": "<go> ",
            "ruby": "<ruby> ",
            "php": "<php> "
        }
        
    def __len__(self):
        return len(self.code_samples)
    
    def __getitem__(self, idx):
        code = self.code_samples[idx]
        
        # Apply language prefix if available
        if self.languages and idx < len(self.languages):
            lang = self.languages[idx].lower()
            if lang in self.lang_prefixes:
                code = f"{self.lang_prefixes[lang]}{code}"
        
        # Tokenize code
        encoded = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        item = {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
        }
        
        # Add label if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item


class UnixCoderClassifier(nn.Module):
    """
    Classifier model that uses UnixCoder as the base encoder with a classification head.
    """
    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the classifier model.
        
        Args:
            model_name: Name or path of the UnixCoder model
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout probability for regularization
            cache_dir: Optional directory to cache the downloaded model
        """
        super(UnixCoderClassifier, self).__init__()
        
        # Load base UnixCoder model
        self.encoder = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Get the embedding dimension from the model config
        hidden_size = self.encoder.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Logits for each class
        """
        # Get UnixCoder embeddings
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        
        # Use CLS token embedding (first token) as code representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classification head
        logits = self.classifier(cls_output)
        
        return logits


class UnixCoderClassifierTrainer:
    """
    Trainer class for fine-tuning and evaluating the UnixCoder classifier.
    """
    def __init__(
        self,
        model: UnixCoderClassifier,
        tokenizer,
        device: Optional[torch.device] = None,
        max_length: int = 1024,
        language_prefixes: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: UnixCoderClassifier model
            tokenizer: Tokenizer for encoding inputs
            device: Device to use for training (cpu or gpu)
            max_length: Maximum sequence length
            language_prefixes: Dictionary mapping language names to prefix tokens
        """
        # Set device
        self.device = device if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang_prefixes = language_prefixes
    
    def train(
        self,
        train_dataset: CodeDataset,
        eval_dataset: Optional[CodeDataset] = None,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        eval_steps: int = 100,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataset: Dataset for training
            eval_dataset: Optional dataset for evaluation
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            warmup_steps: Warmup steps for learning rate scheduler
            weight_decay: Weight decay for regularization
            eval_steps: Number of steps between evaluations
            save_dir: Directory to save model checkpoints
            
        Returns:
            Dictionary containing training metrics
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        loss_fn = nn.CrossEntropyLoss()
        
        # Tracking metrics
        metrics = {
            'train_loss': [],
            'eval_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        # Training loop
        logger.info(f"Starting training for {num_epochs} epochs...")
        global_step = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Evaluation
                if eval_dataset and global_step % eval_steps == 0:
                    eval_results = self.evaluate(eval_dataset, batch_size)
                    
                    # Log evaluation results
                    logger.info(
                        f"Step {global_step} | "
                        f"Eval Loss: {eval_results['loss']:.4f} | "
                        f"Accuracy: {eval_results['accuracy']:.4f} | "
                        f"F1: {eval_results['f1']:.4f}"
                    )
                    
                    # Save metrics
                    metrics['eval_loss'].append(eval_results['loss'])
                    metrics['accuracy'].append(eval_results['accuracy'])
                    metrics['precision'].append(eval_results['precision'])
                    metrics['recall'].append(eval_results['recall'])
                    metrics['f1'].append(eval_results['f1'])
                    metrics['auc'].append(eval_results['auc'])
                    
                    # Save model checkpoint
                    if save_dir:
                        step_save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        os.makedirs(step_save_path, exist_ok=True)
                        self.save_model(step_save_path)
            
            # Epoch-level logging
            avg_epoch_loss = epoch_loss / len(train_loader)
            metrics['train_loss'].append(avg_epoch_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_epoch_loss:.4f}")
            
            # Final evaluation for this epoch
            if eval_dataset:
                eval_results = self.evaluate(eval_dataset, batch_size)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Eval Loss: {eval_results['loss']:.4f} | "
                    f"Accuracy: {eval_results['accuracy']:.4f} | "
                    f"F1: {eval_results['f1']:.4f}"
                )
                
                # Save final model for this epoch
                if save_dir:
                    epoch_save_path = os.path.join(save_dir, f"epoch-{epoch+1}")
                    os.makedirs(epoch_save_path, exist_ok=True)
                    self.save_model(epoch_save_path)
        
        # Save final model
        if save_dir:
            final_save_path = os.path.join(save_dir, "final")
            os.makedirs(final_save_path, exist_ok=True)
            self.save_model(final_save_path)
            
        return metrics
    
    def evaluate(
        self, 
        eval_dataset: CodeDataset, 
        batch_size: int = 16
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            eval_dataset: Dataset for evaluation
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Collect results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (AI)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        # Calculate AUC if possible (requires both classes to be present)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
        
        return {
            'loss': total_loss / len(eval_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def predict(
        self, 
        code_samples: List[str], 
        languages: Optional[List[str]] = None,
        batch_size: int = 16
    ) -> Tuple[List[int], List[float]]:
        """
        Predict on new code samples.
        
        Args:
            code_samples: List of source code samples
            languages: Optional list of programming languages
            batch_size: Batch size for prediction
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Create dataset
        dataset = CodeDataset(
            code_samples=code_samples,
            labels=None,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            languages=languages,
            language_prefixes=self.lang_prefixes
        )
        
        # Create dataloader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Make predictions
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Collect results
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (AI)
        
        return all_preds, all_probs
    
    def save_model(self, save_path: str):
        """
        Save model, tokenizer, and configuration.
        
        Args:
            save_path: Directory path to save model
        """
        # Save model
        torch.save(self.model.state_dict(), os.path.join(save_path, "model.pt"))
        
        # Save tokenizer and model config
        self.tokenizer.save_pretrained(save_path)
        
        # Save model parameters
        config = {
            "model_type": "unixcoder-classifier",
            "max_length": self.max_length,
            "language_prefixes": self.lang_prefixes
        }
        
        # Save config as JSON
        import json
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(
        cls,
        model_path: str,
        device: Optional[torch.device] = None
    ) -> 'UnixCoderClassifierTrainer':
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model
            device: Device to load model onto
            
        Returns:
            Loaded trainer instance
        """
        import json

        # Load configuration
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            config = json.load(f)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Initialize model
        model = UnixCoderClassifier()
        model.load_state_dict(torch.load(
            os.path.join(model_path, "model.pt"),
            map_location=device if device else torch.device("cpu")
        ))
        
        # Create trainer
        trainer = cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=config.get("max_length", 512),
            language_prefixes=config.get("language_prefixes", None)
        )
        
        return trainer 