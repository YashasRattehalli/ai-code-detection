import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Set up logging
logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    """
    Dataset for code classification tasks.
    """
    def __init__(
        self,
        embeddings: List[np.ndarray],
        labels: Optional[List[int]] = None,
        max_length: int = 1024
    ):
        """
        Initialize dataset with code samples and labels.
        
        Args:
            embeddings: List of precomputed embeddings
            labels: Optional list of labels (1 for AI, 0 for human)
            max_length: Maximum sequence length
        """
        self.embeddings = embeddings
        self.labels = labels
        self.max_length = max_length

        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        item = {
            "embedding": self.embeddings[idx]
        }
        # Add label if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item


class UnixCoderClassifier(nn.Module):
    """
    Classifier model that takes precomputed embeddings as input with a classification head.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int = 2,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the classifier model.
        Args:
            embedding_dim: Dimension of the input embeddings
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout probability for regularization
        """
        super(UnixCoderClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, num_classes)
        )
    
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


class UnixCoderClassifierTrainer:
    """
    Trainer class for training and evaluating the classifier on precomputed embeddings.
    """
    def __init__(
        self,
        model: UnixCoderClassifier,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer.
        Args:
            model: UnixCoderClassifier model
            device: Device to use for training (cpu or gpu)
        """
        self.device = device if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def train(
        self,
        train_dataset: CodeDataset,
        eval_dataset: Optional[CodeDataset] = None,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
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
            weight_decay: Weight decay for regularization
            eval_steps: Number of steps between evaluations
            save_dir: Directory to save model checkpoints
        Returns:
            Dictionary containing training metrics
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_loader) * num_epochs
        metrics: Dict[str, List[float]] = {
            'train_loss': [], 'eval_loss': [], 'accuracy': [],
            'precision': [], 'recall': [], 'f1': [], 'auc': []
        }
        logger.info(f"Starting training for {num_epochs} epochs...")
        global_step = 0
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            for batch in train_loader:
                embeddings = batch['embedding'].to(self.device)
                labels = batch['labels'].to(self.device)
                optimizer.zero_grad()
                logits = self.model(embeddings)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                global_step += 1
                if eval_dataset and global_step % eval_steps == 0:
                    eval_results = self.evaluate(eval_dataset, batch_size)
                    logger.info(
                        f"Step {global_step} | "
                        f"Eval Loss: {eval_results['loss']:.4f} | "
                        f"Accuracy: {eval_results['accuracy']:.4f} | "
                        f"F1: {eval_results['f1']:.4f}"
                    )
                    for key in eval_results:
                        if key in metrics:
                            metrics[key].append(eval_results[key])
            avg_epoch_loss = epoch_loss / len(train_loader)
            metrics['train_loss'].append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_epoch_loss:.4f}")
            if eval_dataset:
                eval_results = self.evaluate(eval_dataset, batch_size)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Eval Loss: {eval_results['loss']:.4f} | "
                    f"Accuracy: {eval_results['accuracy']:.4f} | "
                    f"F1: {eval_results['f1']:.4f}"
                )
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.save_model(
                save_dir,
                train_metrics=self._extract_latest_metrics(metrics),
                test_metrics=self.evaluate(eval_dataset, batch_size) if eval_dataset else None
            )
        return metrics
    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embeddings = batch['embedding'].to(self.device)
        labels = batch.get('labels', None)
        if labels is not None:
            labels = labels.to(self.device)
        logits = self.model(embeddings)
        return logits, labels
    def evaluate(
        self, 
        eval_dataset: CodeDataset, 
        batch_size: int = 16
    ) -> Dict[str, float]:
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[float] = []
        total_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                logits, labels = self._process_batch(batch)
                if labels is not None:
                    loss = self.loss_fn(logits, labels)
                    total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                if labels is not None:
                    all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        metrics = {'loss': total_loss / len(eval_loader)}
        if all_labels:
            metrics['accuracy'] = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
            try:
                metrics['auc'] = roc_auc_score(all_labels, all_probs)
            except ValueError:
                metrics['auc'] = 0.0
        return metrics
    
    def predict(
        self, 
        embeddings: List[np.ndarray],
        batch_size: int = 16
    ) -> Tuple[List[int], List[float]]:
        dataset = CodeDataset(
            embeddings=embeddings,
            labels=None
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        all_preds: List[int] = []
        all_probs: List[float] = []
        with torch.no_grad():
            for batch in data_loader:
                logits, _ = self._process_batch(batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        return all_preds, all_probs
    def save_model(
        self, 
        save_path: str, 
        train_metrics: Optional[Dict[str, float]] = None, 
        test_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, "model.pt")
        logger.info(f"Saving model to {model_path}...")
        torch.save(self.model.state_dict(), model_path)
        
        # Save model configuration
        model_config = {
            "embedding_dim": self.model.classifier[1].in_features,
            "num_classes": self.model.classifier[-1].out_features,
            "dropout_rate": self.model.classifier[0].p
        }
        
        model_info: Dict[str, Any] = {
            "model_type": "unixcoder-classifier",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_config": model_config,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }
        info_path = os.path.join(save_path, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Model and metrics saved to {save_path}")
    @classmethod
    def load_model(
        cls,
        model_path: str,
        embedding_dim: Optional[int] = None,
        device: Optional[torch.device] = None
    ) -> 'UnixCoderClassifierTrainer':
        # Load model config
        with open(os.path.join(model_path, "model_info.json"), 'r') as f:
            model_info = json.load(f)
        
        # Use saved config if available, otherwise use provided parameters
        model_config = model_info.get("model_config", {})
        embedding_dim = model_config.get("embedding_dim", embedding_dim)
        num_classes = model_config.get("num_classes", 2)
        dropout_rate = model_config.get("dropout_rate", 0.1)
        
        if embedding_dim is None:
            raise ValueError("embedding_dim must be provided either in model_info.json or as a parameter")
        
        # Initialize model with configuration
        model = UnixCoderClassifier(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # Load the model weights
        model.load_state_dict(torch.load(
            os.path.join(model_path, "model.pt"),
            map_location=device if device else torch.device("cpu")
        ))
        
        trainer = cls(
            model=model,
            device=device
        )
        return trainer
    def _extract_latest_metrics(self, metrics_dict: Dict) -> Dict[str, float]:
        final_metrics = {}
        for key, value in metrics_dict.items():
            if isinstance(value, list) and value:
                final_metrics[key] = value[-1]
            else:
                final_metrics[key] = value
        return final_metrics 