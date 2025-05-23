import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from ai_code_detector.models.embedding_classifier import EmbeddingClassifier
from ai_code_detector.models.unixcoder import UnixCoderModel

# Set up logging
logger = logging.getLogger(__name__)
    

class Trainer:
    """
    Generic trainer class for training and evaluating embedding-based classifiers.
    """
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer.
        Args:
            model: EmbeddingClassifier model
            device: Device to use for training (cpu or gpu)
        """
        self.device = device if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
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
            save_dir: Directory to save model checkpoints
        Returns:
            Dictionary containing training metrics
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        metrics: Dict[str, List[float]] = {
            'train_loss': [], 'eval_loss': [], 'accuracy': [],
            'precision': [], 'recall': [], 'f1': [], 'auc': []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                logits, labels = self._process_batch(batch)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            metrics['train_loss'].append(avg_epoch_loss)
            
            # Evaluate only at the end of each epoch
            if eval_dataset:
                eval_results = self.evaluate(eval_dataset, batch_size)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {avg_epoch_loss:.4f} | "
                    f"Eval Loss: {eval_results['loss']:.4f} | "
                    f"Accuracy: {eval_results['accuracy']:.4f} | "
                    f"F1: {eval_results['f1']:.4f}"
                )
                for key in eval_results:
                    if key in metrics:
                        metrics[key].append(eval_results[key])
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_epoch_loss:.4f}")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.save_model(
                save_dir,
                train_metrics=self._extract_latest_metrics(metrics),
                test_metrics=self.evaluate(eval_dataset, batch_size) if eval_dataset else None
            )
        return metrics
        
    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process a batch based on the model type."""
        labels = batch.get('labels', None)
        
        # Check if this is an embedding-based model or a tokenized input model
        if 'embedding' in batch:
            # EmbeddingClassifier - takes embeddings directly
            embeddings = batch['embedding']
            logits = self.model(embeddings)
        elif 'input_ids' in batch and 'attention_mask' in batch:
            # UnixCoderModel - takes tokenized inputs
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            raise ValueError("Batch must contain either 'embedding' or both 'input_ids' and 'attention_mask'")
        
        return logits, labels
        
    def evaluate(
        self, 
        eval_dataset: Dataset, 
        batch_size: int = 16
    ) -> Dict[str, float]:
        """
        Evaluate the model on a given dataset.
        Args:
            eval_dataset: Dataset to evaluate
            batch_size: Batch size for evaluation
        Returns:
            Dictionary containing evaluation metrics    
        """
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
        dataset: Dataset,
        batch_size: int = 16,
    ) -> Tuple[List[int], List[float]]:
        """
        Predict the labels of a dataset.
        Args:
            dataset: Dataset to predict
            batch_size: Batch size for prediction
        Returns:
            Tuple of lists containing predicted labels and probabilities
        """
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
        
        # Save model configuration based on model type
        model_config = {}
        
        # Try to extract configuration safely based on model type
        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Sequential):
            # EmbeddingClassifier case
            try:
                classifier = self.model.classifier
                # Find the first Linear layer for embedding_dim
                for layer in classifier:
                    if isinstance(layer, nn.Linear):
                        model_config["embedding_dim"] = layer.in_features
                        break
                
                # Find the last Linear layer for num_classes
                for layer in reversed(list(classifier)):
                    if isinstance(layer, nn.Linear):
                        model_config["num_classes"] = layer.out_features
                        break
                
                # Find dropout rate
                for layer in classifier:
                    if isinstance(layer, nn.Dropout):
                        model_config["dropout_rate"] = layer.p
                        break
            except (AttributeError, IndexError):
                logger.warning("Could not extract model configuration from classifier layers")
        
        elif hasattr(self.model, 'unixcoder') and hasattr(self.model, 'classifier'):
            # UnixCoderModel case
            try:
                model_config["model_name"] = getattr(self.model, 'model_name', 'microsoft/unixcoder-base')
                if hasattr(self.model.unixcoder, 'config') and hasattr(self.model.unixcoder.config, 'hidden_size'):
                    model_config["hidden_size"] = self.model.unixcoder.config.hidden_size
                
                # Find the last Linear layer for num_classes if classifier is Sequential
                if isinstance(self.model.classifier, nn.Sequential):
                    classifier = self.model.classifier
                    for layer in reversed(list(classifier)):
                        if isinstance(layer, nn.Linear):
                            model_config["num_classes"] = layer.out_features
                            break
                    
                    # Find dropout rate
                    for layer in classifier:
                        if isinstance(layer, nn.Dropout):
                            model_config["dropout_rate"] = layer.p
                            break
            except (AttributeError, IndexError):
                logger.warning("Could not extract model configuration from UnixCoder model")
        
        model_info: Dict[str, Any] = {
            "model_type": type(self.model).__name__.lower().replace('model', '').replace('classifier', ''),
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
        model_name: str,
        model_path: str,
        embedding_dim: Optional[int] = None,
        device: Optional[torch.device] = None
    ) -> 'Trainer':
        """
        Load a model from a given path.
        Args:
            model_name: Name of the model ('embedding_classifier' or 'unixcoder')
            model_path: Path to the model
            embedding_dim: Dimension of the embeddings (for embedding_classifier)
            device: Device to use for the model
        Returns:
            Trainer object
        """
        with open(os.path.join(model_path, "model_info.json"), 'r') as f:
            model_info = json.load(f)
        
        model_config = model_info.get("model_config", {})
        
        # Initialize model based on model type
        if model_name == "embedding_classifier":
            embedding_dim = model_config.get("embedding_dim", embedding_dim)
            num_classes = model_config.get("num_classes", 2)
            dropout_rate = model_config.get("dropout_rate", 0.1)
            
            if embedding_dim is None:
                raise ValueError("embedding_dim must be provided either in model_info.json or as a parameter")
            
            model = EmbeddingClassifier(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                dropout_rate=dropout_rate
            )
        elif model_name == "unixcoder":
            model_name_str = model_config.get("model_name", "microsoft/unixcoder-base")
            num_classes = model_config.get("num_classes", 2)
            dropout_rate = model_config.get("dropout_rate", 0.1)
            
            model = UnixCoderModel(
                model_name=model_name_str,
                dropout_rate=dropout_rate,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Model type {model_name} not supported. Use 'embedding_classifier' or 'unixcoder'")
        
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