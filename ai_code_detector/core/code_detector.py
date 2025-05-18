"""
Base class for AI code detection functionality.

This module provides the core functionality for AI-generated code detection
that is shared between training and inference pipelines.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ai_code_detector.models.classifier_adapter import ClassifierAdapter
from ai_code_detector.models.feature_extractor import FeatureExtractor
from ai_code_detector.models.unixcoder_embedder import UnixCoderEncoder

# Set up logging
logger = logging.getLogger(__name__)

class CodeDetector:
    """
    Base class for AI-generated code detection.
    
    This class provides the core functionality for preparing data, generating
    embeddings, and making predictions. It is designed to be extended by
    both training and inference pipeline implementations.
    """
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
        load_model: bool = False,
        model_type: str = "xgboost"
    ):
        """
        Initialize the code detector with configuration.
        
        Args:
            model_config: Configuration dictionary for the model
            feature_columns: List of feature column names to use
            model_path: Path to load/save the model
            embeddings_path: Path to load/save embeddings
            load_model: Whether to load the model on initialization
            model_type: Type of model to use ("xgboost" or "unixcoder")
        """
        self.model_config = model_config or {}
        self.feature_columns = feature_columns or []
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self.model_type = model_type
        
        # Initialize encoder
        encoder_config = self.model_config.get("encoder", {})
        self.encoder = UnixCoderEncoder(
            model_name=encoder_config.get("model_name", "microsoft/unixcoder-base"),
            max_length=encoder_config.get("max_length", 512),
            language_prefixes=encoder_config.get("language_prefixes", None)
        )
        
        # Initialize classifier adapter
        self.classifier = ClassifierAdapter(
            model_type=self.model_type,
            model_config=self.model_config,
            feature_columns=self.feature_columns
        )
        
        # Load model if requested
        if load_model and self.model_path:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model if it exists."""
        if not self.model_path:
            logger.warning("No model path specified, skipping model loading")
            return
            
        try:
            model_info = self.classifier.load_model(
                model_path=self.model_path
            )
            logger.info(f"Successfully loaded {self.model_type} model from {self.model_path}")
            if model_info and 'metrics' in model_info:
                logger.info(f"Model metrics: {model_info['metrics']}")
        except FileNotFoundError as e:
            logger.warning(f"Could not load model: {str(e)}")
    
    def load_embeddings(self, embeddings_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Load pre-computed embeddings from a file.
        
        Args:
            embeddings_path: Path to the embeddings file
            
        Returns:
            Array of embeddings or None if loading fails
        """
        path = embeddings_path or self.embeddings_path
        if not path:
            return None
            
        try:
            if os.path.exists(path):
                logger.info(f"Loading pre-computed embeddings from {path}")
                with open(path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading embeddings: {str(e)}")
            
        return None
    
    def save_embeddings(self, embeddings: np.ndarray, embeddings_path: Optional[str] = None) -> bool:
        """
        Save embeddings to a file.
        
        Args:
            embeddings: Array of embeddings to save
            embeddings_path: Path to save the embeddings
            
        Returns:
            True if saving was successful, False otherwise
        """
        path = embeddings_path or self.embeddings_path
        if not path:
            logger.warning("No embeddings path specified, skipping save")
            return False
            
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info(f"Saving embeddings to {path}")
            with open(path, 'wb') as f:
                pickle.dump(embeddings, f)
            return True
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            return False
    
    def generate_embeddings(
        self, 
        code_samples: List[str], 
        languages: Optional[List[Optional[str]]] = None,
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Generate embeddings for code samples.
        
        Args:
            code_samples: List of code samples
            languages: List of languages for each sample
            batch_size: Batch size for embedding generation
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Generating embeddings for {len(code_samples)} code samples")
        return self.encoder.batch_encode(
            code_samples, 
            languages, 
            batch_size=batch_size
        )
    
    def extract_features(self, code_samples: List[str]) -> List[Dict[str, float]]:
        """
        Extract features from code samples.
        
        Args:
            code_samples: List of code samples
            
        Returns:
            List of feature dictionaries
        """
        logger.info(f"Extracting features from {len(code_samples)} code samples")
        return [FeatureExtractor.extract_basic_features(code) for code in tqdm(
            code_samples, desc="Extracting features")]
    
    def detect_languages(
        self, 
        code_samples: List[str], 
        file_paths: Optional[List[str]] = None
    ) -> List[Optional[str]]:
        """
        Detect programming languages for code samples.
        
        Args:
            code_samples: List of code samples
            file_paths: Optional list of file paths
            
        Returns:
            List of detected languages
        """
        languages = []
        
        for i, code in enumerate(code_samples):
            file_path = file_paths[i] if file_paths and i < len(file_paths) else None
            languages.append(FeatureExtractor.detect_language(code))
            
        return languages
    
    def prepare_features_matrix(
        self, 
        embeddings: np.ndarray, 
        features_list: List[Dict[str, float]]
    ) -> np.ndarray:
        """
        Prepare the feature matrix by combining embeddings and extracted features.
        
        Args:
            embeddings: Array of code embeddings
            features_list: List of feature dictionaries
            
        Returns:
            Combined feature matrix
        """
        if len(embeddings) != len(features_list):
            raise ValueError("Number of embeddings and feature dictionaries must match")
            
        # Extract feature values in the correct order
        if self.feature_columns:
            feature_matrix = np.array([
                [features.get(f, 0.0) for f in self.feature_columns] 
                for features in features_list
            ])
            # Combine embeddings and features
            return np.hstack((embeddings, feature_matrix))
        else:
            return embeddings
    
    def sample_balanced_dataset(
        self, 
        df: pd.DataFrame, 
        total_samples: int = 10000
    ) -> pd.DataFrame:
        """
        Sample a dataset with balanced class distribution.
        
        Args:
            df: Input DataFrame with 'target_binary' column
            total_samples: Total number of samples to return
                
        Returns:
            DataFrame with balanced class distribution
        """
        if len(df) <= total_samples:
            logger.info(f"Using all {len(df)} rows from dataset (less than {total_samples})")
            return df
        
        logger.info(f"Sampling {total_samples} records with equal target distribution...")
        # Get samples per class (half of total)
        samples_per_class = total_samples // 2
        
        # Get dataframes for each class
        df_ai = df[df['target_binary'] == 1]
        df_human = df[df['target_binary'] == 0]
        
        # Adjust samples_per_class if one class has fewer than required samples
        samples_per_class = min(samples_per_class, len(df_ai), len(df_human))
        
        # Sample equal number from each class
        df_ai_sampled = df_ai.sample(n=samples_per_class, random_state=42)
        df_human_sampled = df_human.sample(n=samples_per_class, random_state=42)
        
        # Combine the samples
        sampled_df = pd.concat([df_ai_sampled, df_human_sampled]).reset_index(drop=True)
        
        logger.info(f"Sampled dataset contains {len(sampled_df)} records")
        # Log new class distribution
        new_class_counts = sampled_df['target_binary'].value_counts()
        logger.info(f"New class distribution: {new_class_counts.to_dict()}")
        
        return sampled_df
    
    def load_code_from_files(
        self, 
        file_paths: List[str]
    ) -> Tuple[List[str], List[Optional[str]], List[Dict[str, float]]]:
        """
        Load code from multiple files and extract features.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Tuple of (code_samples, languages, features_list)
        """
        code_samples = []
        languages = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    
                code_samples.append(code)
                languages.append(FeatureExtractor.detect_language(code))
                
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                # Skip this file
                continue
        
        # Extract features for loaded code
        features_list = self.extract_features(code_samples)
        
        return code_samples, languages, features_list
    
    def predict(
        self, 
        code_samples: List[str], 
        languages: Optional[List[Optional[str]]] = None,
        features_list: Optional[List[Dict[str, float]]] = None
    ) -> Tuple[np.ndarray, List[Optional[str]]]:
        """
        Make predictions for code samples.
        
        Args:
            code_samples: List of code samples
            languages: Optional list of languages
            features_list: Optional list of pre-extracted features
            
        Returns:
            Tuple of (prediction_probabilities, detected_languages)
        """
        # Detect languages if not provided
        if languages is None:
            languages = self.detect_languages(code_samples)
            
        # Extract features if not provided
        if features_list is None:
            features_list = self.extract_features(code_samples)
            
        # For UnixCoder models, we can directly use the model's predict method
        if self.model_type == "unixcoder":
            probabilities = self.classifier.predict(code_samples, languages)
            return probabilities, languages
            
        # For other models (XGBoost), we need to generate embeddings first
        # Generate embeddings
        batch_size = self.model_config.get("encoder", {}).get("batch_size", 16)
        embeddings = self.generate_embeddings(code_samples, languages, batch_size)
        
        # Prepare feature matrix
        X = self.prepare_features_matrix(embeddings, features_list)
        
        # Make predictions with the classifier
        probabilities = self.classifier.predict(X)
        
        return probabilities, languages