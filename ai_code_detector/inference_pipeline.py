#!/usr/bin/env python3
"""
Inference pipeline for AI-generated code detection.

This script handles the prediction pipeline for detecting AI-generated code:
1. Code embedding generation
2. Feature extraction
3. Model prediction
"""

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional

# Import project modules
from ai_code_detector.config import INFERENCE_FILE_PATHS, LOGGING_CONFIG, MODEL_CONFIGS
from ai_code_detector.models.code_embedder import CodeEmbeddingEncoder
from ai_code_detector.models.unixcoder import UnixCoderModel
from ai_code_detector.models.xgboost_classifier import XGBoostClassifier
from ai_code_detector.trainer import Trainer

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Pipeline for making predictions on new code samples.
    
    This class provides a simple interface for code detection.
    """
    
    def __init__(
        self,
        model_name: str = "xgboost",
        threshold: float = 0.5,
        model_config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            model_name: Type of model to use ("xgboost" or "unixcoder")
            threshold: Probability threshold for binary classification
            model_config: Optional model configuration
            model_path: Path to the model file
        """
        # Get configurations
        self.model_name = model_name
        self.threshold = threshold
        self.model_config = model_config
        self.model_path = model_path or INFERENCE_FILE_PATHS[model_name]
        self.code_embedding_model = CodeEmbeddingEncoder()
        
        # Use provided config or get from MODEL_CONFIGS
        if model_config is None:
            if model_name not in MODEL_CONFIGS:
                raise ValueError(f"Unsupported model type: {model_name}. "
                                 f"Available options: {list(MODEL_CONFIGS.keys())}")
            model_config = MODEL_CONFIGS[model_name]

        if model_name == "xgboost":
            self.classifier = XGBoostClassifier()
            model_info = self.classifier.load_model(self.model_path)
            logger.info(f"Loaded model info: {model_info}")
        elif model_name == "unixcoder":
            self.model = UnixCoderModel()
            self.classifier = Trainer(self.model)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}. "
                                 f"Available options: {list(MODEL_CONFIGS.keys())}")


    
    def predict_single(
        self,
        code: str,
    ) -> Dict[str, Any]:
        """
        Make a prediction for a single code sample.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary with prediction results
        """
        
        # Get prediction from base class
        if self.model_name == "xgboost":
            embedding = self.code_embedding_model.encode(code)
            probabilities = self.classifier.predict([embedding])
        elif self.model_name == "embedding_classifier":
            embedding = self.code_embedding_model.encode(code)
            probabilities = self.classifier.predict(embedding)
        probability = float(probabilities[0])
        
        # Format the result
        return {
            'probability': probability,
            'is_ai_generated': probability > self.threshold,
        }
    
    def predict_batch(
        self,
        code_samples: List[str],
        sample_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of code samples.
        
        Args:
            code_samples: List of code strings to analyze
            sample_ids: Optional list of identifiers for each sample
            
        Returns:
            List of dictionaries with prediction results
        """
        if not code_samples:
            logger.warning("No code samples provided")
            return []
    
        # Get prediction from base class
        if self.model_name == "xgboost":
            embedding = self.code_embedding_model.batch_encode(code_samples)
            probabilities = self.classifier.predict(embedding)
        elif self.model_name == "embedding_classifier":
            embedding = self.code_embedding_model.batch_encode(code_samples)
            probabilities = self.classifier.predict(embedding)
        if not probabilities:
            logger.warning("No probabilities found")
            return []
        
        # Format results
        results = []
        for i, prob in enumerate(probabilities):
            sample_id = sample_ids[i] if sample_ids and i < len(sample_ids) else f"sample_{i}"
            results.append({
                'id': sample_id,
                'probability': float(prob),
                'is_ai_generated': float(prob) > self.threshold,
            })
        
        return results


def main():
    """Main function to run the inference pipeline."""
    parser = argparse.ArgumentParser(description='Detect AI-generated code')
    parser.add_argument('--code', type=str, help='Direct code string to analyze (optional)')
    parser.add_argument('--model-name', type=str, default='xgboost', 
                        choices=['xgboost', 'unixcoder', 'embedding_classifier'],
                        help='Type of model to use for prediction')
    parser.add_argument('--json', action='store_true', 
                        help='Output results in JSON format')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for classifying as AI-generated')
    
    args = parser.parse_args()
    
    # Initialize the inference pipeline
    pipeline = InferencePipeline(
        model_name=args.model_name,
        threshold=args.threshold
    )
    
    if args.code:
        # Process single code sample as string
        result = pipeline.predict_single(args.code)
        logger.info(result)
    else:
        # No input provided
        logger.error("No input provided. Use --code or pipe code through stdin")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 