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
from typing import Any, Dict, List, Optional, Tuple

# Import project modules
from ai_code_detector.config import FEATURE_COLUMNS, FILE_PATHS, LOGGING_CONFIG, MODEL_CONFIGS
from ai_code_detector.core import CodeDetector
from ai_code_detector.models.feature_extractor import FeatureExtractor

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class InferencePipeline(CodeDetector):
    """
    Pipeline for making predictions on new code samples.
    
    This class extends the base CodeDetector to provide a simple
    interface for code detection.
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        threshold: float = 0.5,
        model_config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        importance_path: Optional[str] = None,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            model_type: Type of model to use ("xgboost" or "unixcoder")
            threshold: Probability threshold for binary classification
            model_config: Optional model configuration
            model_path: Path to the model file
            importance_path: Path to the feature importance file
            feature_columns: List of feature column names
        """
        # Get configurations
        self.model_type = model_type
        self.threshold = threshold
        
        # Use provided config or get from MODEL_CONFIGS
        if model_config is None:
            if model_type not in MODEL_CONFIGS:
                raise ValueError(f"Unsupported model type: {model_type}. "
                                 f"Available options: {list(MODEL_CONFIGS.keys())}")
            model_config = MODEL_CONFIGS[model_type]
        
        # Use provided paths or get from FILE_PATHS
        if model_path is None:
            if model_type == "xgboost":
                model_path = FILE_PATHS["model"]
            elif model_type == "unixcoder":
                model_path = FILE_PATHS["unixcoder_model"]
            else:
                model_path = FILE_PATHS["model"]
                
        if importance_path is None:
            if model_type == "xgboost":
                importance_path = FILE_PATHS["feature_importance"]
            elif model_type == "unixcoder":
                importance_path = FILE_PATHS["unixcoder_importance"]
            else:
                importance_path = FILE_PATHS["feature_importance"]
                
        if feature_columns is None:
            feature_columns = FEATURE_COLUMNS
        
        # Initialize base class
        super().__init__(
            model_config=model_config,
            feature_columns=feature_columns,
            model_path=model_path,
            importance_path=importance_path,
            load_model=True,
            model_type=model_type
        )
    
    def predict_single(
        self,
        code: str,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Make a prediction for a single code sample.
        
        Args:
            code: Source code to analyze
            language: Programming language of the code
            
        Returns:
            Dictionary with prediction results
        """
        # Auto-detect language if not provided
        if not language:
            language = FeatureExtractor.detect_language(code)
            logger.info(f"Detected language: {language}")
        
        # Extract features
        features = FeatureExtractor.extract_basic_features(code)
        
        # Get prediction from base class
        probabilities, detected_languages = super().predict([code], [language] if language else None, [features])
        probability = float(probabilities[0])
        
        # Format the result
        return {
            'probability': probability,
            'is_ai_generated': probability > self.threshold,
            'language': detected_languages[0],
            'features': features
        }
    
    def predict_batch(
        self,
        code_samples: List[str],
        languages: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of code samples.
        
        Args:
            code_samples: List of code strings to analyze
            languages: Optional list of programming languages for each sample
            sample_ids: Optional list of identifiers for each sample
            
        Returns:
            List of dictionaries with prediction results
        """
        if not code_samples:
            logger.warning("No code samples provided")
            return []
    
        # Get predictions using base class
        probabilities, detected_languages = super().predict(code_samples, languages)
        
        # Format results
        results = []
        for i, prob in enumerate(probabilities):
            sample_id = sample_ids[i] if sample_ids and i < len(sample_ids) else f"sample_{i}"
            results.append({
                'id': sample_id,
                'probability': float(prob),
                'is_ai_generated': float(prob) > self.threshold,
                'language': detected_languages[i],
            })
        
        return results


def main():
    """Main function to run the inference pipeline."""
    parser = argparse.ArgumentParser(description='Detect AI-generated code')
    parser.add_argument('--code', type=str, help='Direct code string to analyze (optional)')
    parser.add_argument('--model-type', type=str, default='xgboost', 
                        choices=['xgboost', 'unixcoder'],
                        help='Type of model to use for prediction')
    parser.add_argument('--json', action='store_true', 
                        help='Output results in JSON format')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for classifying as AI-generated')
    
    args = parser.parse_args()
    
    # Initialize the inference pipeline
    pipeline = InferencePipeline(
        model_type=args.model_type,
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