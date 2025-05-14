#!/usr/bin/env python3
"""
Inference pipeline for AI-generated code detection.

This script handles the prediction pipeline for detecting AI-generated code:
1. Code embedding generation
2. Feature extraction
3. Model prediction
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np

# Import project modules
from config import FEATURE_COLUMNS, FILE_PATHS, MODEL_CONFIGS, LOGGING_CONFIG
from models.xgboost_model import UnixCoderEncoder, XGBoostClassifier

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

def extract_code_features(code: str) -> Dict[str, float]:
    """
    Extract basic code features for the model.
    
    This is a simplified version - in a production system, you would use 
    more sophisticated feature extraction.
    
    Args:
        code: Source code to analyze
        
    Returns:
        Dictionary containing code features
    """
    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    whitespace_ratio = sum(1 for c in code if c.isspace()) / max(1, len(code))
    
    # Calculate average line length
    avg_line_length = sum(len(line) for line in lines) / max(1, len(lines))
    
    # Calculate average identifier length (simplified)
    identifiers = []
    for line in lines:
        tokens = line.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').split()
        identifiers.extend([token for token in tokens if token.isalnum() and not token.isdigit()])
    
    avg_identifier_length = sum(len(id) for id in identifiers) / max(1, len(identifiers))
    
    # Calculate empty lines density
    empty_lines_density = (len(lines) - len(non_empty_lines)) / max(1, len(lines))
    
    # Return dictionary with all required features
    return {
        'avgFunctionLength': 1.0,  # Placeholder
        'avgIdentifierLength': avg_identifier_length,
        'avgLineLength': avg_line_length,
        'emptyLinesDensity': empty_lines_density,
        'functionDefinitionDensity': 0.0,  # Placeholder
        'maintainabilityIndex': 0.0,  # Placeholder
        'maxDecisionTokens': 0.0,  # Placeholder
        'whiteSpaceRatio': whitespace_ratio
    }

def detect_language_from_file(file_path: str) -> Optional[str]:
    """
    Detect programming language from file extension.
    
    Args:
        file_path: Path to the code file
        
    Returns:
        Detected language or None if unknown
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.py':
        return 'python'
    elif extension in ['.java', '.kt']:
        return 'java'
    elif extension in ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp']:
        return 'cpp'
    elif extension in ['.js', '.ts']:
        return 'javascript'
    elif extension in ['.rb']:
        return 'ruby'
    elif extension in ['.go']:
        return 'go'
    elif extension in ['.rs']:
        return 'rust'
    elif extension in ['.php']:
        return 'php'
    
    return None

def load_code_from_files(file_paths: List[str]) -> Tuple[List[str], List[str], List[Dict[str, float]]]:
    """
    Load code from multiple files and extract features.
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        Tuple of (code_samples, languages, features_list)
    """
    code_samples = []
    languages = []
    features_list = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            code_samples.append(code)
            languages.append(detect_language_from_file(file_path))
            features_list.append(extract_code_features(code))
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            # Skip this file
            continue
    
    return code_samples, languages, features_list

def predict_single_sample(
    code: str, 
    language: Optional[str] = None,
    model_config: Dict[str, Any] = None,
    model_path: str = None,
    importance_path: str = None,
    feature_columns: List[str] = None
) -> Dict[str, Any]:
    """
    Make a prediction for a single code sample.
    
    Args:
        code: Source code to analyze
        language: Programming language of the code
        model_config: Model configuration dictionary
        model_path: Path to the model file
        importance_path: Path to the feature importance file
        feature_columns: List of feature column names
        
    Returns:
        Dictionary with prediction results
    """
    # Extract features
    features = extract_code_features(code)
    
    # Use inference pipeline
    result = inference_pipeline(
        [code], 
        [language] if language else None,
        [features],
        model_config,
        model_path,
        importance_path,
        feature_columns
    )
    
    probability = float(result[0])
    
    return {
        'probability': probability,
        'is_ai_generated': probability > 0.5,
        'language': language,
        'features': features
    }

def inference_pipeline(
    code_samples: List[str], 
    languages: Optional[List[str]] = None, 
    features_list: Optional[List[Dict[str, float]]] = None,
    model_config: Dict[str, Any] = None,
    model_path: str = None,
    importance_path: str = None,
    feature_columns: List[str] = None
) -> np.ndarray:
    """
    Execute the optimized inference pipeline on new code samples.
    
    Args:
        code_samples: List of code samples to classify
        languages: List of programming languages for the samples
        features_list: List of feature dictionaries for each sample
        model_config: Model configuration dictionary
        model_path: Path to the model file
        importance_path: Path to the feature importance file
        feature_columns: List of feature column names
        
    Returns:
        Array of predicted probabilities (1 = AI-generated, 0 = human)
    """
    # Set defaults
    if model_config is None:
        model_config = MODEL_CONFIGS["xgboost"]
    if model_path is None:
        model_path = FILE_PATHS["model"]
    if importance_path is None:
        importance_path = FILE_PATHS["feature_importance"]
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    
    # Validate inputs
    if not code_samples:
        raise ValueError("No code samples provided")
    
    if languages and len(languages) != len(code_samples):
        raise ValueError("Number of languages must match number of code samples")
        
    if features_list and len(features_list) != len(code_samples):
        raise ValueError("Number of feature dictionaries must match number of code samples")
    
    # Load the UnixCoder encoder with config
    encoder_config = model_config["encoder"]
    encoder = UnixCoderEncoder(
        model_name=encoder_config["model_name"],
        max_length=encoder_config["max_length"],
        language_prefixes=encoder_config["language_prefixes"]
    )
    
    # Generate embeddings for the code samples in batches
    logger.info(f"Generating embeddings for {len(code_samples)} samples...")
    embeddings = encoder.batch_encode(code_samples, languages, batch_size=encoder_config["batch_size"])
    
    # Prepare input data
    X_embeddings = embeddings
    
    # If we have features, prepare feature matrices
    if features_list:
        # Extract feature values in the correct order
        X_features = np.array([[features.get(f, 0) for f in feature_columns] for features in features_list])
        # Combine embeddings and features
        X = np.hstack((X_embeddings, X_features))
    else:
        X = X_embeddings
    
    # Load the classifier
    classifier = XGBoostClassifier(feature_columns=feature_columns)
    
    try:
        classifier.load_model(model_path, importance_path)
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    
    # Make predictions
    logger.info("Making predictions...")
    probabilities = classifier.predict(X)
    
    return probabilities

def process_file_batch(
    file_paths: List[str], 
    model_config: Dict[str, Any] = None,
    model_path: str = None,
    importance_path: str = None,
    feature_columns: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Process a batch of files and make predictions.
    
    Args:
        file_paths: List of file paths to process
        model_config: Model configuration dictionary
        model_path: Path to the model file
        importance_path: Path to the feature importance file
        feature_columns: List of feature column names
        
    Returns:
        List of prediction results for each file
    """
    # Load code and extract features
    code_samples, languages, features_list = load_code_from_files(file_paths)
    
    if not code_samples:
        logger.warning("No valid code samples found")
        return []
    
    # Get predictions
    probabilities = inference_pipeline(
        code_samples, 
        languages,
        features_list,
        model_config,
        model_path,
        importance_path,
        feature_columns
    )
    
    # Format results
    results = []
    for i, (file_path, prob) in enumerate(zip(file_paths, probabilities)):
        if i < len(code_samples):  # Make sure we only process valid samples
            results.append({
                'file': file_path,
                'probability': float(prob),
                'is_ai_generated': float(prob) > 0.5,
                'language': languages[i],
                'features': features_list[i]
            })
    
    return results

def main():
    """Main function to run the inference pipeline."""
    parser = argparse.ArgumentParser(description='Detect AI-generated code')
    parser.add_argument('files', nargs='+', help='File path(s) to analyze')
    parser.add_argument('--model-type', type=str, default='xgboost', 
                        help='Type of model to use for prediction')
    parser.add_argument('--json', action='store_true', 
                        help='Output results in JSON format')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for classifying as AI-generated')
    
    args = parser.parse_args()
    
    # Check if model type is supported
    if args.model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {args.model_type}. Available options: {list(MODEL_CONFIGS.keys())}")
    
    # Get configuration for the specified model
    model_config = MODEL_CONFIGS[args.model_type]
    
    # Process files
    results = process_file_batch(
        args.files,
        model_config=model_config,
        model_path=FILE_PATHS["model"],
        importance_path=FILE_PATHS["feature_importance"],
        feature_columns=FEATURE_COLUMNS
    )
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            verdict = "AI-GENERATED" if result['probability'] > args.threshold else "HUMAN-WRITTEN"
            print(f"File: {result['file']}")
            print(f"Language: {result['language'] or 'unknown'}")
            print(f"AI probability: {result['probability']:.4f}")
            print(f"Verdict: {verdict}")
            print("-" * 50)

if __name__ == "__main__":
    main() 