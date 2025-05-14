#!/usr/bin/env python3
"""
AI-Generated Code Detection Script

This script demonstrates how to use the model to detect AI-generated code.
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

# Import project modules
from config import FEATURE_COLUMNS, FILE_PATHS, MODEL_CONFIGS
from inference_pipeline import (
    detect_language_from_file,
    extract_code_features,
    predict_single_sample,
    process_file_batch,
)


def detect_code_from_string(
    code: str, 
    language: Optional[str] = None,
    model_type: str = "xgboost"
) -> Dict[str, Any]:
    """
    Detect if a code string is AI-generated.
    
    Args:
        code: The source code to analyze
        language: Programming language of the code
        model_type: Type of model to use
        
    Returns:
        Dictionary with detection results
    """
    # Get model config for the specified type
    model_config = MODEL_CONFIGS.get(model_type)
    if not model_config:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    # Make prediction
    return predict_single_sample(
        code=code,
        language=language,
        model_config=model_config,
        model_path=FILE_PATHS["model"],
        importance_path=FILE_PATHS["feature_importance"],
        feature_columns=FEATURE_COLUMNS
    )

def detect_code_from_files(
    file_paths: List[str],
    model_type: str = "xgboost",
    json_output: bool = False
) -> List[Dict[str, Any]]:
    """
    Detect if code in files is AI-generated.
    
    Args:
        file_paths: List of files to analyze
        model_type: Type of model to use
        json_output: Whether to print JSON output
        
    Returns:
        List of dictionaries with detection results
    """
    # Get model config for the specified type
    model_config = MODEL_CONFIGS.get(model_type)
    if not model_config:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Process files
    results = process_file_batch(
        file_paths=file_paths,
        model_config=model_config,
        model_path=FILE_PATHS["model"],
        importance_path=FILE_PATHS["feature_importance"],
        feature_columns=FEATURE_COLUMNS
    )
    
    # Print results
    if json_output:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            verdict = "AI-GENERATED" if result['is_ai_generated'] else "HUMAN-WRITTEN"
            print(f"File: {result['file']}")
            print(f"AI probability: {result['probability']:.4f}")
            print(f"Verdict: {verdict}")
            print("-" * 50)
            
    return results

def main():
    """Main function to detect AI-generated code."""
    parser = argparse.ArgumentParser(description="Detect AI-generated code")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--files", nargs="+", help="Files to analyze")
    input_group.add_argument("-c", "--code", help="Code string to analyze")
    input_group.add_argument("--stdin", action="store_true", help="Read code from stdin")
    
    # Additional options
    parser.add_argument("-l", "--language", help="Specify language for code input")
    parser.add_argument("-m", "--model", default="xgboost", help="Model type to use")
    parser.add_argument("-j", "--json", action="store_true", help="Output in JSON format")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, 
                      help="Probability threshold for classification")
    
    args = parser.parse_args()
    
    try:
        # Process files
        if args.files:
            results = detect_code_from_files(args.files, args.model, args.json)
            
        # Process code string
        elif args.code:
            result = detect_code_from_string(args.code, args.language, args.model)
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                verdict = "AI-GENERATED" if result['probability'] > args.threshold else "HUMAN-WRITTEN"
                print(f"AI probability: {result['probability']:.4f}")
                print(f"Verdict: {verdict}")
                
        # Process stdin
        elif args.stdin:
            code = sys.stdin.read()
            result = detect_code_from_string(code, args.language, args.model)
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                verdict = "AI-GENERATED" if result['probability'] > args.threshold else "HUMAN-WRITTEN"
                print(f"AI probability: {result['probability']:.4f}")
                print(f"Verdict: {verdict}")
                
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 