#!/usr/bin/env python3
"""
Test script for the AI Code Detector API.

This script sends sample code snippets to the API and displays the responses,
testing both with and without explicit language specification.
"""

import json
import sys

import requests


def test_api(url="http://localhost:8000"):
    """Send test requests to the API and print the responses."""
    
    # Test cases for different languages
    test_cases = [
        {
            "name": "Python code with explicit language",
            "code": """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

if __name__ == "__main__":
    print(fibonacci(10))
""",
            "language": "python"
        },
        {
            "name": "Python code with auto-detection",
            "code": """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

if __name__ == "__main__":
    print(fibonacci(10))
"""
        },
        {
            "name": "Java code with auto-detection",
            "code": """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""
        },
        {
            "name": "C++ code with auto-detection",
            "code": """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""
        }
    ]
    
    print("=== AI Code Detector API Test ===\n")
    
    # Test each case
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['name']} ---")
        
        # Prepare request data
        request_data = {"code": test_case["code"]}
        if "language" in test_case:
            request_data["language"] = test_case["language"]
        
        # Send POST request to /detect endpoint
        try:
            response = requests.post(
                f"{url}/detect",
                json=request_data,
                timeout=30
            )
            
            # Print the status code and response
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\nAPI Response:")
                print(f"AI-generated probability: {result['probability']:.2%}")
                print(f"Is AI-generated: {result['is_ai_generated']}")
                print(f"Detected language: {result['language']}")
                print("\nExtracted features:")
                for feature, value in result['features'].items():
                    print(f"  {feature}: {value:.4f}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Request failed: {str(e)}")
    
    # Get model info
    try:
        info_response = requests.get(f"{url}/model-info")
        if info_response.status_code == 200:
            print("\n=== Model Information ===")
            info = info_response.json()
            print(f"Model type: {info['model_type']}")
            print(f"Encoder: {info['encoder']}")
            print(f"Max sequence length: {info['max_sequence_length']}")
            print(f"Feature columns: {', '.join(info['feature_columns'])}")
        else:
            print(f"Error getting model info: {info_response.text}")
    except Exception as e:
        print(f"Error getting model info: {str(e)}")

if __name__ == "__main__":
    # Use custom URL if provided as command line argument
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_api(url) 