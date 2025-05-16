"""
Feature extraction utilities for code analysis.

This module contains classes and functions for extracting features from source code.
"""

import logging
import re
from typing import Dict, List, Optional

# Set up logging
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Class to extract features from source code for AI detection.
    
    Extracts various code quality and complexity metrics that can be used
    to train classifiers for AI-generated code detection.
    """
    
    @staticmethod
    def extract_basic_features(code: str) -> Dict[str, float]:
        """
        Extract basic features from code string.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary containing code features
        """
        if not code or not isinstance(code, str):
            logger.warning("Invalid code input for feature extraction")
            return FeatureExtractor._create_empty_features()
            
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Calculate whitespace ratio
        whitespace_ratio = sum(1 for c in code if c.isspace()) / max(1, len(code))
        
        # Calculate average line length
        avg_line_length = sum(len(line) for line in lines) / max(1, len(lines))
        
        # Calculate average identifier length
        identifiers = []
        for line in lines:
            tokens = line.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').split()
            identifiers.extend([token for token in tokens if token.isalnum() and not token.isdigit()])
        
        avg_identifier_length = sum(len(id) for id in identifiers) / max(1, len(identifiers))
        
        # Calculate empty lines density
        empty_lines_density = (len(lines) - len(non_empty_lines)) / max(1, len(lines))
        
        # Try to calculate function-related metrics if possible
        function_metrics = FeatureExtractor._extract_function_metrics(code, lines)
        
        # Combine all features
        return {
            'avgFunctionLength': function_metrics.get('avgFunctionLength', 0.0),
            'avgIdentifierLength': avg_identifier_length,
            'avgLineLength': avg_line_length,
            'emptyLinesDensity': empty_lines_density,
            'functionDefinitionDensity': function_metrics.get('functionDefinitionDensity', 0.0),
            'maintainabilityIndex': function_metrics.get('maintainabilityIndex', 50.0),  # Default mid-range value
            'maxDecisionTokens': function_metrics.get('maxDecisionTokens', 0.0),
            'whiteSpaceRatio': whitespace_ratio
        }
    
    @staticmethod
    def _extract_function_metrics(code: str, lines: List[str]) -> Dict[str, float]:
        """
        Extract function-related metrics from code.
        
        Args:
            code: Source code string
            lines: List of code lines
            
        Returns:
            Dictionary with function-related metrics
        """
        # Basic function detection regex patterns for common languages
        function_patterns = {
            'python': r'def\s+\w+\s*\(.*\):',
            'javascript': r'function\s+\w+\s*\(.*\)\s*{',
            'java': r'(?:public|private|protected|static|\s)+[\w<>[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:\{|throws)',
            'cpp': r'[\w<>[\]]+\s+(\w+)\s*\([^)]*\)\s*\{',
            'go': r'func\s+(\w+)\s*\([^)]*\)\s*(?:\([^)]*\))?\s*\{',
            'rust': r'fn\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*\{'
        }
        
        # Try to find functions using the patterns
        functions = []
        for pattern in function_patterns.values():
            functions.extend(re.findall(pattern, code))
            
        function_count = len(functions)
        total_lines = len(lines)
        
        # Calculate function metrics
        if function_count > 0 and total_lines > 0:
            # This is a simplified approximation - in a real system, 
            # we would need more sophisticated parsing to get exact function bounds
            avg_function_length = total_lines / function_count
            function_density = function_count / total_lines
        else:
            avg_function_length = 0.0
            function_density = 0.0
            
        # Count decision points (if, while, for, etc.)
        decision_pattern = r'\b(if|else|for|while|switch|case)\b'
        decision_tokens = len(re.findall(decision_pattern, code))
            
        return {
            'avgFunctionLength': avg_function_length,
            'functionDefinitionDensity': function_density,
            'maxDecisionTokens': float(decision_tokens),
            'maintainabilityIndex': FeatureExtractor._calculate_maintainability_index(
                total_lines, function_count, decision_tokens
            )
        }
    
    @staticmethod
    def _calculate_maintainability_index(lines: int, functions: int, decisions: int) -> float:
        """
        Calculate a simplified maintainability index.
        
        Args:
            lines: Total number of code lines
            functions: Number of functions
            decisions: Number of decision points
            
        Returns:
            Maintainability index (0-100 scale)
        """
        # This is a simplified version of maintainability index
        # A real implementation would include cyclomatic complexity, Halstead metrics, etc.
        
        # Handle division by zero
        if lines == 0:
            return 50.0  # Default mid-range value
            
        # More functions relative to size is good
        function_ratio = min(1.0, functions / max(1, lines / 20))
        
        # More decision points relative to size can indicate complexity
        decision_density = min(1.0, decisions / max(1, lines))
        
        # Simple formula: higher is better
        # Scale from 0-100 where higher means more maintainable
        raw_index = 100 * (0.5 * function_ratio - 0.2 * decision_density + 0.3)
        
        # Ensure result is in 0-100 range
        return max(0.0, min(100.0, raw_index))
    
    @staticmethod
    def _create_empty_features() -> Dict[str, float]:
        """
        Create a dictionary with default feature values.
        
        Returns:
            Dictionary with zero values for all features
        """
        return {
            'avgFunctionLength': 0.0,
            'avgIdentifierLength': 0.0,
            'avgLineLength': 0.0,
            'emptyLinesDensity': 0.0,
            'functionDefinitionDensity': 0.0,
            'maintainabilityIndex': 50.0,  # Default mid-range value
            'maxDecisionTokens': 0.0,
            'whiteSpaceRatio': 0.0
        }
        
    @staticmethod
    def detect_language(code: str) -> Optional[str]:
        """
        Detect programming language from code or file extension.
        
        Args:
            code: Source code string
            file_path: Optional path to the code file
            
        Returns:
            Detected language or None if unknown
        """

        if re.search(r'def\s+\w+\s*\(.*\):', code) or \
           re.search(r'import\s+\w+', code) or \
           re.search(r'from\s+\w+\s+import', code):
            return 'python'
        
        # Java
        elif re.search(r'public\s+class', code) or \
             re.search(r'public\s+static\s+void\s+main', code) or \
             re.search(r'import\s+java\.', code):
            return 'java'
        
        # C/C++
        elif re.search(r'#include\s+[<"].*[>"]', code) or \
             re.search(r'int\s+main\s*\(\s*(?:void|int\s+argc,\s*char\s*\*\s*argv\[\]|)\s*\)', code):
            return 'cpp'
        
        # JavaScript
        elif re.search(r'function\s+\w+\s*\(.*\)\s*{', code) or \
             re.search(r'const\s+\w+\s*=', code) or \
             re.search(r'let\s+\w+\s*=', code):
            return 'javascript'
        
        # Go
        elif re.search(r'package\s+\w+', code) or \
             re.search(r'func\s+\w+\s*\([^)]*\)', code):
            return 'go'
        
        # Rust
        elif re.search(r'fn\s+\w+\s*\([^)]*\)', code) or \
             re.search(r'use\s+\w+::', code):
            return 'rust'
        
        # No clear indicators
        return None 