#!/usr/bin/env python3
"""
Customer Service Classification System Package.

This package provides a comprehensive system for classifying customer service
queries using OpenAI's GPT models. It includes validation, parsing, formatting,
and configuration management.

Main Components:
- CustomerServiceClassifier: Main classification class
- ClassificationConfig: Configuration management
- QueryValidator: Input validation
- ResponseParser: Response parsing strategies
- OutputFormatter: Terminal output formatting
"""

from .classifier import CustomerServiceClassifier
from .config import ClassificationConfig, DEFAULT_CONFIG, build_system_prompt, CATEGORIES
from .validators import QueryValidator, MessageValidator, ResponseValidator
from .parsers import (
    ResponseParser, 
    JSONParser, 
    StructuredTextParser, 
    JSONExtractorParser, 
    ResponseParserManager,
    parse_response
)
from .formatters import OutputFormatter, JSONFormatter, CSVFormatter

__version__ = "1.0.0"
__author__ = "Customer Service Classification System"
__description__ = "A comprehensive system for classifying customer service queries using OpenAI API"

# Main exports for easy access
__all__ = [
    # Main classifier
    'CustomerServiceClassifier',
    
    # Configuration
    'ClassificationConfig',
    'DEFAULT_CONFIG',
    'build_system_prompt',
    'CATEGORIES',
    
    # Validators
    'QueryValidator',
    'MessageValidator', 
    'ResponseValidator',
    
    # Parsers
    'ResponseParser',
    'JSONParser',
    'StructuredTextParser',
    'JSONExtractorParser',
    'ResponseParserManager',
    'parse_response',
    
    # Formatters
    'OutputFormatter',
    'JSONFormatter',
    'CSVFormatter',
]

# Convenience function for quick classification
def classify_query(query: str, api_key: str = None, config: ClassificationConfig = None) -> dict:
    """
    Quick classification function for single queries.
    
    Args:
        query: The customer service query to classify
        api_key: OpenAI API key (optional, will use environment variable if not provided)
        config: Configuration object (optional, will use default if not provided)
        
    Returns:
        dict: Classification result
        
    Raises:
        ValueError: If query is invalid
        OpenAIError: If API call fails
        RuntimeError: For other unexpected errors
    """
    classifier = CustomerServiceClassifier(api_key=api_key, config=config)
    return classifier.classify_query(query)

# Convenience function for batch classification
def classify_batch(queries: list, api_key: str = None, config: ClassificationConfig = None) -> list:
    """
    Batch classification function for multiple queries.
    
    Args:
        queries: List of customer service queries to classify
        api_key: OpenAI API key (optional, will use environment variable if not provided)
        config: Configuration object (optional, will use default if not provided)
        
    Returns:
        list: List of classification results
        
    Raises:
        ValueError: If any query is invalid
        OpenAIError: If API call fails
        RuntimeError: For other unexpected errors
    """
    classifier = CustomerServiceClassifier(api_key=api_key, config=config)
    results = []
    
    for query in queries:
        try:
            result = classifier.classify_query(query)
            results.append(result)
        except Exception as e:
            # Add error result for failed queries
            results.append({
                'original_query': query,
                'error': str(e),
                'classification_successful': False
            })
    
    return results 