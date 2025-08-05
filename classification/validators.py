#!/usr/bin/env python3
"""
Validation module for the Customer Service Classification System.

This module contains all input validation logic for queries, messages,
and other data structures used in the classification system.
"""

import re
from typing import Dict, List, Tuple, Any
from .config import ClassificationConfig, DEFAULT_CONFIG

class QueryValidator:
    """Handles validation of customer service queries."""
    
    def __init__(self, config: ClassificationConfig = None):
        """
        Initialize the validator with configuration.
        
        Args:
            config: Configuration object. If None, uses DEFAULT_CONFIG.
        """
        self.config = config or DEFAULT_CONFIG
    
    def validate_query(self, query: Any) -> None:
        """
        Validate the input query with comprehensive checks.
        
        Args:
            query: The query to validate
            
        Raises:
            ValueError: If query is invalid with specific reason
        """
        # Type validation
        if not isinstance(query, str):
            raise ValueError("Query must be a string, got type: " + str(type(query)))
        
        # Null/empty validation
        if query is None:
            raise ValueError("Query cannot be None")
        
        if len(query) == 0:
            raise ValueError("Query cannot be empty")
        
        # Whitespace validation
        stripped_query = query.strip()
        if len(stripped_query) == 0:
            raise ValueError("Query cannot contain only whitespace characters")
        
        # Length validation
        if len(query) > self.config.max_query_length:
            raise ValueError(
                f"Query is too long ({len(query)} characters). "
                f"Maximum {self.config.max_query_length} characters allowed."
            )
        
        if len(stripped_query) < self.config.min_query_length:
            raise ValueError(
                f"Query is too short. Please provide more details for accurate classification. "
                f"Minimum {self.config.min_query_length} characters required."
            )
        
        # Content validation
        self._validate_query_content(stripped_query)
    
    def _validate_query_content(self, query: str) -> None:
        """
        Validate the content of the query for potential issues.
        
        Args:
            query: The stripped query to validate
            
        Raises:
            ValueError: If query content is problematic
        """
        # Check for common problematic patterns
        problematic_patterns = [
            (r'^\s*[^\w\s]+\s*$', "Query contains only special characters"),
            (r'^\d+$', "Query contains only numbers"),
            (r'^[a-zA-Z]{1,2}$', "Query is too short (1-2 letters)"),
            (r'^\s*[^\w\s]+\s*$', "Query contains only punctuation"),
        ]
        
        for pattern, message in problematic_patterns:
            if re.match(pattern, query):
                raise ValueError(f"{message}. Please provide a meaningful query.")
        
        # Check for excessive repetition
        words = query.split()
        if len(words) > 1:
            word_counts = {}
            for word in words:
                word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1
            
            # Check if any word appears more than 50% of the time
            max_repetition = max(word_counts.values())
            if max_repetition > len(words) * 0.5:
                raise ValueError("Query contains excessive word repetition. Please provide a more varied query.")
        
        # Check for all caps (potential spam)
        if query.isupper() and len(query) > 10:
            raise ValueError("Query is in all caps. Please use normal case for better classification.")
    
    def sanitize_query(self, query: str) -> str:
        """
        Sanitize the query for safe processing.
        
        Args:
            query: The query to sanitize
            
        Returns:
            str: Sanitized query
        """
        # Remove excessive whitespace
        sanitized = ' '.join(query.split())
        
        # Truncate if too long (with safety margin)
        max_length = self.config.max_query_length - 100  # Leave some buffer
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        
        return sanitized
    
    def validate_and_sanitize_query(self, query: Any) -> Dict[str, Any]:
        """
        Validate and sanitize a query without making an API call.
        
        Args:
            query: The query to validate and sanitize
            
        Returns:
            dict: Validation result with sanitized query and validation info
        """
        try:
            # Validate input
            self.validate_query(query)
            
            # Sanitize query
            sanitized_query = self.sanitize_query(query)
            
            return {
                'is_valid': True,
                'original_query': query,
                'sanitized_query': sanitized_query,
                'validation_messages': ['Query passed all validation checks']
            }
            
        except ValueError as e:
            return {
                'is_valid': False,
                'original_query': query,
                'error_message': str(e),
                'validation_messages': [str(e)]
            }

class MessageValidator:
    """Handles validation of message structures for API calls."""
    
    def __init__(self, config: ClassificationConfig = None):
        """
        Initialize the validator with configuration.
        
        Args:
            config: Configuration object. If None, uses DEFAULT_CONFIG.
        """
        self.config = config or DEFAULT_CONFIG
    
    def validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Validate the messages array before sending to API.
        
        Args:
            messages: Messages array to validate
            
        Raises:
            ValueError: If messages are invalid
        """
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list")
        
        if len(messages) < 2:
            raise ValueError("Messages must contain at least system and user messages")
        
        required_roles = {'system', 'user'}
        found_roles = set()
        
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            
            if 'role' not in message or 'content' not in message:
                raise ValueError(f"Message {i} must contain 'role' and 'content' keys")
            
            role = message['role']
            content = message['content']
            
            if role not in ['system', 'user', 'assistant']:
                raise ValueError(f"Invalid role '{role}' in message {i}")
            
            if not isinstance(content, str):
                raise ValueError(f"Content in message {i} must be a string")
            
            if len(content.strip()) == 0:
                raise ValueError(f"Content in message {i} cannot be empty")
            
            found_roles.add(role)
        
        if not required_roles.issubset(found_roles):
            missing = required_roles - found_roles
            raise ValueError(f"Missing required roles: {missing}")

class ResponseValidator:
    """Handles validation of classification responses."""
    
    def __init__(self, config: ClassificationConfig = None):
        """
        Initialize the validator with configuration.
        
        Args:
            config: Configuration object. If None, uses DEFAULT_CONFIG.
        """
        self.config = config or DEFAULT_CONFIG
    
    def validate_classification_json(self, parsed_json: Dict[str, Any]) -> bool:
        """
        Validate that the parsed JSON has the expected classification structure.
        
        Args:
            parsed_json: Parsed JSON to validate
            
        Returns:
            bool: True if valid classification format
        """
        if not isinstance(parsed_json, dict):
            return False
        
        # Check for required fields
        required_fields = ['primary', 'secondary']
        if not all(field in parsed_json for field in required_fields):
            return False
        
        # Validate field types
        if not isinstance(parsed_json['primary'], str) or not isinstance(parsed_json['secondary'], str):
            return False
        
        # Validate against known categories
        valid_primary_categories = self.config.get_primary_categories()
        if parsed_json['primary'] not in valid_primary_categories:
            return False
        
        # Validate secondary category
        valid_secondary_categories = self.config.get_secondary_categories(parsed_json['primary'])
        if parsed_json['secondary'] not in valid_secondary_categories:
            return False
        
        return True 