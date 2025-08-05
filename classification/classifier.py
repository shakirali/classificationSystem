#!/usr/bin/env python3
"""
Main classifier module for the Customer Service Classification System.

This module contains the CustomerServiceClassifier class with all the core
classification logic, API communication, and message building functionality.
"""

import os
from typing import Dict, List, Any, Optional
from openai import OpenAI, OpenAIError
from datetime import datetime

from .config import ClassificationConfig, DEFAULT_CONFIG, build_system_prompt
from .validators import QueryValidator, MessageValidator
from .parsers import parse_response

class CustomerServiceClassifier:
    """A class to handle customer service query classification using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[ClassificationConfig] = None):
        """
        Initialize the classifier with OpenAI API key and configuration.
        
        Args:
            api_key: OpenAI API key. If not provided, will try to get from environment.
            config: Configuration object. If None, uses DEFAULT_CONFIG.
            
        Raises:
            ValueError: If API key is not provided and not found in environment.
            RuntimeError: If initialization fails.
        """
        self.config = config or DEFAULT_CONFIG
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass it to constructor."
            )
        
        if not self.api_key.startswith('sk-'):
            raise ValueError("Invalid API key format. API key should start with 'sk-'")
        
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.system_prompt = build_system_prompt(self.config)
            
            # Initialize validators
            self.query_validator = QueryValidator(self.config)
            self.message_validator = MessageValidator(self.config)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _build_messages(self, query: str, include_context: bool = True, max_context_length: int = None) -> List[Dict[str, str]]:
        """
        Build the messages array for the OpenAI API call with enhanced features.
        
        Args:
            query: The customer service query to classify
            include_context: Whether to include additional context
            max_context_length: Maximum length for context information
            
        Returns:
            list: Messages array for OpenAI API
        """
        if max_context_length is None:
            max_context_length = self.config.max_context_length
            
        messages = []
        
        # Add system message
        system_message = self._build_system_message(include_context, max_context_length)
        messages.append({'role': 'system', 'content': system_message})
        
        # Add user message with proper formatting
        user_message = self._build_user_message(query)
        messages.append({'role': 'user', 'content': user_message})
        
        return messages
    
    def _build_system_message(self, include_context: bool = True, max_context_length: int = None) -> str:
        """
        Build the system message with optional context.
        
        Args:
            include_context: Whether to include additional context
            max_context_length: Maximum length for context information
            
        Returns:
            str: Formatted system message
        """
        if max_context_length is None:
            max_context_length = self.config.max_context_length
            
        base_prompt = self.system_prompt
        
        if include_context:
            context = self._get_classification_context()
            if context and len(context) <= max_context_length:
                base_prompt += f"\n\nAdditional Context:\n{context}"
        
        return base_prompt
    
    def _build_user_message(self, query: str) -> str:
        """
        Build the user message with proper formatting and delimiters.
        
        Args:
            query: The query to format
            
        Returns:
            str: Formatted user message
        """
        # Ensure proper delimiter formatting
        formatted_query = query.strip()
        
        # Add delimiters with proper spacing
        user_message = f"{self.config.delimiter}\n{formatted_query}\n{self.config.delimiter}"
        
        return user_message
    
    def _get_classification_context(self) -> str:
        """
        Get additional context for classification if needed.
        
        Returns:
            str: Additional context information or empty string
        """
        # This can be extended to include:
        # - User history
        # - Previous classifications
        # - System status
        # - Time-based context
        # - User preferences
        
        context_parts = []
        
        # Add timestamp context
        current_time = datetime.now()
        context_parts.append(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add model information
        context_parts.append(f"Model: {self.config.model_name}")
        
        # Add category information
        context_parts.append(f"Available categories: {', '.join(self.config.get_primary_categories())}")
        
        return "\n".join(context_parts)
    
    def _call_openai_api(self, messages: List[Dict[str, str]]) -> Any:
        """
        Make the API call to OpenAI.
        
        Args:
            messages: Messages array for the API call
            
        Returns:
            OpenAI API response
            
        Raises:
            ValueError: If messages are invalid
            OpenAIError: If API call fails
            RuntimeError: For other unexpected errors
        """
        # Validate messages before API call
        self.message_validator.validate_messages(messages)
        
        try:
            return self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
        except OpenAIError as e:
            # Handle specific OpenAI errors
            if "insufficient_quota" in str(e).lower():
                raise OpenAIError("API quota exceeded. Please check your billing and plan details.")
            elif "invalid_api_key" in str(e).lower():
                raise OpenAIError("Invalid API key. Please check your API key and try again.")
            elif "rate_limit" in str(e).lower():
                raise OpenAIError("Rate limit exceeded. Please wait a moment and try again.")
            else:
                raise OpenAIError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during API call: {str(e)}")
    
    def classify_query(self, query: str, include_context: bool = True) -> Dict[str, Any]:
        """
        Classify a customer service query.
        
        Args:
            query: The customer service query to classify
            include_context: Whether to include additional context in the request
            
        Returns:
            dict: Classification result with 'response', 'usage', and 'raw_response' keys
            
        Raises:
            ValueError: If query is invalid
            OpenAIError: If API call fails
            RuntimeError: For other unexpected errors
        """
        try:
            # Validate input
            self.query_validator.validate_query(query)
            
            # Sanitize query for processing
            sanitized_query = self.query_validator.sanitize_query(query)
            
            # Build messages and make API call
            messages = self._build_messages(sanitized_query, include_context)
            response = self._call_openai_api(messages)
            
            # Parse and return response
            result = parse_response(response, self.config)
            
            # Add validation info to result
            result['original_query'] = query
            result['sanitized_query'] = sanitized_query
            result['validation_passed'] = True
            result['context_included'] = include_context
            
            return result
            
        except (ValueError, OpenAIError, RuntimeError):
            # Re-raise specific exceptions
            raise
        except Exception as e:
            # Catch any other unexpected errors
            raise RuntimeError(f"Unexpected error during classification: {str(e)}")
    
    def build_messages_for_query(self, query: str, include_context: bool = True) -> Dict[str, Any]:
        """
        Build messages for a query without making an API call.
        Useful for debugging and testing.
        
        Args:
            query: The query to build messages for
            include_context: Whether to include additional context
            
        Returns:
            dict: Messages and validation information
        """
        try:
            # Validate and sanitize query
            self.query_validator.validate_query(query)
            sanitized_query = self.query_validator.sanitize_query(query)
            
            # Build messages
            messages = self._build_messages(sanitized_query, include_context)
            
            # Validate messages
            self.message_validator.validate_messages(messages)
            
            return {
                'success': True,
                'original_query': query,
                'sanitized_query': sanitized_query,
                'messages': messages,
                'context_included': include_context,
                'message_count': len(messages)
            }
            
        except Exception as e:
            return {
                'success': False,
                'original_query': query,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def validate_and_sanitize_query(self, query: Any) -> Dict[str, Any]:
        """
        Validate and sanitize a query without making an API call.
        
        Args:
            query: The query to validate and sanitize
            
        Returns:
            dict: Validation result with sanitized query and validation info
        """
        return self.query_validator.validate_and_sanitize_query(query)
    
    def get_config(self) -> ClassificationConfig:
        """
        Get the current configuration.
        
        Returns:
            ClassificationConfig: Current configuration object
        """
        return self.config
    
    def update_config(self, new_config: ClassificationConfig) -> None:
        """
        Update the configuration.
        
        Args:
            new_config: New configuration object
        """
        self.config = new_config
        self.system_prompt = build_system_prompt(self.config)
        
        # Update validators with new config
        self.query_validator = QueryValidator(self.config)
        self.message_validator = MessageValidator(self.config) 