#!/usr/bin/env python3
"""
Parsing module for the Customer Service Classification System.

This module contains all response parsing logic with different parsing strategies
for handling various response formats from the classification API.
"""

import json
import re
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from .config import ClassificationConfig, DEFAULT_CONFIG
from .validators import ResponseValidator

class ResponseParser(ABC):
    """Abstract base class for response parsers."""
    
    @abstractmethod
    def can_parse(self, content: str) -> bool:
        """Check if this parser can handle the given content."""
        pass
    
    @abstractmethod
    def parse(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse the content and return structured data."""
        pass

class JSONParser(ResponseParser):
    """Parser for JSON-formatted responses."""
    
    def __init__(self, validator: ResponseValidator = None):
        """
        Initialize the JSON parser.
        
        Args:
            validator: Response validator. If None, creates a new one.
        """
        self.validator = validator or ResponseValidator()
    
    def can_parse(self, content: str) -> bool:
        """Check if content is valid JSON."""
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    def parse(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON content.
        
        Args:
            content: JSON string to parse
            
        Returns:
            Parsed data if valid, None otherwise
        """
        try:
            parsed_json = json.loads(content)
            if self.validator.validate_classification_json(parsed_json):
                return {
                    'primary': parsed_json['primary'],
                    'secondary': parsed_json['secondary'],
                    'parsing_method': 'json'
                }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return None

class StructuredTextParser(ResponseParser):
    """Parser for structured text responses."""
    
    def __init__(self, validator: ResponseValidator = None):
        """
        Initialize the structured text parser.
        
        Args:
            validator: Response validator. If None, creates a new one.
        """
        self.validator = validator or ResponseValidator()
    
    def can_parse(self, content: str) -> bool:
        """Check if content contains structured text patterns."""
        patterns = [
            r'primary[:\s]+([^,\n]+)',
            r'secondary[:\s]+([^,\n]+)',
            r'category[:\s]+([^,\n]+)',
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def parse(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse structured text content.
        
        Args:
            content: Text content to parse
            
        Returns:
            Parsed data if valid, None otherwise
        """
        patterns = [
            r'primary[:\s]+([^,\n]+)',
            r'secondary[:\s]+([^,\n]+)',
            r'category[:\s]+([^,\n]+)',
        ]
        
        primary_match = None
        secondary_match = None
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                if 'primary' in pattern.lower():
                    primary_match = matches[0].strip()
                elif 'secondary' in pattern.lower():
                    secondary_match = matches[0].strip()
        
        if primary_match and secondary_match:
            parsed_data = {
                'primary': primary_match,
                'secondary': secondary_match,
                'parsing_method': 'structured_text'
            }
            
            if self.validator.validate_classification_json(parsed_data):
                return parsed_data
        
        return None

class JSONExtractorParser(ResponseParser):
    """Parser that extracts JSON from text that might contain additional content."""
    
    def __init__(self, validator: ResponseValidator = None):
        """
        Initialize the JSON extractor parser.
        
        Args:
            validator: Response validator. If None, creates a new one.
        """
        self.validator = validator or ResponseValidator()
    
    def can_parse(self, content: str) -> bool:
        """Check if content contains JSON-like patterns."""
        json_patterns = [
            r'\{[^{}]*"primary"[^{}]*"secondary"[^{}]*\}',
            r'\{[^{}]*"secondary"[^{}]*"primary"[^{}]*\}',
            r'\{[^{}]*\}',  # Any JSON object
        ]
        
        for pattern in json_patterns:
            if re.search(pattern, content, re.DOTALL):
                return True
        return False
    
    def parse(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from text content.
        
        Args:
            content: Text content that might contain JSON
            
        Returns:
            Parsed data if valid, None otherwise
        """
        json_patterns = [
            r'\{[^{}]*"primary"[^{}]*"secondary"[^{}]*\}',
            r'\{[^{}]*"secondary"[^{}]*"primary"[^{}]*\}',
            r'\{[^{}]*\}',  # Any JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                # Try to clean up the match
                cleaned = match.strip()
                if cleaned.startswith('{') and cleaned.endswith('}'):
                    try:
                        parsed_json = json.loads(cleaned)
                        if self.validator.validate_classification_json(parsed_json):
                            return {
                                'primary': parsed_json['primary'],
                                'secondary': parsed_json['secondary'],
                                'parsing_method': 'extracted_json'
                            }
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        return None

class ResponseParserManager:
    """Manages multiple parsing strategies for response handling."""
    
    def __init__(self, config: ClassificationConfig = None):
        """
        Initialize the parser manager.
        
        Args:
            config: Configuration object. If None, uses DEFAULT_CONFIG.
        """
        self.config = config or DEFAULT_CONFIG
        self.validator = ResponseValidator(config)
        
        # Initialize parsers in order of preference
        self.parsers = [
            JSONParser(self.validator),
            JSONExtractorParser(self.validator),
            StructuredTextParser(self.validator),
        ]
    
    def parse_content(self, content: str) -> Dict[str, Any]:
        """
        Parse content using multiple parsing strategies.
        
        Args:
            content: Raw content from API response
            
        Returns:
            dict: Parsing result with parsed content and metadata
        """
        parsing_result = {
            'parsed_content': content,
            'was_json': False,
            'parsing_successful': True,
            'parsing_errors': []
        }
        
        # Try each parser in order
        for parser in self.parsers:
            if parser.can_parse(content):
                try:
                    parsed_data = parser.parse(content)
                    if parsed_data:
                        parsing_result.update({
                            'parsed_content': parsed_data,
                            'was_json': isinstance(parser, (JSONParser, JSONExtractorParser)),
                            'parsing_successful': True
                        })
                        return parsing_result
                except Exception as e:
                    parsing_result['parsing_errors'].append(f"{parser.__class__.__name__} failed: {str(e)}")
        
        # If no parser succeeded, return fallback
        parsing_result.update({
            'parsed_content': content,
            'was_json': False,
            'parsing_successful': False,
            'parsing_errors': ['All parsing strategies failed, returning raw content']
        })
        
        return parsing_result

def parse_response(response: Any, config: ClassificationConfig = None) -> Dict[str, Any]:
    """
    Parse and validate the API response with comprehensive error handling.
    
    Args:
        response: OpenAI API response object
        config: Configuration object. If None, uses DEFAULT_CONFIG.
        
    Returns:
        dict: Parsed response with structured data and metadata
        
    Raises:
        RuntimeError: If response format is invalid
    """
    try:
        # Extract basic response data
        content = response.choices[0].message.content
        usage = response.usage.model_dump() if response.usage else None
        finish_reason = response.choices[0].finish_reason
        
        # Parse and validate the content
        parser_manager = ResponseParserManager(config)
        parsed_result = parser_manager.parse_content(content)
        
        # Build comprehensive response
        result = {
            'response': parsed_result['parsed_content'],
            'usage': usage,
            'raw_response': content,
            'parsing_info': {
                'was_json': parsed_result['was_json'],
                'parsing_successful': parsed_result['parsing_successful'],
                'parsing_errors': parsed_result['parsing_errors']
            },
            'metadata': {
                'finish_reason': finish_reason,
                'model_used': config.model_name if config else DEFAULT_CONFIG.model_name,
                'timestamp': _get_current_timestamp()
            }
        }
        
        return result
        
    except (AttributeError, IndexError) as e:
        raise RuntimeError(f"Invalid response format from OpenAI: {str(e)}")

def _get_current_timestamp() -> str:
    """
    Get current timestamp for response metadata.
    
    Returns:
        str: Formatted timestamp
    """
    from datetime import datetime
    return datetime.now().isoformat() 