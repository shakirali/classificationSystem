#!/usr/bin/env python3
"""
Customer Service Query Classification System using OpenAI API.

This script classifies customer service queries into primary and secondary categories
using OpenAI's GPT model. It can categorize queries into:
- Primary categories: Billing, Technical Support, Account Management, General Inquiry
- Secondary categories: Specific subcategories for each primary category

The system returns classification results in JSON format for easy integration
with customer service workflows.
"""

import os
import json
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
delimiter = "####"
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 100
TEMPERATURE = 0.7

# Classification categories
CATEGORIES = {
    "Billing": [
        "Unsubscribe or upgrade",
        "Add a payment method", 
        "Explanation for charge",
        "Dispute a charge"
    ],
    "Technical Support": [
        "General troubleshooting",
        "Device compatibility",
        "Software updates"
    ],
    "Account Management": [
        "Password reset",
        "Update personal information",
        "Close account",
        "Account security"
    ],
    "General Inquiry": [
        "Product information",
        "Pricing",
        "Feedback",
        "Speak to a human"
    ]
}

def build_system_prompt():
    """Build the system prompt dynamically from the categories configuration."""
    primary_categories = ", ".join(CATEGORIES.keys())
    
    # Build secondary categories section
    secondary_sections = []
    for primary, secondaries in CATEGORIES.items():
        secondary_list = "\n    ".join(secondaries)
        secondary_sections.append(f"{primary} secondary categories:\n    {secondary_list}")
    
    secondary_categories_text = "\n\n    ".join(secondary_sections)
    
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with \
    {delimiter} characters.
    Classify each query into a primary category \
    and a secondary category. 
    Provide your output in json format with the \
    keys: primary and secondary.

    Primary categories: {primary_categories}.

    {secondary_categories_text}
    """
    return system_message

class CustomerServiceClassifier:
    """A class to handle customer service query classification using OpenAI API."""
    
    def __init__(self, api_key=None):
        """
        Initialize the classifier with OpenAI API key.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to get from environment.
            
        Raises:
            ValueError: If API key is not provided and not found in environment.
        """
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
            self.system_prompt = build_system_prompt()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _validate_query(self, query):
        """
        Validate the input query with comprehensive checks.
        
        Args:
            query (str): The query to validate
            
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
        if len(query) > 4000:
            raise ValueError(f"Query is too long ({len(query)} characters). Maximum 4000 characters allowed.")
        
        if len(stripped_query) < 3:
            raise ValueError("Query is too short. Please provide more details for accurate classification.")
        
        # Content validation
        self._validate_query_content(stripped_query)
    
    def _validate_query_content(self, query):
        """
        Validate the content of the query for potential issues.
        
        Args:
            query (str): The stripped query to validate
            
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
        
        import re
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
    
    def _sanitize_query(self, query):
        """
        Sanitize the query for safe processing.
        
        Args:
            query (str): The query to sanitize
            
        Returns:
            str: Sanitized query
        """
        # Remove excessive whitespace
        sanitized = ' '.join(query.split())
        
        # Truncate if too long (with safety margin)
        if len(sanitized) > 3900:  # Leave some buffer
            sanitized = sanitized[:3900] + "..."
        
        return sanitized
    
    def _build_messages(self, query, include_context=True, max_context_length=1000):
        """
        Build the messages array for the OpenAI API call with enhanced features.
        
        Args:
            query (str): The customer service query to classify
            include_context (bool): Whether to include additional context
            max_context_length (int): Maximum length for context information
            
        Returns:
            list: Messages array for OpenAI API
        """
        messages = []
        
        # Add system message
        system_message = self._build_system_message(include_context, max_context_length)
        messages.append({'role': 'system', 'content': system_message})
        
        # Add user message with proper formatting
        user_message = self._build_user_message(query)
        messages.append({'role': 'user', 'content': user_message})
        
        return messages
    
    def _build_system_message(self, include_context=True, max_context_length=1000):
        """
        Build the system message with optional context.
        
        Args:
            include_context (bool): Whether to include additional context
            max_context_length (int): Maximum length for context information
            
        Returns:
            str: Formatted system message
        """
        base_prompt = self.system_prompt
        
        if include_context:
            context = self._get_classification_context()
            if context and len(context) <= max_context_length:
                base_prompt += f"\n\nAdditional Context:\n{context}"
        
        return base_prompt
    
    def _build_user_message(self, query):
        """
        Build the user message with proper formatting and delimiters.
        
        Args:
            query (str): The query to format
            
        Returns:
            str: Formatted user message
        """
        # Ensure proper delimiter formatting
        formatted_query = query.strip()
        
        # Add delimiters with proper spacing
        user_message = f"{delimiter}\n{formatted_query}\n{delimiter}"
        
        return user_message
    
    def _get_classification_context(self):
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
        from datetime import datetime
        current_time = datetime.now()
        context_parts.append(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add model information
        context_parts.append(f"Model: {MODEL_NAME}")
        
        # Add category information
        context_parts.append(f"Available categories: {', '.join(CATEGORIES.keys())}")
        
        return "\n".join(context_parts)
    
    def _validate_messages(self, messages):
        """
        Validate the messages array before sending to API.
        
        Args:
            messages (list): Messages array to validate
            
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
    
    def _call_openai_api(self, messages):
        """
        Make the API call to OpenAI.
        
        Args:
            messages (list): Messages array for the API call
            
        Returns:
            dict: OpenAI API response
            
        Raises:
            ValueError: If messages are invalid
            OpenAIError: If API call fails
            RuntimeError: For other unexpected errors
        """
        # Validate messages before API call
        self._validate_messages(messages)
        
        try:
            return self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
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
    
    def _parse_response(self, response):
        """
        Parse and validate the API response with comprehensive error handling.
        
        Args:
            response: OpenAI API response object
            
        Returns:
            dict: Parsed response with structured data and metadata
            
        Raises:
            RuntimeError: If response format is invalid
        """
        try:
            # Extract basic response data
            content = response.choices[0].message.content
            usage = response.usage.dict() if response.usage else None
            finish_reason = response.choices[0].finish_reason
            
            # Parse and validate the content
            parsed_result = self._parse_content(content)
            
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
                    'model_used': MODEL_NAME,
                    'timestamp': self._get_current_timestamp()
                }
            }
            
            return result
            
        except (AttributeError, IndexError) as e:
            raise RuntimeError(f"Invalid response format from OpenAI: {str(e)}")
    
    def _parse_content(self, content):
        """
        Parse the content with multiple parsing strategies.
        
        Args:
            content (str): Raw content from API response
            
        Returns:
            dict: Parsing result with parsed content and metadata
        """
        parsing_result = {
            'parsed_content': content,
            'was_json': False,
            'parsing_successful': True,
            'parsing_errors': []
        }
        
        # Strategy 1: Try direct JSON parsing
        try:
            parsed_json = json.loads(content)
            if self._validate_classification_json(parsed_json):
                parsing_result.update({
                    'parsed_content': parsed_json,
                    'was_json': True,
                    'parsing_successful': True
                })
                return parsing_result
        except json.JSONDecodeError as e:
            parsing_result['parsing_errors'].append(f"JSON parsing failed: {str(e)}")
        
        # Strategy 2: Try to extract JSON from text
        extracted_json = self._extract_json_from_text(content)
        if extracted_json:
            try:
                parsed_json = json.loads(extracted_json)
                if self._validate_classification_json(parsed_json):
                    parsing_result.update({
                        'parsed_content': parsed_json,
                        'was_json': True,
                        'parsing_successful': True
                    })
                    return parsing_result
            except json.JSONDecodeError as e:
                parsing_result['parsing_errors'].append(f"Extracted JSON parsing failed: {str(e)}")
        
        # Strategy 3: Try to parse structured text
        structured_result = self._parse_structured_text(content)
        if structured_result:
            parsing_result.update({
                'parsed_content': structured_result,
                'was_json': False,
                'parsing_successful': True
            })
            return parsing_result
        
        # Strategy 4: Fallback to raw content
        parsing_result.update({
            'parsed_content': content,
            'was_json': False,
            'parsing_successful': False,
            'parsing_errors': ['All parsing strategies failed, returning raw content']
        })
        
        return parsing_result
    
    def _validate_classification_json(self, parsed_json):
        """
        Validate that the parsed JSON has the expected classification structure.
        
        Args:
            parsed_json (dict): Parsed JSON to validate
            
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
        valid_primary_categories = list(CATEGORIES.keys())
        if parsed_json['primary'] not in valid_primary_categories:
            return False
        
        # Validate secondary category
        valid_secondary_categories = CATEGORIES.get(parsed_json['primary'], [])
        if parsed_json['secondary'] not in valid_secondary_categories:
            return False
        
        return True
    
    def _extract_json_from_text(self, text):
        """
        Extract JSON from text that might contain additional content.
        
        Args:
            text (str): Text that might contain JSON
            
        Returns:
            str: Extracted JSON string or None
        """
        import re
        
        # Look for JSON-like patterns
        json_patterns = [
            r'\{[^{}]*"primary"[^{}]*"secondary"[^{}]*\}',
            r'\{[^{}]*"secondary"[^{}]*"primary"[^{}]*\}',
            r'\{[^{}]*\}',  # Any JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                # Try to clean up the match
                cleaned = match.strip()
                if cleaned.startswith('{') and cleaned.endswith('}'):
                    return cleaned
        
        return None
    
    def _parse_structured_text(self, text):
        """
        Parse structured text that's not JSON but contains classification info.
        
        Args:
            text (str): Text to parse
            
        Returns:
            dict: Parsed classification or None
        """
        import re
        
        # Look for patterns like "Primary: X, Secondary: Y"
        patterns = [
            r'primary[:\s]+([^,\n]+)',
            r'secondary[:\s]+([^,\n]+)',
            r'category[:\s]+([^,\n]+)',
        ]
        
        primary_match = None
        secondary_match = None
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if 'primary' in pattern.lower():
                    primary_match = matches[0].strip()
                elif 'secondary' in pattern.lower():
                    secondary_match = matches[0].strip()
        
        if primary_match and secondary_match:
            return {
                'primary': primary_match,
                'secondary': secondary_match,
                'parsing_method': 'structured_text'
            }
        
        return None
    
    def _get_current_timestamp(self):
        """
        Get current timestamp for response metadata.
        
        Returns:
            str: Formatted timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def classify_query(self, query, include_context=True):
        """
        Classify a customer service query.
        
        Args:
            query (str): The customer service query to classify
            include_context (bool): Whether to include additional context in the request
            
        Returns:
            dict: Classification result with 'response', 'usage', and 'raw_response' keys
            
        Raises:
            ValueError: If query is invalid
            OpenAIError: If API call fails
            RuntimeError: For other unexpected errors
        """
        try:
            # Validate input
            self._validate_query(query)
            
            # Sanitize query for processing
            sanitized_query = self._sanitize_query(query)
            
            # Build messages and make API call
            messages = self._build_messages(sanitized_query, include_context)
            response = self._call_openai_api(messages)
            
            # Parse and return response
            result = self._parse_response(response)
            
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
    
    def build_messages_for_query(self, query, include_context=True):
        """
        Build messages for a query without making an API call.
        Useful for debugging and testing.
        
        Args:
            query (str): The query to build messages for
            include_context (bool): Whether to include additional context
            
        Returns:
            dict: Messages and validation information
        """
        try:
            # Validate and sanitize query
            self._validate_query(query)
            sanitized_query = self._sanitize_query(query)
            
            # Build messages
            messages = self._build_messages(sanitized_query, include_context)
            
            # Validate messages
            self._validate_messages(messages)
            
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
    
    def validate_and_sanitize_query(self, query):
        """
        Validate and sanitize a query without making an API call.
        
        Args:
            query (str): The query to validate and sanitize
            
        Returns:
            dict: Validation result with sanitized query and validation info
            
        Raises:
            ValueError: If query is invalid
        """
        try:
            # Validate input
            self._validate_query(query)
            
            # Sanitize query
            sanitized_query = self._sanitize_query(query)
            
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

def main():
    """Main function to demonstrate the classification system with validation."""
    
    try:
        # Create classifier instance
        classifier = CustomerServiceClassifier()
        
        print("🤖 Customer Service Classification System")
        print("=" * 45)
        
        # Test queries to demonstrate validation
        test_queries = [
            "I want you to delete my profile and all of my user data",  # Valid
            "   ",  # Invalid: only whitespace
            "Hi",  # Invalid: too short
            "HELLO THIS IS A TEST MESSAGE IN ALL CAPS",  # Invalid: all caps
            "test test test test test test test test test test",  # Invalid: repetition
            "12345",  # Invalid: only numbers
            "!!!",  # Invalid: only special characters
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: '{query}' ---")
            
            # First validate the query
            validation_result = classifier.validate_and_sanitize_query(query)
            
            if validation_result['is_valid']:
                print(f"✅ Validation passed")
                print(f"📝 Sanitized: '{validation_result['sanitized_query']}'")
                
                # Proceed with classification
                try:
                    result = classifier.classify_query(query)
                    
                    # Display results with enhanced parsing info
                    print(f"🔍 Parsing Info:")
                    print(f"   JSON Format: {result['parsing_info']['was_json']}")
                    print(f"   Parsing Successful: {result['parsing_info']['parsing_successful']}")
                    
                    if result['parsing_info']['parsing_errors']:
                        print(f"   Parsing Errors: {', '.join(result['parsing_info']['parsing_errors'])}")
                    
                    # Display classification results
                    if isinstance(result['response'], dict):
                        print(f"🏷️  Primary Category: {result['response'].get('primary', 'N/A')}")
                        print(f"🏷️  Secondary Category: {result['response'].get('secondary', 'N/A')}")
                        if 'parsing_method' in result['response']:
                            print(f"📝  Parsing Method: {result['response']['parsing_method']}")
                    else:
                        print(f"🏷️  Classification: {result['response']}")
                    
                    # Show metadata
                    print(f"📊 Metadata:")
                    print(f"   Model: {result['metadata']['model_used']}")
                    print(f"   Finish Reason: {result['metadata']['finish_reason']}")
                    print(f"   Timestamp: {result['metadata']['timestamp']}")
                    
                    # Show usage info
                    if result['usage']:
                        print(f"   Tokens Used: {result['usage']['total_tokens']}")
                        
                except (OpenAIError, RuntimeError) as e:
                    print(f"❌ Classification failed: {e}")
            else:
                print(f"❌ Validation failed: {validation_result['error_message']}")
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        if "API key" in str(e):
            print("💡 Please add your API key to the .env file:")
            print("   OPENAI_API_KEY=your_api_key_here")
        elif "Query" in str(e):
            print("💡 Please provide a valid query string")
    except OpenAIError as e:
        print(f"❌ OpenAI API Error: {e}")
        if "quota" in str(e).lower():
            print("💡 Please check your OpenAI billing and plan details")
        elif "rate limit" in str(e).lower():
            print("💡 Please wait a moment and try again")
    except RuntimeError as e:
        print(f"❌ Runtime Error: {e}")
        print("💡 Please check your internet connection and try again")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print("💡 Please contact support if this error persists")

if __name__ == "__main__":
    main() 