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
    
    def _build_messages(self, query):
        """
        Build the messages array for the OpenAI API call.
        
        Args:
            query (str): The customer service query to classify
            
        Returns:
            list: Messages array for OpenAI API
        """
        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': f"{delimiter}{query}{delimiter}"}
        ]
    
    def _call_openai_api(self, messages):
        """
        Make the API call to OpenAI.
        
        Args:
            messages (list): Messages array for the API call
            
        Returns:
            dict: OpenAI API response
            
        Raises:
            OpenAIError: If API call fails
            RuntimeError: For other unexpected errors
        """
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
        Parse and validate the API response.
        
        Args:
            response: OpenAI API response object
            
        Returns:
            dict: Parsed response with content and usage
        """
        try:
            content = response.choices[0].message.content
            usage = response.usage.dict() if response.usage else None
            
            # Try to parse JSON response
            try:
                parsed_content = json.loads(content)
                return {
                    'response': parsed_content,
                    'usage': usage,
                    'raw_response': content
                }
            except json.JSONDecodeError:
                # If not JSON, return as string
                return {
                    'response': content,
                    'usage': usage,
                    'raw_response': content
                }
        except (AttributeError, IndexError) as e:
            raise RuntimeError(f"Invalid response format from OpenAI: {str(e)}")
    
    def classify_query(self, query):
        """
        Classify a customer service query.
        
        Args:
            query (str): The customer service query to classify
            
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
            messages = self._build_messages(sanitized_query)
            response = self._call_openai_api(messages)
            
            # Parse and return response
            result = self._parse_response(response)
            
            # Add validation info to result
            result['original_query'] = query
            result['sanitized_query'] = sanitized_query
            result['validation_passed'] = True
            
            return result
            
        except (ValueError, OpenAIError, RuntimeError):
            # Re-raise specific exceptions
            raise
        except Exception as e:
            # Catch any other unexpected errors
            raise RuntimeError(f"Unexpected error during classification: {str(e)}")
    
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
                    
                    # Display results
                    if isinstance(result['response'], dict):
                        print(f"🏷️  Primary Category: {result['response'].get('primary', 'N/A')}")
                        print(f"🏷️  Secondary Category: {result['response'].get('secondary', 'N/A')}")
                    else:
                        print(f"🏷️  Classification: {result['response']}")
                    
                    # Show usage info
                    if result['usage']:
                        print(f"📊 Tokens used: {result['usage']['total_tokens']}")
                        
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