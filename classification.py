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
from openai import OpenAI
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
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it to constructor.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = build_system_prompt()
    
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
        """
        return self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
    
    def classify_query(self, query):
        """
        Classify a customer service query.
        
        Args:
            query (str): The customer service query to classify
            
        Returns:
            dict: Classification result with 'response' and 'usage' keys
        """
        try:
            messages = self._build_messages(query)
            response = self._call_openai_api(messages)
            
            return {
                'response': response.choices[0].message.content,
                'usage': response.usage.dict() if response.usage else None
            }
        except Exception as e:
            return {
                'response': f"Error: {str(e)}",
                'usage': None
            }

def main():
    """Main function to demonstrate the classification system."""
    
    try:
        # Create classifier instance
        classifier = CustomerServiceClassifier()
        
        print("🤖 Customer Service Classification System")
        print("=" * 45)
        
        # Test query
        user_message = "I want you to delete my profile and all of my user data"
        
        # Classify the query
        result = classifier.classify_query(user_message)
        
        # Display results
        print(f"Query: {user_message}")
        print(f"Classification: {result['response']}")
        
        # Show usage info
        if result['usage']:
            print(f"\n📊 Tokens used: {result['usage']['total_tokens']}")
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Please add your API key to the .env file:")
        print("   OPENAI_API_KEY=your_api_key_here")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 