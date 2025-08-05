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

def get_system_prompt():
    """Get the system prompt for classification."""
    return build_system_prompt()

def main():
    """Simple hello example using OpenAI API."""
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("💡 Please add your API key to the .env file:")
        print("   OPENAI_API_KEY=your_api_key_here")
        return
    
    try:
        # Create OpenAI client
        client = OpenAI(api_key=api_key)
        
        print("🤖 Hello OpenAI API!")
        print("=" * 30)
        
        user_message = f"""\
        I want you to delete my profile and all of my user data"""

        messages =  [  
        {'role':'system', 
        'content': get_system_prompt()},    
        {'role':'user', 
        'content': f"{delimiter}{user_message}{delimiter}"},  
        ] 

        # Simple chat completion
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        # Get the response
        message = response.choices[0].message.content
        print(f"AI Response: {message}")
        
        # Show usage info
        if response.usage:
            print(f"\n📊 Tokens used: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 