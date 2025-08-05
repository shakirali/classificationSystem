#!/usr/bin/env python3
"""
Simple "Hello" sample using OpenAI API directly.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
        
        # Simple chat completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Try a different model
            messages=[
                {"role": "user", "content": "Say hello and introduce yourself briefly"}
            ],
            max_tokens=100,
            temperature=0.7
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