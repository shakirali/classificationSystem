#!/usr/bin/env python3
"""
Simple two-agent conversation using OpenAI API.
One agent says "hello", the other says "bye".
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_agent_response(client, system_prompt, user_message):
    """Create a response from an agent with specific personality."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=50,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def main():
    """Main function to run the two-agent conversation."""
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return
    
    # Create OpenAI client
    client = OpenAI(api_key=api_key)
    
    print("🤖 Two Agent Conversation")
    print("=" * 40)
    
    # Agent 1: Hello Agent
    hello_agent_prompt = "You are a friendly agent. Always respond with a cheerful greeting. Keep responses short and positive."
    
    # Agent 2: Bye Agent  
    bye_agent_prompt = "You are a polite agent. Always respond with a friendly goodbye. Keep responses short and courteous."
    
    # Conversation flow
    print("\n👋 Agent 1 (Hello Agent):")
    hello_response = create_agent_response(
        client, 
        hello_agent_prompt, 
        "Introduce yourself and say hello"
    )
    print(f"   {hello_response}")
    
    print("\n👋 Agent 2 (Bye Agent):")
    bye_response = create_agent_response(
        client, 
        bye_agent_prompt, 
        "Respond to the hello and say goodbye"
    )
    print(f"   {bye_response}")
    
    print("\n🔄 Let's have a longer conversation:")
    print("-" * 30)
    
    # Simulate a conversation
    conversation = [
        ("Hello Agent", hello_agent_prompt, "Start a conversation"),
        ("Bye Agent", bye_agent_prompt, "Respond to the hello"),
        ("Hello Agent", hello_agent_prompt, "Ask how they are doing"),
        ("Bye Agent", bye_agent_prompt, "Answer and say goodbye")
    ]
    
    for agent_name, prompt, message in conversation:
        print(f"\n{agent_name}:")
        response = create_agent_response(client, prompt, message)
        print(f"   {response}")
    
    print("\n✅ Conversation complete!")

if __name__ == "__main__":
    main() 