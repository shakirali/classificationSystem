#!/usr/bin/env python3
"""
Customer Service Classification Application.

This is the main application file that demonstrates the modular classification system.
It uses the refactored classification package with separated concerns.
"""

import sys
from typing import List, Dict, Any

# Import from the modular classification package
from classification import (
    CustomerServiceClassifier,
    ClassificationConfig,
    OutputFormatter,
    JSONFormatter,
    CSVFormatter
)
from openai import OpenAIError

def main():
    """Main function to demonstrate the modular classification system."""
    
    try:
        # Create classifier instance using the modular structure
        classifier = CustomerServiceClassifier()
        
        OutputFormatter.print_header("Modular Customer Service Classification System")
        
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
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            OutputFormatter.print_test_case(i, query)
            
            # First validate the query using the modular validator
            validation_result = classifier.validate_and_sanitize_query(query)
            OutputFormatter.print_validation_result(validation_result)
            
            result_info = {
                'test_number': i,
                'query': query,
                'validation_passed': validation_result['is_valid']
            }
            
            if validation_result['is_valid']:
                # Proceed with classification using the modular classifier
                try:
                    result = classifier.classify_query(query)
                    OutputFormatter.print_classification_result(result)
                    result_info['classification_successful'] = True
                    result_info['classification_result'] = result
                except (OpenAIError, RuntimeError) as e:
                    OutputFormatter.print_error("Classification Failed", str(e))
                    result_info['classification_successful'] = False
                    result_info['error'] = str(e)
            
            results.append(result_info)
        
        # Print summary
        OutputFormatter.print_summary(results)
        
        # Demonstrate JSON export
        print("\n" + "=" * 60)
        print("📄 JSON Export Example:")
        print("=" * 60)
        
        # Get successful results for JSON export
        successful_results = [r for r in results if r.get('classification_successful', False)]
        if successful_results:
            json_output = JSONFormatter.format_batch_results(successful_results)
            print(json_output[:500] + "..." if len(json_output) > 500 else json_output)
        
    except ValueError as e:
        suggestions = []
        if "API key" in str(e):
            suggestions = [
                "Add your API key to the .env file",
                "Format: OPENAI_API_KEY=your_api_key_here",
                "Make sure the API key starts with 'sk-'"
            ]
        elif "Query" in str(e):
            suggestions = [
                "Provide a valid query string",
                "Ensure the query is not empty",
                "Make sure the query is meaningful"
            ]
        
        OutputFormatter.print_error("Configuration Error", str(e), suggestions)
        
    except OpenAIError as e:
        suggestions = []
        if "quota" in str(e).lower():
            suggestions = [
                "Check your OpenAI billing and plan details",
                "Add payment method to your account",
                "Verify your account has sufficient credits"
            ]
        elif "rate limit" in str(e).lower():
            suggestions = [
                "Wait a moment and try again",
                "Reduce the frequency of requests",
                "Check your API usage limits"
            ]
        
        OutputFormatter.print_error("OpenAI API Error", str(e), suggestions)
        
    except RuntimeError as e:
        OutputFormatter.print_error("Runtime Error", str(e), [
            "Check your internet connection",
            "Verify the API endpoint is accessible",
            "Try again in a few moments"
        ])
        
    except Exception as e:
        OutputFormatter.print_error("Unexpected Error", str(e), [
            "Contact support if this error persists",
            "Check the system logs for more details",
            "Try restarting the application"
        ])

def demonstrate_modular_features():
    """Demonstrate the modular features of the classification system."""
    
    print("\n" + "=" * 60)
    print("🔧 Modular Features Demonstration")
    print("=" * 60)
    
    try:
        # Create a custom configuration
        custom_config = ClassificationConfig(
            model_name="gpt-4o-mini",
            max_tokens=150,
            temperature=0.5,
            max_query_length=3000
        )
        
        print(f"📋 Custom Configuration:")
        print(f"   Model: {custom_config.model_name}")
        print(f"   Max Tokens: {custom_config.max_tokens}")
        print(f"   Temperature: {custom_config.temperature}")
        print(f"   Max Query Length: {custom_config.max_query_length}")
        
        # Create classifier with custom config
        classifier = CustomerServiceClassifier(config=custom_config)
        
        # Demonstrate configuration update
        print(f"\n🔄 Updating Configuration...")
        new_config = ClassificationConfig(
            model_name="gpt-4o-mini",
            max_tokens=200,
            temperature=0.3
        )
        classifier.update_config(new_config)
        
        print(f"✅ Configuration Updated:")
        print(f"   New Max Tokens: {classifier.config.max_tokens}")
        print(f"   New Temperature: {classifier.config.temperature}")
        
        # Demonstrate message building without API call
        print(f"\n🔍 Message Building Demo:")
        message_info = classifier.build_messages_for_query("I need help with billing")
        if message_info['success']:
            print(f"   Messages Built: {message_info['message_count']}")
            print(f"   Context Included: {message_info['context_included']}")
            print(f"   Sanitized Query: '{message_info['sanitized_query']}'")
        
    except Exception as e:
        print(f"❌ Error in modular features demo: {e}")

if __name__ == "__main__":
    # Run main demonstration
    main()
    
    # Run modular features demonstration
    demonstrate_modular_features() 