#!/usr/bin/env python3
"""
Unit tests and testing utilities for the Customer Service Classification System.
"""

import unittest
import json
import sys
import os
from unittest.mock import Mock, patch

# Add the current directory to the path to import the classification module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classification import CustomerServiceClassifier, CATEGORIES, MODEL_NAME

class TestMode:
    """Testing utilities and mock responses for the classification system."""
    
    # Mock responses for different scenarios
    MOCK_RESPONSES = {
        'billing_unsubscribe': {
            'primary': 'Billing',
            'secondary': 'Unsubscribe or upgrade'
        },
        'technical_troubleshooting': {
            'primary': 'Technical Support',
            'secondary': 'General troubleshooting'
        },
        'account_password': {
            'primary': 'Account Management',
            'secondary': 'Password reset'
        },
        'general_pricing': {
            'primary': 'General Inquiry',
            'secondary': 'Pricing'
        },
        'invalid_json': "This is not valid JSON",
        'malformed_json': '{"primary": "Billing", "secondary":}',
        'unknown_category': {
            'primary': 'Unknown Category',
            'secondary': 'Unknown Subcategory'
        }
    }
    
    @classmethod
    def get_mock_response(cls, scenario='billing_unsubscribe'):
        """Get a mock response for testing."""
        return cls.MOCK_RESPONSES.get(scenario, cls.MOCK_RESPONSES['billing_unsubscribe'])
    
    @classmethod
    def create_mock_openai_response(cls, content, usage=None):
        """Create a mock OpenAI response object."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = content
        mock_choice.finish_reason = 'stop'
        mock_response.choices = [mock_choice]
        
        if usage:
            mock_usage = Mock()
            mock_usage.dict.return_value = usage
            mock_response.usage = mock_usage
        else:
            mock_response.usage = None
        
        return mock_response
    
    @classmethod
    def get_test_queries(cls):
        """Get a comprehensive set of test queries."""
        return {
            'valid_queries': [
                "I want to cancel my subscription",
                "My password doesn't work",
                "How much does the premium plan cost?",
                "I can't log into my account",
                "What are the system requirements?",
                "I need help with billing",
                "How do I update my profile?",
                "Is there a free trial available?"
            ],
            'invalid_queries': [
                "",  # Empty
                "   ",  # Whitespace only
                "Hi",  # Too short
                "A" * 5000,  # Too long
                "HELLO THIS IS ALL CAPS",  # All caps
                "test test test test test",  # Repetition
                "12345",  # Only numbers
                "!!!",  # Only special characters
                None,  # None value
                123,  # Non-string
            ],
            'edge_cases': [
                "I want to delete my account and all data",  # Account deletion
                "The system is down and I can't access anything",  # System issues
                "I need to speak to a human representative",  # Human support
                "What's the difference between plan A and plan B?",  # Comparison
                "I'm getting an error code 404",  # Technical error
            ]
        }

class CustomerServiceClassifierTest(unittest.TestCase):
    """Unit tests for the CustomerServiceClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = CustomerServiceClassifier(api_key="sk-test1234567890abcdef")
    
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertIsNotNone(self.classifier)
        self.assertIsNotNone(self.classifier.system_prompt)
        # Check that the classifier has the expected attributes
        self.assertTrue(hasattr(self.classifier, 'client'))
        self.assertTrue(hasattr(self.classifier, 'api_key'))
    
    def test_validate_query_valid(self):
        """Test validation of valid queries."""
        valid_queries = [
            "This is a valid query",
            "Another valid query with more content",
            "Query with numbers 123 and symbols !@#"
        ]
        
        for query in valid_queries:
            with self.subTest(query=query):
                # Should not raise an exception
                self.classifier._validate_query(query)
    
    def test_validate_query_invalid(self):
        """Test validation of invalid queries."""
        invalid_queries = [
            ("", "Query cannot be empty"),
            ("   ", "Query cannot contain only whitespace characters"),
            ("Hi", "Query is too short"),
            ("A" * 5000, "Query is too long"),
            (None, "Query must be a string"),
            (123, "Query must be a string"),
        ]
        
        for query, expected_error in invalid_queries:
            with self.subTest(query=query):
                with self.assertRaises(ValueError) as context:
                    self.classifier._validate_query(query)
                self.assertIn(expected_error, str(context.exception))
    
    def test_sanitize_query(self):
        """Test query sanitization."""
        test_cases = [
            ("  hello world  ", "hello world"),
            ("\n\ttest query\n\t", "test query"),
            ("multiple    spaces", "multiple spaces"),
        ]
        
        for input_query, expected_output in test_cases:
            with self.subTest(input_query=input_query):
                result = self.classifier._sanitize_query(input_query)
                self.assertEqual(result, expected_output)
    
    def test_build_messages(self):
        """Test message building."""
        query = "test query"
        messages = self.classifier._build_messages(query)
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        self.assertIn(query, messages[1]['content'])
    
    def test_validate_classification_json(self):
        """Test JSON validation."""
        # Valid JSON
        valid_json = {
            'primary': 'Billing',
            'secondary': 'Unsubscribe or upgrade'
        }
        self.assertTrue(self.classifier._validate_classification_json(valid_json))
        
        # Invalid JSON - missing fields
        invalid_json = {'primary': 'Billing'}
        self.assertFalse(self.classifier._validate_classification_json(invalid_json))
        
        # Invalid JSON - wrong types
        invalid_json = {'primary': 123, 'secondary': 'test'}
        self.assertFalse(self.classifier._validate_classification_json(invalid_json))
        
        # Invalid JSON - unknown category
        invalid_json = {'primary': 'Unknown', 'secondary': 'test'}
        self.assertFalse(self.classifier._validate_classification_json(invalid_json))
    
    @patch('openai.OpenAI')
    def test_classify_query_success(self, mock_openai):
        """Test successful classification."""
        # Mock the OpenAI response
        mock_response = TestMode.create_mock_openai_response(
            json.dumps(TestMode.get_mock_response('billing_unsubscribe')),
            {'total_tokens': 50, 'prompt_tokens': 30, 'completion_tokens': 20}
        )
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        # Create a new classifier instance with the mocked client
        classifier = CustomerServiceClassifier(api_key="sk-test1234567890abcdef")
        classifier.client = mock_openai.return_value
        
        result = classifier.classify_query("I want to cancel my subscription")
        
        self.assertIn('response', result)
        self.assertIn('usage', result)
        self.assertIn('parsing_info', result)
        self.assertIn('metadata', result)
        self.assertTrue(result['parsing_info']['was_json'])
    
    @patch('openai.OpenAI')
    def test_classify_query_api_error(self, mock_openai):
        """Test classification with API error."""
        from openai import OpenAIError
        mock_openai.return_value.chat.completions.create.side_effect = OpenAIError("API Error")
        
        with self.assertRaises(OpenAIError):
            self.classifier.classify_query("test query")
    
    def test_validate_and_sanitize_query(self):
        """Test validation and sanitization without API call."""
        # Valid query
        result = self.classifier.validate_and_sanitize_query("valid query")
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['sanitized_query'], "valid query")
        
        # Invalid query
        result = self.classifier.validate_and_sanitize_query("")
        self.assertFalse(result['is_valid'])
        self.assertIn('error_message', result)
    
    def test_categories_structure(self):
        """Test that categories are properly structured."""
        self.assertIsInstance(CATEGORIES, dict)
        self.assertGreater(len(CATEGORIES), 0)
        
        for primary, secondaries in CATEGORIES.items():
            self.assertIsInstance(primary, str)
            self.assertIsInstance(secondaries, list)
            self.assertGreater(len(secondaries), 0)
            
            for secondary in secondaries:
                self.assertIsInstance(secondary, str)

def run_tests():
    """Run all tests for the classification system."""
    print("🧪 Running Customer Service Classification Tests...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(CustomerServiceClassifierTest)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Tests Run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"   • {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()

def demo_test_mode():
    """Demonstrate the testing capabilities."""
    print("🧪 Testing Mode Demonstration")
    print("=" * 50)
    
    # Create classifier
    classifier = CustomerServiceClassifier(api_key="sk-test1234567890abcdef")
    
    # Get test queries
    test_queries = TestMode.get_test_queries()
    
    print(f"\n📝 Valid Queries ({len(test_queries['valid_queries'])}):")
    for i, query in enumerate(test_queries['valid_queries'], 1):
        print(f"   {i}. {query}")
    
    print(f"\n❌ Invalid Queries ({len(test_queries['invalid_queries'])}):")
    for i, query in enumerate(test_queries['invalid_queries'], 1):
        print(f"   {i}. {repr(query)}")
    
    print(f"\n🔍 Edge Cases ({len(test_queries['edge_cases'])}):")
    for i, query in enumerate(test_queries['edge_cases'], 1):
        print(f"   {i}. {query}")
    
    print(f"\n🎭 Mock Responses Available:")
    for scenario in TestMode.MOCK_RESPONSES.keys():
        print(f"   • {scenario}")
    
    # Test validation
    print(f"\n✅ Validation Tests:")
    for query in test_queries['valid_queries'][:3]:  # Test first 3
        result = classifier.validate_and_sanitize_query(query)
        status = "✅ PASS" if result['is_valid'] else "❌ FAIL"
        print(f"   {status}: {query[:50]}...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_test_mode()
    else:
        success = run_tests()
        sys.exit(0 if success else 1) 