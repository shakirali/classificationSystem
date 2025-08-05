#!/usr/bin/env python3
"""
Test script to verify modular imports work correctly.
"""

def test_imports():
    """Test that all modular imports work correctly."""
    
    print("🧪 Testing Modular Imports...")
    print("=" * 50)
    
    try:
        # Test main classifier import
        from classification import CustomerServiceClassifier
        print("✅ CustomerServiceClassifier imported successfully")
        
        # Test configuration import
        from classification import ClassificationConfig, DEFAULT_CONFIG
        print("✅ Configuration classes imported successfully")
        
        # Test validators import
        from classification import QueryValidator, MessageValidator, ResponseValidator
        print("✅ Validator classes imported successfully")
        
        # Test parsers import
        from classification import ResponseParser, JSONParser, parse_response
        print("✅ Parser classes imported successfully")
        
        # Test formatters import
        from classification import OutputFormatter, JSONFormatter, CSVFormatter
        print("✅ Formatter classes imported successfully")
        
        # Test convenience functions
        from classification import classify_query, classify_batch
        print("✅ Convenience functions imported successfully")
        
        # Test package metadata
        from classification import __version__, __author__, __description__
        print(f"✅ Package metadata: {__version__} by {__author__}")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the modular system."""
    
    print("\n🔧 Testing Basic Functionality...")
    print("=" * 50)
    
    try:
        from classification import CustomerServiceClassifier, ClassificationConfig
        
        # Test configuration creation
        config = ClassificationConfig(
            model_name="gpt-4o-mini",
            max_tokens=100,
            temperature=0.7
        )
        print("✅ Configuration created successfully")
        
        # Test classifier creation (without API key for testing)
        try:
            classifier = CustomerServiceClassifier(api_key="sk-test1234567890abcdef")
            print("✅ Classifier created successfully")
            
            # Test validation
            validation_result = classifier.validate_and_sanitize_query("test query")
            print("✅ Query validation works")
            
        except ValueError as e:
            if "Invalid API key" in str(e):
                print("✅ Classifier creation works (expected API key validation)")
            else:
                raise e
        
        print("\n🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    # Run import tests
    import_success = test_imports()
    
    # Run functionality tests
    func_success = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Import Tests: {'✅ PASSED' if import_success else '❌ FAILED'}")
    print(f"   Functionality Tests: {'✅ PASSED' if func_success else '❌ FAILED'}")
    
    if import_success and func_success:
        print("\n🎉 All tests passed! Modular structure is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.") 