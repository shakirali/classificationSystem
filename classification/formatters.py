#!/usr/bin/env python3
"""
Formatters module for the Customer Service Classification System.

This module contains all output formatting logic for terminal display,
structured output, and user-friendly presentation of classification results.
"""

from typing import Dict, List, Any, Optional

class OutputFormatter:
    """Handles formatted output for the classification system."""
    
    # Color codes for terminal output
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'gray': '\033[90m'
    }
    
    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """
        Add color to text if terminal supports it.
        
        Args:
            text: Text to colorize
            color: Color name from COLORS dict
            
        Returns:
            str: Colorized text or original text if color not supported
        """
        try:
            return f"{cls.COLORS.get(color, '')}{text}{cls.COLORS['reset']}"
        except:
            return text
    
    @classmethod
    def print_header(cls, title: str) -> None:
        """
        Print a formatted header.
        
        Args:
            title: Header title to display
        """
        print("\n" + "=" * 60)
        print(cls.colorize(f"🤖 {title}", 'bold'))
        print("=" * 60)
    
    @classmethod
    def print_section(cls, title: str) -> None:
        """
        Print a section header.
        
        Args:
            title: Section title to display
        """
        print(f"\n{cls.colorize('─' * 40, 'gray')}")
        print(cls.colorize(f"📋 {title}", 'cyan'))
        print(f"{cls.colorize('─' * 40, 'gray')}")
    
    @classmethod
    def print_test_case(cls, test_num: int, query: str) -> None:
        """
        Print a test case header.
        
        Args:
            test_num: Test case number
            query: Query being tested
        """
        print(f"\n{cls.colorize(f'🧪 Test {test_num}', 'yellow')}")
        print(f"{cls.colorize('Query:', 'gray')} '{query}'")
    
    @classmethod
    def print_validation_result(cls, validation_result: Dict[str, Any]) -> None:
        """
        Print validation results.
        
        Args:
            validation_result: Validation result dictionary
        """
        if validation_result['is_valid']:
            print(f"{cls.colorize('✅ Validation Passed', 'green')}")
            print(f"{cls.colorize('📝 Sanitized:', 'gray')} '{validation_result['sanitized_query']}'")
        else:
            print(f"{cls.colorize('❌ Validation Failed', 'red')}")
            print(f"{cls.colorize('💬 Error:', 'red')} {validation_result['error_message']}")
    
    @classmethod
    def print_classification_result(cls, result: Dict[str, Any]) -> None:
        """
        Print classification results in a structured format.
        
        Args:
            result: Classification result dictionary
        """
        # Parsing Information
        cls.print_section("Parsing Information")
        parsing_info = result['parsing_info']
        
        json_status = "✅ JSON" if parsing_info['was_json'] else "📄 Text"
        success_status = "✅ Successful" if parsing_info['parsing_successful'] else "❌ Failed"
        
        print(f"{cls.colorize('Format:', 'gray')} {json_status}")
        print(f"{cls.colorize('Status:', 'gray')} {success_status}")
        
        if parsing_info['parsing_errors']:
            print(f"{cls.colorize('Errors:', 'red')} {', '.join(parsing_info['parsing_errors'])}")
        
        # Classification Results
        cls.print_section("Classification Results")
        response = result['response']
        
        if isinstance(response, dict):
            primary = response.get('primary', 'N/A')
            secondary = response.get('secondary', 'N/A')
            
            # Color code based on category
            primary_color = cls._get_category_color(primary)
            secondary_color = cls._get_category_color(secondary)
            
            print(f"{cls.colorize('🏷️  Primary Category:', 'gray')} {cls.colorize(primary, primary_color)}")
            print(f"{cls.colorize('🏷️  Secondary Category:', 'gray')} {cls.colorize(secondary, secondary_color)}")
            
            if 'parsing_method' in response:
                print(f"{cls.colorize('📝  Parsing Method:', 'gray')} {response['parsing_method']}")
        else:
            print(f"{cls.colorize('🏷️  Classification:', 'gray')} {response}")
        
        # Metadata
        cls.print_section("System Information")
        metadata = result['metadata']
        
        print(f"{cls.colorize('🤖 Model:', 'gray')} {metadata['model_used']}")
        print(f"{cls.colorize('⏱️  Timestamp:', 'gray')} {metadata['timestamp']}")
        print(f"{cls.colorize('🏁 Finish Reason:', 'gray')} {metadata['finish_reason']}")
        
        # Usage Information
        if result['usage']:
            usage = result['usage']
            print(f"{cls.colorize('📊 Token Usage:', 'gray')}")
            print(f"   {cls.colorize('Total Tokens:', 'gray')} {usage['total_tokens']}")
            print(f"   {cls.colorize('Prompt Tokens:', 'gray')} {usage.get('prompt_tokens', 'N/A')}")
            print(f"   {cls.colorize('Completion Tokens:', 'gray')} {usage.get('completion_tokens', 'N/A')}")
    
    @classmethod
    def print_error(cls, error_type: str, message: str, suggestions: Optional[List[str]] = None) -> None:
        """
        Print formatted error messages.
        
        Args:
            error_type: Type of error
            message: Error message
            suggestions: Optional list of suggestions
        """
        print(f"\n{cls.colorize('❌ ' + error_type, 'red')}")
        print(f"{cls.colorize('💬 Message:', 'red')} {message}")
        
        if suggestions:
            print(f"{cls.colorize('💡 Suggestions:', 'yellow')}")
            for suggestion in suggestions:
                print(f"   • {suggestion}")
    
    @classmethod
    def print_summary(cls, results: List[Dict[str, Any]]) -> None:
        """
        Print a summary of all test results.
        
        Args:
            results: List of test result dictionaries
        """
        cls.print_section("Test Summary")
        
        total_tests = len(results)
        successful_validations = sum(1 for r in results if r.get('validation_passed', False))
        successful_classifications = sum(1 for r in results if r.get('classification_successful', False))
        
        print(f"{cls.colorize('📊 Total Tests:', 'gray')} {total_tests}")
        print(f"{cls.colorize('✅ Valid Queries:', 'green')} {successful_validations}")
        print(f"{cls.colorize('🤖 Successful Classifications:', 'green')} {successful_classifications}")
        print(f"{cls.colorize('❌ Failed Tests:', 'red')} {total_tests - successful_classifications}")
    
    @classmethod
    def _get_category_color(cls, category: str) -> str:
        """
        Get color for different categories.
        
        Args:
            category: Category name
            
        Returns:
            str: Color name for the category
        """
        category_colors = {
            'Billing': 'red',
            'Technical Support': 'blue',
            'Account Management': 'green',
            'General Inquiry': 'magenta'
        }
        return category_colors.get(category, 'white')

class JSONFormatter:
    """Handles JSON formatting for API responses and data export."""
    
    @staticmethod
    def format_classification_result(result: Dict[str, Any], pretty: bool = True) -> str:
        """
        Format classification result as JSON.
        
        Args:
            result: Classification result dictionary
            pretty: Whether to format with indentation
            
        Returns:
            str: JSON formatted string
        """
        import json
        
        # Create a clean version for JSON output
        json_result = {
            'classification': result.get('response', {}),
            'metadata': result.get('metadata', {}),
            'usage': result.get('usage', {}),
            'parsing_info': result.get('parsing_info', {})
        }
        
        if pretty:
            return json.dumps(json_result, indent=2, ensure_ascii=False)
        else:
            return json.dumps(json_result, ensure_ascii=False)
    
    @staticmethod
    def format_batch_results(results: List[Dict[str, Any]], pretty: bool = True) -> str:
        """
        Format multiple classification results as JSON.
        
        Args:
            results: List of classification result dictionaries
            pretty: Whether to format with indentation
            
        Returns:
            str: JSON formatted string
        """
        import json
        
        batch_result = {
            'total_queries': len(results),
            'successful_classifications': sum(1 for r in results if r.get('classification_successful', False)),
            'failed_classifications': sum(1 for r in results if not r.get('classification_successful', False)),
            'results': results
        }
        
        if pretty:
            return json.dumps(batch_result, indent=2, ensure_ascii=False)
        else:
            return json.dumps(batch_result, ensure_ascii=False)

class CSVFormatter:
    """Handles CSV formatting for data export."""
    
    @staticmethod
    def format_classification_results(results: List[Dict[str, Any]], include_metadata: bool = False) -> str:
        """
        Format classification results as CSV.
        
        Args:
            results: List of classification result dictionaries
            include_metadata: Whether to include metadata columns
            
        Returns:
            str: CSV formatted string
        """
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        headers = ['query', 'primary_category', 'secondary_category', 'parsing_method']
        if include_metadata:
            headers.extend(['model', 'timestamp', 'total_tokens'])
        
        writer.writerow(headers)
        
        # Write data rows
        for result in results:
            if result.get('classification_successful', False):
                response = result.get('response', {})
                row = [
                    result.get('original_query', ''),
                    response.get('primary', ''),
                    response.get('secondary', ''),
                    response.get('parsing_method', '')
                ]
                
                if include_metadata:
                    metadata = result.get('metadata', {})
                    usage = result.get('usage', {})
                    row.extend([
                        metadata.get('model_used', ''),
                        metadata.get('timestamp', ''),
                        usage.get('total_tokens', '')
                    ])
                
                writer.writerow(row)
        
        return output.getvalue() 