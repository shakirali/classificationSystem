#!/usr/bin/env python3
"""
Configuration module for the Customer Service Classification System.

This module contains all configuration constants, settings, and environment
variable management for the classification system.
"""

import os
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

@dataclass
class ClassificationConfig:
    """Configuration class for the classification system."""
    
    # API Configuration
    delimiter: str = "####"
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 100
    temperature: float = 0.7
    
    # Validation Configuration
    max_query_length: int = 4000
    min_query_length: int = 3
    max_context_length: int = 1000
    
    # Categories
    categories: Dict[str, List[str]] = None
    
    def __post_init__(self):
        """Initialize categories if not provided."""
        if self.categories is None:
            self.categories = CATEGORIES
    
    @classmethod
    def from_env(cls) -> 'ClassificationConfig':
        """Create configuration from environment variables."""
        return cls(
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "100")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_query_length=int(os.getenv("MAX_QUERY_LENGTH", "4000")),
            min_query_length=int(os.getenv("MIN_QUERY_LENGTH", "3")),
            max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "1000"))
        )
    
    def get_primary_categories(self) -> List[str]:
        """Get list of primary categories."""
        return list(self.categories.keys())
    
    def get_secondary_categories(self, primary: str) -> List[str]:
        """Get secondary categories for a primary category."""
        return self.categories.get(primary, [])
    
    def validate_category(self, primary: str, secondary: str) -> bool:
        """Validate that a primary and secondary category combination is valid."""
        if primary not in self.categories:
            return False
        return secondary in self.categories[primary]

# Default configuration instance
DEFAULT_CONFIG = ClassificationConfig.from_env()

def build_system_prompt(config: ClassificationConfig = None) -> str:
    """
    Build the system prompt dynamically from the categories configuration.
    
    Args:
        config: Configuration object. If None, uses DEFAULT_CONFIG.
        
    Returns:
        str: Formatted system prompt
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    primary_categories = ", ".join(config.get_primary_categories())
    
    # Build secondary categories section
    secondary_sections = []
    for primary, secondaries in config.categories.items():
        secondary_list = "\n    ".join(secondaries)
        secondary_sections.append(f"{primary} secondary categories:\n    {secondary_list}")
    
    secondary_categories_text = "\n\n    ".join(secondary_sections)
    
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with \
    {config.delimiter} characters.
    Classify each query into a primary category \
    and a secondary category. 
    Provide your output in json format with the \
    keys: primary and secondary.

    Primary categories: {primary_categories}.

    {secondary_categories_text}
    """
    return system_message 