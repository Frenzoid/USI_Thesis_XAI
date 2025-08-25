import pandas as pd
from typing import Dict, List, Any
import re

from config import Config
from utils import setup_logging

logger = setup_logging("prompt_manager")

class PromptManager:
    """
    Centralized prompt management with JSON configuration support and generic field mapping.
    
    This class handles:
    1. Loading and validating prompt templates with positional placeholders
    2. Generic dataset field mapping without hardcoded structures
    3. Formatting templates using question fields in order
    4. Checking prompt-dataset compatibility and field count validation
    5. Generating few-shot examples
    6. Providing prompt statistics and information
    """
    
    def __init__(self):
        """Initialize prompt manager with configuration from JSON"""
        self.prompts_config = Config.load_prompts_config()
        self.datasets_config = Config.load_datasets_config()
        logger.info(f"PromptManager initialized with {len(self.prompts_config)} prompt templates")
    
    # =============================================================================
    # PROMPT INFORMATION AND INTROSPECTION
    # =============================================================================
    
    def get_prompt_info(self, prompt_name: str = None) -> Dict[str, Any]:
        """
        Get detailed information about prompt templates.
        
        Args:
            prompt_name: Specific prompt to get info for, or None for all prompts
            
        Returns:
            dict: Prompt information including required placeholder count
        """
        if prompt_name:
            # Information for specific prompt
            if prompt_name not in self.prompts_config:
                return {"error": f"Unknown prompt: {prompt_name}"}
            
            config = self.prompts_config[prompt_name]
            info = dict(config)
            
            # Count positional placeholders {}
            template = config['template']
            placeholder_count = template.count('{}')
            info['required_placeholder_count'] = placeholder_count
            
            return info
        else:
            # Information for all prompts
            all_info = {}
            for prompt_key in self.prompts_config:
                all_info[prompt_key] = self.get_prompt_info(prompt_key)
            return all_info
    
    # =============================================================================
    # COMPATIBILITY VALIDATION
    # =============================================================================
    
    def validate_prompt_dataset_compatibility(self, prompt_name: str, dataset_name: str) -> bool:
        """
        Check if a prompt is designed to work with a specific dataset.
        
        Args:
            prompt_name: Name of prompt template
            dataset_name: Name of dataset
            
        Returns:
            bool: True if compatible, False otherwise
        """
        if prompt_name not in self.prompts_config:
            logger.error(f"Unknown prompt: {prompt_name}")
            return False
        
        prompt_config = self.prompts_config[prompt_name]
        compatible_dataset = prompt_config.get('compatible_dataset', '')
        
        is_compatible = compatible_dataset == dataset_name
        
        if not is_compatible:
            logger.warning(f"Prompt '{prompt_name}' is designed for dataset '{compatible_dataset}', not '{dataset_name}'")
        
        return is_compatible
    
    def validate_prompt_field_count(self, prompt_name: str, dataset_name: str) -> bool:
        """
        Validate that the number of placeholders in the prompt matches the number of question fields.
        
        Args:
            prompt_name: Name of prompt template
            dataset_name: Name of dataset
            
        Returns:
            bool: True if placeholder count matches question field count
            
        Raises:
            ValueError: If field counts don't match
        """
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Count placeholders in prompt template
        template = self.prompts_config[prompt_name]['template']
        placeholder_count = template.count('{}')
        
        # Count question fields in dataset (only question fields, not answer field)
        dataset_config = self.datasets_config[dataset_name]
        question_fields = dataset_config.get('question_fields', [])
        question_field_count = len(question_fields)
        
        if placeholder_count != question_field_count:
            error_msg = (
                f"Prompt '{prompt_name}' has {placeholder_count} placeholders but "
                f"dataset '{dataset_name}' has {question_field_count} question fields. "
                f"Question fields: {question_fields}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Prompt-dataset field count validation passed: {placeholder_count} placeholders = {question_field_count} question fields")
        return True
    
    def get_compatible_prompts(self, dataset_name: str) -> List[str]:
        """
        Get list of prompts that are compatible with a specific dataset.
        
        Args:
            dataset_name: Name of dataset to find prompts for
            
        Returns:
            list: Names of compatible prompts
        """
        compatible = []
        
        for prompt_name, config in self.prompts_config.items():
            if config.get('compatible_dataset') == dataset_name:
                compatible.append(prompt_name)
        
        return compatible
    
    # =============================================================================
    # PROMPT FORMATTING AND GENERATION
    # =============================================================================
    
    def format_prompt(self, prompt_name: str, field_values: List[str]) -> str:
        """
        Format a prompt template with provided field values using positional placeholders.
        
        Args:
            prompt_name: Name of prompt template to format
            field_values: List of field values to substitute in order
            
        Returns:
            str: Formatted prompt ready for model input
            
        Raises:
            ValueError: If prompt unknown or field count mismatch
        """
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        prompt_config = self.prompts_config[prompt_name]
        template = prompt_config['template']
        
        # Count required placeholders
        placeholder_count = template.count('{}')
        
        # Check if we have the right number of values
        if len(field_values) != placeholder_count:
            logger.error(f"Field count mismatch for prompt '{prompt_name}': expected {placeholder_count}, got {len(field_values)}")
            raise ValueError(f"Expected {placeholder_count} field values, got {len(field_values)}")
        
        try:
            # Use Python's format method with positional arguments
            formatted_prompt = template.format(*field_values)
            logger.debug(f"Formatted prompt '{prompt_name}' with {len(field_values)} field values")
            return formatted_prompt
            
        except Exception as e:
            logger.error(f"Template formatting error for prompt '{prompt_name}': {e}")
            raise
    
    def prepare_prompt_for_row(self, prompt_name: str, row: pd.Series, dataset_name: str, 
                              additional_vars: Dict[str, Any] = None) -> str:
        """
        Prepare a formatted prompt for a specific dataset row using generic field mapping.
        
        This method extracts question fields from the dataset row in order and maps them
        to positional placeholders in the prompt template.
        
        Args:
            prompt_name: Name of prompt template to use
            row: Dataset row containing the data
            dataset_name: Name of the dataset (for field mapping)
            additional_vars: Extra variables (e.g., few-shot examples) - handled separately
            
        Returns:
            str: Formatted prompt ready for model input
            
        Raises:
            ValueError: If dataset not supported or field count mismatch
        """
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Dataset '{dataset_name}' not supported")
        
        # Validate compatibility (with warning, not hard failure)
        if not self.validate_prompt_dataset_compatibility(prompt_name, dataset_name):
            logger.warning(f"Using potentially incompatible prompt '{prompt_name}' with dataset '{dataset_name}'")
        
        # Validate field count matches
        self.validate_prompt_field_count(prompt_name, dataset_name)
        
        # Get question fields from dataset configuration
        dataset_config = self.datasets_config[dataset_name]
        question_fields = dataset_config.get('question_fields', [])
        
        # Extract field values in order
        field_values = []
        for field in question_fields:
            if field in row and not pd.isna(row[field]):
                field_values.append(str(row[field]))
            else:
                field_values.append("")  # Empty string for missing fields
                logger.warning(f"Missing or null field '{field}' in dataset row, using empty string")
        
        # Handle few-shot prompts specially by inserting examples at the beginning
        if 'few_shot' in prompt_name and additional_vars and 'few_shot_examples' in additional_vars:
            # For few-shot prompts, the first placeholder is for examples
            field_values = [additional_vars['few_shot_examples']] + field_values
        
        # Format the prompt with field values
        return self.format_prompt(prompt_name, field_values)
    
    # =============================================================================
    # FEW-SHOT EXAMPLE GENERATION
    # =============================================================================
    
    def generate_few_shot_examples(self, dataset: pd.DataFrame, dataset_name: str, 
                                 n_examples: int = 3, exclude_indices: List[int] = None) -> str:
        """
        Generate few-shot examples from a dataset for in-context learning.
        
        Creates formatted examples showing input-output pairs that help
        the model understand the expected task format and quality.
        
        Args:
            dataset: Full dataset to sample examples from
            dataset_name: Name of dataset (for field mapping)
            n_examples: Number of examples to generate
            exclude_indices: Row indices to exclude (e.g., current test sample)
            
        Returns:
            str: Formatted few-shot examples ready for inclusion in prompts
            
        Raises:
            ValueError: If dataset not supported for few-shot generation
        """
        logger.info(f"Generating {n_examples} few-shot examples for dataset {dataset_name}")
        
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Few-shot generation not implemented for dataset: {dataset_name}")
        
        dataset_config = self.datasets_config[dataset_name]
        question_fields = dataset_config.get('question_fields', [])
        answer_field = dataset_config.get('answer_field', '')
        
        # Filter out excluded indices (typically the current test sample)
        if exclude_indices:
            available_data = dataset[~dataset.index.isin(exclude_indices)]
        else:
            available_data = dataset
        
        # Ensure we have enough data
        if len(available_data) < n_examples:
            logger.warning(f"Not enough data for {n_examples} examples, using {len(available_data)}")
            n_examples = len(available_data)
        
        # Sample examples with fixed seed for reproducibility
        examples = available_data.sample(n=n_examples, random_state=Config.RANDOM_SEED)
        
        formatted_examples = []
        
        # Process each example
        for _, row in examples.iterrows():
            # Get question field values
            question_values = []
            for field in question_fields:
                value = str(row.get(field, 'N/A'))
                question_values.append(value)
            
            # Get the answer field value
            explanation = str(row.get(answer_field, 'N/A')) if answer_field else 'N/A'
            
            # Skip examples marked as 'NA' or 'not annotatable'
            na_patterns = ['na', 'n/a', 'not applicable', 'not annotatable', '', 'none']
            if explanation.lower().strip() in na_patterns:
                continue
            
            # Format example based on dataset structure - generic approach
            if dataset_name == 'gmeg':
                # GMEG has original and revised texts
                if len(question_values) >= 2:
                    example_text = f"Original: {question_values[0]}\nRevised: {question_values[1]}\nExplanation: {explanation}"
                else:
                    example_text = f"Text: {' | '.join(question_values)}\nExplanation: {explanation}"
            else:
                # Generic format for other datasets
                field_pairs = []
                for i, field in enumerate(question_fields):
                    if i < len(question_values):
                        field_pairs.append(f"{field}: {question_values[i]}")
                example_text = f"{' | '.join(field_pairs)}\nExplanation: {explanation}"
            
            formatted_examples.append(example_text)
        
        # If we don't have enough valid examples after filtering NA, use fallback
        if not formatted_examples and len(examples) > 0:
            row = examples.iloc[0]
            question_values = [str(row.get(field, 'N/A')) for field in question_fields]
            if dataset_name == 'gmeg' and len(question_values) >= 2:
                example_text = f"Original: {question_values[0]}\nRevised: {question_values[1]}\nExplanation: (Example explanation would go here)"
            else:
                field_pairs = [f"{field}: {question_values[i]}" for i, field in enumerate(question_fields) if i < len(question_values)]
                example_text = f"{' | '.join(field_pairs)}\nExplanation: (Example explanation would go here)"
            formatted_examples.append(example_text)
        
        # Join examples with separators
        result = "\n\n".join(formatted_examples[:n_examples])
        logger.debug(f"Generated few-shot examples with {len(formatted_examples)} examples")
        
        return result
    
    # =============================================================================
    # PROMPT CATALOG AND STATISTICS
    # =============================================================================
    
    def list_prompts(self) -> Dict[str, str]:
        """
        List all available prompts with their descriptions.
        
        Returns:
            dict: Mapping of prompt names to descriptions
        """
        return {name: config['description'] for name, config in self.prompts_config.items()}
    