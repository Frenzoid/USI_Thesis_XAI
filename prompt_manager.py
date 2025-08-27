import pandas as pd
from typing import Dict, List, Any, Optional
import re

from config import Config
from utils import setup_logging

logger = setup_logging("prompt_manager")

class PromptManager:
    """
    Centralized prompt management with mode-based prompting support.
    
    This class handles:
    1. Loading and validating prompt templates with mode support (zero-shot/few-shot)
    2. Generic dataset field mapping without hardcoded structures
    3. Mode validation and filtering
    4. Few-shot example generation from dataset rows
    5. Template concatenation for few-shot prompts
    6. Prompt-dataset compatibility checking
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
            dict: Prompt information including required placeholder count and mode
        """
        if prompt_name:
            # Information for specific prompt
            if prompt_name not in self.prompts_config:
                return {"error": f"Unknown prompt: {prompt_name}"}
            
            config = self.prompts_config[prompt_name]
            info = dict(config)
            
            # Count positional placeholders in template
            template = config['template']
            placeholder_count = template.count('{}')
            info['template_placeholder_count'] = placeholder_count
            
            # For few-shot prompts, also count placeholders in few_shot_example
            if config.get('mode') == 'few-shot' and 'few_shot_example' in config:
                few_shot_count = config['few_shot_example'].count('{}')
                info['few_shot_example_placeholder_count'] = few_shot_count
            
            return info
        else:
            # Information for all prompts
            all_info = {}
            for prompt_key in self.prompts_config:
                all_info[prompt_key] = self.get_prompt_info(prompt_key)
            return all_info
    
    def get_prompts_by_mode(self, mode: str) -> List[str]:
        """
        Get list of prompt names that match the specified mode.
        
        Args:
            mode: Mode to filter by ('zero-shot' or 'few-shot')
            
        Returns:
            list: Names of prompts that match the specified mode
        """
        matching_prompts = []
        for prompt_name, config in self.prompts_config.items():
            prompt_mode = config.get('mode', 'zero-shot')  # Default to zero-shot
            if prompt_mode == mode:
                matching_prompts.append(prompt_name)
        
        return matching_prompts
    
    # =============================================================================
    # MODE VALIDATION
    # =============================================================================
    
    def validate_mode_compatibility(self, prompt_name: str, requested_mode: str) -> bool:
        """
        Validate that a prompt's mode matches the requested mode.
        
        Args:
            prompt_name: Name of prompt to check
            requested_mode: Requested mode ('zero-shot' or 'few-shot')
            
        Returns:
            bool: True if modes match, False otherwise
        """
        if prompt_name not in self.prompts_config:
            logger.error(f"Unknown prompt: {prompt_name}")
            return False
        
        config = self.prompts_config[prompt_name]
        prompt_mode = config.get('mode', 'zero-shot')
        
        if prompt_mode != requested_mode:
            logger.error(f"Mode mismatch: prompt '{prompt_name}' is {prompt_mode}, but {requested_mode} was requested")
            return False
        
        return True
    
    def validate_few_shot_prompt_structure(self, prompt_name: str) -> bool:
        """
        Validate that a few-shot prompt has the required structure.
        
        Args:
            prompt_name: Name of prompt to validate
            
        Returns:
            bool: True if structure is valid, False otherwise
        """
        if prompt_name not in self.prompts_config:
            logger.error(f"Unknown prompt: {prompt_name}")
            return False
        
        config = self.prompts_config[prompt_name]
        
        if config.get('mode') != 'few-shot':
            return True  # Zero-shot prompts don't need few-shot validation
        
        # Few-shot prompts must have both template and few_shot_example
        if 'few_shot_example' not in config:
            logger.error(f"Few-shot prompt '{prompt_name}' missing 'few_shot_example' field")
            return False
        
        if 'template' not in config:
            logger.error(f"Few-shot prompt '{prompt_name}' missing 'template' field")
            return False
        
        return True
    
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
        Validate that the number of placeholders matches the dataset fields.
        
        For zero-shot: template placeholders = question_fields count
        For few-shot: template placeholders = question_fields count, 
                     few_shot_example placeholders = question_fields + answer_field count
        
        Args:
            prompt_name: Name of prompt template
            dataset_name: Name of dataset
            
        Returns:
            bool: True if field counts match, False otherwise
            
        Raises:
            ValueError: If field counts don't match
        """
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        prompt_config = self.prompts_config[prompt_name]
        dataset_config = self.datasets_config[dataset_name]
        
        question_fields = dataset_config.get('question_fields', [])
        question_field_count = len(question_fields)
        
        # Validate template placeholder count
        template = prompt_config['template']
        template_placeholder_count = template.count('{}')
        
        if template_placeholder_count != question_field_count:
            error_msg = (
                f"Template in prompt '{prompt_name}' has {template_placeholder_count} placeholders but "
                f"dataset '{dataset_name}' has {question_field_count} question fields. "
                f"Question fields: {question_fields}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # For few-shot prompts, validate few_shot_example placeholder count
        if prompt_config.get('mode') == 'few-shot':
            if 'few_shot_example' not in prompt_config:
                raise ValueError(f"Few-shot prompt '{prompt_name}' missing 'few_shot_example' field")
            
            few_shot_example = prompt_config['few_shot_example']
            few_shot_placeholder_count = few_shot_example.count('{}')
            expected_few_shot_count = question_field_count + 1  # question_fields + answer_field
            
            if few_shot_placeholder_count != expected_few_shot_count:
                error_msg = (
                    f"Few-shot example in prompt '{prompt_name}' has {few_shot_placeholder_count} placeholders but "
                    f"expected {expected_few_shot_count} (question fields + answer field). "
                    f"Question fields: {question_fields}, Answer field: {dataset_config.get('answer_field', 'N/A')}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        logger.debug(f"Prompt-dataset field count validation passed for '{prompt_name}' + '{dataset_name}'")
        return True
    
    def get_compatible_prompts(self, dataset_name: str, mode: str = None) -> List[str]:
        """
        Get list of prompts that are compatible with a specific dataset and optional mode.
        
        Args:
            dataset_name: Name of dataset to find prompts for
            mode: Optional mode filter ('zero-shot' or 'few-shot')
            
        Returns:
            list: Names of compatible prompts
        """
        compatible = []
        
        for prompt_name, config in self.prompts_config.items():
            # Check dataset compatibility
            if config.get('compatible_dataset') != dataset_name:
                continue
            
            # Check mode compatibility if specified
            if mode is not None:
                prompt_mode = config.get('mode', 'zero-shot')
                if prompt_mode != mode:
                    continue
            
            compatible.append(prompt_name)
        
        return compatible
    
    # =============================================================================
    # FEW-SHOT EXAMPLE GENERATION
    # =============================================================================
    
    def generate_few_shot_example(self, dataset: pd.DataFrame, dataset_name: str, 
                                 row_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a few-shot example from a specific or random row in the dataset.
        
        Args:
            dataset: Full dataset DataFrame
            dataset_name: Name of dataset (for field mapping)
            row_index: Specific row to use (0-based), or None for random
            
        Returns:
            dict: Contains 'question_values' (list) and 'answer_value' (str)
            
        Raises:
            ValueError: If dataset not supported or row_index out of bounds
        """
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Dataset '{dataset_name}' not supported for few-shot generation")
        
        dataset_config = self.datasets_config[dataset_name]
        question_fields = dataset_config.get('question_fields', [])
        answer_field = dataset_config.get('answer_field', '')
        
        if not answer_field:
            raise ValueError(f"Dataset '{dataset_name}' has no answer field for few-shot examples")
        
        # Validate row index
        if row_index is not None:
            if row_index < 0 or row_index >= len(dataset):
                raise ValueError(f"Row index {row_index} out of bounds for dataset with {len(dataset)} rows")
            row = dataset.iloc[row_index]
            logger.debug(f"Using specified row {row_index} for few-shot example")
        else:
            # Random row selection
            row = dataset.sample(n=1, random_state=Config.RANDOM_SEED).iloc[0]
            logger.debug("Using random row for few-shot example")
        
        # Extract question field values
        question_values = []
        for field in question_fields:
            value = str(row.get(field, 'N/A')) if field in row and not pd.isna(row[field]) else 'N/A'
            question_values.append(value)
        
        # Extract answer field value
        answer_value = str(row.get(answer_field, 'N/A')) if answer_field in row and not pd.isna(row[answer_field]) else 'N/A'
        
        # Skip examples marked as 'NA' or 'not annotatable'
        na_patterns = ['na', 'n/a', 'not applicable', 'not annotatable', '', 'none']
        if answer_value.lower().strip() in na_patterns:
            logger.warning("Selected few-shot example has NA answer, this may affect prompt quality")
        
        return {
            'question_values': question_values,
            'answer_value': answer_value,
            'row_index': row_index if row_index is not None else row.name
        }
    
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
            logger.debug(f"Formatted template for prompt '{prompt_name}' with {len(field_values)} field values")
            return formatted_prompt
            
        except Exception as e:
            logger.error(f"Template formatting error for prompt '{prompt_name}': {e}")
            raise
    
    def format_few_shot_example(self, prompt_name: str, question_values: List[str], answer_value: str) -> str:
        """
        Format a few-shot example template with provided values.
        
        Args:
            prompt_name: Name of prompt template
            question_values: List of question field values
            answer_value: Answer field value
            
        Returns:
            str: Formatted few-shot example
            
        Raises:
            ValueError: If prompt is not few-shot or field count mismatch
        """
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        prompt_config = self.prompts_config[prompt_name]
        
        if prompt_config.get('mode') != 'few-shot':
            raise ValueError(f"Prompt '{prompt_name}' is not a few-shot prompt")
        
        if 'few_shot_example' not in prompt_config:
            raise ValueError(f"Few-shot prompt '{prompt_name}' missing 'few_shot_example' field")
        
        few_shot_template = prompt_config['few_shot_example']
        
        # Combine question values and answer value
        all_values = question_values + [answer_value]
        
        # Count required placeholders
        placeholder_count = few_shot_template.count('{}')
        
        if len(all_values) != placeholder_count:
            logger.error(f"Few-shot field count mismatch for prompt '{prompt_name}': expected {placeholder_count}, got {len(all_values)}")
            raise ValueError(f"Expected {placeholder_count} field values for few-shot example, got {len(all_values)}")
        
        try:
            formatted_example = few_shot_template.format(*all_values)
            logger.debug(f"Formatted few-shot example for prompt '{prompt_name}'")
            return formatted_example
            
        except Exception as e:
            logger.error(f"Few-shot example formatting error for prompt '{prompt_name}': {e}")
            raise
    
    def prepare_prompt_for_row(self, prompt_name: str, row: pd.Series, dataset_name: str, 
                              mode: str, dataset: pd.DataFrame = None, 
                              few_shot_row: Optional[int] = None) -> str:
        """
        Prepare a complete formatted prompt for a specific dataset row with mode support.
        
        Args:
            prompt_name: Name of prompt template to use
            row: Dataset row containing the current data
            dataset_name: Name of the dataset (for field mapping)
            mode: Prompting mode ('zero-shot' or 'few-shot')
            dataset: Full dataset DataFrame (required for few-shot mode)
            few_shot_row: Specific row index for few-shot example (optional)
            
        Returns:
            str: Complete formatted prompt ready for model input
            
        Raises:
            ValueError: If validation fails or required parameters missing
        """
        # Validate basic parameters
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Dataset '{dataset_name}' not supported")
        
        # Validate mode compatibility
        if not self.validate_mode_compatibility(prompt_name, mode):
            raise ValueError(f"Prompt '{prompt_name}' is not compatible with mode '{mode}'")
        
        # Validate field count matches
        self.validate_prompt_field_count(prompt_name, dataset_name)
        
        # Get dataset configuration
        dataset_config = self.datasets_config[dataset_name]
        question_fields = dataset_config.get('question_fields', [])
        
        # Extract field values for current row
        current_question_values = []
        for field in question_fields:
            if field in row and not pd.isna(row[field]):
                current_question_values.append(str(row[field]))
            else:
                current_question_values.append("")
                logger.warning(f"Missing or null field '{field}' in current row, using empty string")
        
        if mode == 'zero-shot':
            # Simple template formatting for zero-shot
            return self.format_prompt(prompt_name, current_question_values)
        
        elif mode == 'few-shot':
            # Few-shot requires dataset for example generation
            if dataset is None:
                raise ValueError("Dataset is required for few-shot prompting")
            
            # Generate few-shot example
            example_data = self.generate_few_shot_example(dataset, dataset_name, few_shot_row)
            
            # Format few-shot example
            few_shot_example = self.format_few_shot_example(
                prompt_name, 
                example_data['question_values'], 
                example_data['answer_value']
            )
            
            # Format current prompt template
            current_template = self.format_prompt(prompt_name, current_question_values)
            
            # Concatenate few-shot example + template
            final_prompt = few_shot_example + current_template
            
            logger.debug(f"Created few-shot prompt with example from row {example_data['row_index']}")
            return final_prompt
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    # =============================================================================
    # PROMPT CATALOG AND STATISTICS
    # =============================================================================
    
    def list_prompts(self, mode: str = None) -> Dict[str, str]:
        """
        List available prompts with their descriptions, optionally filtered by mode.
        
        Args:
            mode: Optional mode filter ('zero-shot' or 'few-shot')
            
        Returns:
            dict: Mapping of prompt names to descriptions
        """
        prompts = {}
        for name, config in self.prompts_config.items():
            if mode is None or config.get('mode', 'zero-shot') == mode:
                description = config.get('description', 'No description')
                prompt_mode = config.get('mode', 'zero-shot')
                prompts[name] = f"{description} [{prompt_mode}]"
        
        return prompts