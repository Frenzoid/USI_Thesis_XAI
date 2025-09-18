import pandas as pd
from typing import Dict, List, Any, Optional, Union
import random

from config import Config
from utils import setup_logging

logger = setup_logging("prompt_manager")

class PromptManager:
    """
    Centralized prompt management with mode-based prompting support and JSON field path resolution.
    
    Handles:
    - Loading prompt templates with mode support (zero-shot/few-shot)
    - Generic dataset field mapping using nested configuration structure
    - JSON field path resolution (e.g., "context.questions[0]")
    - Few-shot example generation with field paths
    - Template formatting and concatenation
    """
    
    def __init__(self):
        """Initialize prompt manager with configuration from JSON"""
        self.prompts_config = Config.load_prompts_config()
        self.setups_config = Config.load_setups_config()
        logger.info(f"PromptManager initialized with {len(self.prompts_config)} prompt templates")
    
    # =============================================================================
    # FIELD PATH RESOLUTION (delegated to DatasetManager)
    # =============================================================================
    
    def resolve_field_path(self, data: Union[Dict, pd.Series, Any], path: str) -> Any:
        """
        Resolve a nested JSON path like 'context.questions[0].text' against data structure.
        This method delegates to DatasetManager's implementation for consistency.
        
        Args:
            data: The data structure to navigate (dict, pd.Series, or primitive)
            path: The field path to resolve
            
        Returns:
            The value at the specified path, or None if path doesn't exist
        """
        # Import here to avoid circular imports
        from dataset_manager import DatasetManager
        
        # Create temporary DatasetManager instance for field resolution
        temp_manager = DatasetManager()
        
        # Convert pandas Series to dict if needed
        if isinstance(data, pd.Series):
            data_dict = data.to_dict()
        else:
            data_dict = data
        
        return temp_manager.resolve_field_path(data_dict, path)
    
    def extract_field_values(self, row: Union[Dict, pd.Series], field_paths: List[str]) -> List[str]:
        """
        Extract values from multiple field paths in a data row.
        
        Args:
            row: Data row (dict for JSON data, pd.Series for CSV/Parquet data)
            field_paths: List of field paths to extract
            
        Returns:
            List of string values extracted from the specified paths
        """
        values = []
        
        for path in field_paths:
            try:
                value = self.resolve_field_path(row, path)
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    values.append("")
                else:
                    values.append(str(value))
            except Exception as e:
                logger.warning(f"Error extracting field path '{path}': {e}")
                values.append("")
        
        return values
    
    # =============================================================================
    # PROMPT MANAGEMENT
    # =============================================================================
    
    def get_prompts_by_mode(self, mode: str) -> List[str]:
        """Get list of prompt names that match the specified mode"""
        matching_prompts = []
        for prompt_name, config in self.prompts_config.items():
            prompt_mode = config.get('mode', 'zero-shot')
            if prompt_mode == mode:
                matching_prompts.append(prompt_name)
        return matching_prompts
    
    def get_compatible_prompts(self, setup_name: str, mode: str = None) -> List[str]:
        """Get prompts compatible with a setup and optional mode"""
        compatible = []
        
        for prompt_name, config in self.prompts_config.items():
            # Check setup compatibility
            if config.get('compatible_setup') != setup_name:
                continue
            
            # Check mode compatibility if specified
            if mode is not None:
                prompt_mode = config.get('mode', 'zero-shot')
                if prompt_mode != mode:
                    continue
            
            compatible.append(prompt_name)
        
        return compatible
    
    # =============================================================================
    # FEW-SHOT EXAMPLE GENERATION (updated for JSON field paths)
    # =============================================================================
    
    def generate_few_shot_example(self, dataset: pd.DataFrame, setup_name: str, 
                                 row_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a few-shot example from dataset row using field path resolution.
        
        Args:
            dataset: Dataset DataFrame
            setup_name: Name of setup for field configuration
            row_index: Optional specific row to use (if None, uses random)
            
        Returns:
            Dict containing question_values, answer_value, and row_index
        """
        if setup_name not in self.setups_config:
            raise ValueError(f"Setup '{setup_name}' not found in configuration")
        
        setup_config = self.setups_config[setup_name]
        
        if 'prompt_fields' not in setup_config:
            raise ValueError(f"Setup '{setup_name}' missing required 'prompt_fields' configuration")
        
        prompt_fields_config = setup_config['prompt_fields']
        
        if 'question_fields' not in prompt_fields_config:
            raise ValueError(f"Setup '{setup_name}' missing 'question_fields' in prompt_fields configuration")
        
        if 'answer_field' not in prompt_fields_config:
            raise ValueError(f"Setup '{setup_name}' missing 'answer_field' in prompt_fields configuration")
        
        question_field_paths = prompt_fields_config['question_fields']
        answer_field_path = prompt_fields_config['answer_field']
        
        if not answer_field_path:
            raise ValueError(f"Setup '{setup_name}' has empty answer_field - required for few-shot examples")
        
        # Select row
        if row_index is not None:
            if row_index < 0 or row_index >= len(dataset):
                raise ValueError(f"Row index {row_index} out of bounds for dataset with {len(dataset)} rows")
            row = dataset.iloc[row_index]
            logger.debug(f"Using specified row {row_index} for few-shot example")
        else:
            row = dataset.sample(n=1, random_state=Config.RANDOM_SEED).iloc[0]
            logger.debug("Using random row for few-shot example")
        
        # Extract question values using field paths
        question_values = self.extract_field_values(row, question_field_paths)
        
        # Extract answer value using field path
        answer_value = self.resolve_field_path(row, answer_field_path)
        if answer_value is None or (isinstance(answer_value, float) and pd.isna(answer_value)):
            answer_value = 'N/A'
        else:
            answer_value = str(answer_value)
        
        # Warn about NA answers
        if answer_value.lower().strip() in ['na', 'n/a', 'not applicable', 'not annotatable', '', 'none']:
            logger.warning("Selected few-shot example has NA answer, may affect prompt quality")
        
        return {
            'question_values': question_values,
            'answer_value': answer_value,
            'row_index': row_index if row_index is not None else row.name
        }
    
    # =============================================================================
    # TEMPLATE FORMATTING
    # =============================================================================
    
    def format_prompt(self, prompt_name: str, field_values: List[str]) -> str:
        """Format a prompt template with field values"""
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        prompt_config = self.prompts_config[prompt_name]
        
        if 'template' not in prompt_config:
            raise ValueError(f"Prompt '{prompt_name}' missing required 'template' field")
        
        template = prompt_config['template']
        
        # Check field count
        placeholder_count = template.count('{}')
        if len(field_values) != placeholder_count:
            raise ValueError(f"Expected {placeholder_count} field values for prompt '{prompt_name}', got {len(field_values)}")
        
        try:
            formatted_prompt = template.format(*field_values)
            logger.debug(f"Formatted template for prompt '{prompt_name}'")
            return formatted_prompt
        except Exception as e:
            logger.error(f"Template formatting error for prompt '{prompt_name}': {e}")
            raise
    
    def format_few_shot_example(self, prompt_name: str, question_values: List[str], answer_value: str) -> str:
        """Format a few-shot example template"""
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        prompt_config = self.prompts_config[prompt_name]
        
        if prompt_config.get('mode') != 'few-shot':
            raise ValueError(f"Prompt '{prompt_name}' is not a few-shot prompt")
        
        if 'few_shot_example' not in prompt_config:
            raise ValueError(f"Few-shot prompt '{prompt_name}' missing required 'few_shot_example' field")
        
        few_shot_template = prompt_config['few_shot_example']
        all_values = question_values + [answer_value]
        
        # Check field count
        placeholder_count = few_shot_template.count('{}')
        if len(all_values) != placeholder_count:
            raise ValueError(f"Expected {placeholder_count} field values for few-shot example in prompt '{prompt_name}', got {len(all_values)}")
        
        try:
            formatted_example = few_shot_template.format(*all_values)
            logger.debug(f"Formatted few-shot example for prompt '{prompt_name}'")
            return formatted_example
        except Exception as e:
            logger.error(f"Few-shot example formatting error for prompt '{prompt_name}': {e}")
            raise
    
    # =============================================================================
    # COMPLETE PROMPT PREPARATION (updated for JSON field paths)
    # =============================================================================
    
    def prepare_prompt_for_row(self, prompt_name: str, row: Union[pd.Series, Dict], setup_name: str, 
                              mode: str, dataset: pd.DataFrame = None, 
                              few_shot_row: Optional[int] = None) -> str:
        """
        Prepare a complete formatted prompt for a dataset row using field path resolution.
        
        Args:
            prompt_name: Name of prompt template to use
            row: Dataset row (pd.Series for CSV/Parquet, dict for JSON/JSONL)
            setup_name: Name of setup for field configuration
            mode: Prompting mode ('zero-shot' or 'few-shot')
            dataset: Full dataset (required for few-shot mode)
            few_shot_row: Specific row index for few-shot example
            
        Returns:
            str: Complete formatted prompt ready for model inference
        """
        
        # Basic validation
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        if setup_name not in self.setups_config:
            raise ValueError(f"Unknown setup: {setup_name}")
        
        # Get configurations
        prompt_config = self.prompts_config[prompt_name]
        setup_config = self.setups_config[setup_name]
        
        # Validate prompt has required fields
        if 'compatible_setup' not in prompt_config:
            raise ValueError(f"Prompt '{prompt_name}' missing required 'compatible_setup' field")
        
        if 'mode' not in prompt_config:
            raise ValueError(f"Prompt '{prompt_name}' missing required 'mode' field")
        
        # Validate setup has required nested structure
        if 'prompt_fields' not in setup_config:
            raise ValueError(f"Setup '{setup_name}' missing required 'prompt_fields' configuration")
        
        prompt_fields_config = setup_config['prompt_fields']
        
        if 'question_fields' not in prompt_fields_config:
            raise ValueError(f"Setup '{setup_name}' missing 'question_fields' in prompt_fields configuration")
        
        # Validate compatibility
        if prompt_config['compatible_setup'] != setup_name:
            raise ValueError(f"Prompt '{prompt_name}' is compatible with setup '{prompt_config['compatible_setup']}', not '{setup_name}'")
        
        if prompt_config['mode'] != mode:
            raise ValueError(f"Prompt '{prompt_name}' is configured for '{prompt_config['mode']}' mode, but '{mode}' mode was requested")
        
        # Extract current row values using field path resolution
        question_field_paths = prompt_fields_config['question_fields']
        current_question_values = self.extract_field_values(row, question_field_paths)
        
        # Log missing fields for debugging
        for i, (path, value) in enumerate(zip(question_field_paths, current_question_values)):
            if not value:
                logger.warning(f"Empty value for field path '{path}' in row")
        
        if mode == 'zero-shot':
            # Simple template formatting
            return self.format_prompt(prompt_name, current_question_values)
        
        elif mode == 'few-shot':
            # Require dataset for example generation
            if dataset is None:
                raise ValueError("Dataset is required for few-shot prompting")
            
            # Generate few-shot example
            example_data = self.generate_few_shot_example(dataset, setup_name, few_shot_row)
            
            # Format few-shot example
            few_shot_example = self.format_few_shot_example(
                prompt_name, 
                example_data['question_values'], 
                example_data['answer_value']
            )
            
            # Format current prompt
            current_template = self.format_prompt(prompt_name, current_question_values)
            
            # Concatenate: example + current prompt
            final_prompt = few_shot_example + current_template
            
            logger.debug(f"Created few-shot prompt with example from row {example_data['row_index']}")
            return final_prompt
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes are 'zero-shot' and 'few-shot'")
    
    # =============================================================================
    # VALIDATION AND DEBUGGING UTILITIES
    # =============================================================================
    
    def validate_prompt_field_paths(self, prompt_name: str, setup_name: str, sample_row: Union[pd.Series, Dict]) -> Dict[str, Any]:
        """
        Validate that field paths in prompt configuration can be resolved against sample data.
        
        Args:
            prompt_name: Name of prompt to validate
            setup_name: Name of setup to validate
            sample_row: Sample data row for testing field path resolution
            
        Returns:
            Dict containing validation results and sample values
        """
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        if setup_name not in self.setups_config:
            raise ValueError(f"Unknown setup: {setup_name}")
        
        setup_config = self.setups_config[setup_name]
        prompt_fields_config = setup_config.get('prompt_fields', {})
        
        question_field_paths = prompt_fields_config.get('question_fields', [])
        answer_field_path = prompt_fields_config.get('answer_field', '')
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'field_values': {},
            'field_path_results': {}
        }
        
        # Test question field paths
        for field_path in question_field_paths:
            try:
                value = self.resolve_field_path(sample_row, field_path)
                validation_result['field_values'][field_path] = str(value) if value is not None else None
                validation_result['field_path_results'][field_path] = {
                    'resolved': value is not None,
                    'value_type': type(value).__name__ if value is not None else None,
                    'value_preview': str(value)[:100] if value is not None else None
                }
                
                if value is None:
                    validation_result['warnings'].append(f"Field path '{field_path}' resolved to None")
            except Exception as e:
                validation_result['valid'] = False
                error_msg = f"Error resolving field path '{field_path}': {e}"
                validation_result['errors'].append(error_msg)
                validation_result['field_path_results'][field_path] = {'error': str(e)}
        
        # Test answer field path if specified
        if answer_field_path:
            try:
                value = self.resolve_field_path(sample_row, answer_field_path)
                validation_result['field_values'][answer_field_path] = str(value) if value is not None else None
                validation_result['field_path_results'][answer_field_path] = {
                    'resolved': value is not None,
                    'value_type': type(value).__name__ if value is not None else None,
                    'value_preview': str(value)[:100] if value is not None else None
                }
                
                if value is None:
                    validation_result['warnings'].append(f"Answer field path '{answer_field_path}' resolved to None")
            except Exception as e:
                validation_result['valid'] = False
                error_msg = f"Error resolving answer field path '{answer_field_path}': {e}"
                validation_result['errors'].append(error_msg)
                validation_result['field_path_results'][answer_field_path] = {'error': str(e)}
        
        return validation_result
    
    # =============================================================================
    # LISTING AND DISCOVERY
    # =============================================================================
    
    def list_prompts(self, mode: str = None) -> Dict[str, str]:
        """List available prompts with descriptions, optionally filtered by mode"""
        prompts = {}
        for name, config in self.prompts_config.items():
            prompt_mode = config.get('mode', 'zero-shot')
            if mode is None or prompt_mode == mode:
                description = config.get('description', 'No description')
                compatible_setup = config.get('compatible_setup', 'Unknown setup')
                prompts[name] = f"{description} [mode: {prompt_mode}, setup: {compatible_setup}]"
        
        return prompts
    
    def get_prompt_field_requirements(self, prompt_name: str) -> Dict[str, Any]:
        """
        Get field requirements for a specific prompt.
        
        Args:
            prompt_name: Name of prompt to analyze
            
        Returns:
            Dict containing field requirements and template info
        """
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        prompt_config = self.prompts_config[prompt_name]
        setup_name = prompt_config.get('compatible_setup', '')
        
        if not setup_name or setup_name not in self.setups_config:
            return {'error': f"Prompt '{prompt_name}' has invalid or missing compatible_setup"}
        
        setup_config = self.setups_config[setup_name]
        prompt_fields_config = setup_config.get('prompt_fields', {})
        
        template = prompt_config.get('template', '')
        placeholder_count = template.count('{}')
        
        return {
            'prompt_name': prompt_name,
            'setup_name': setup_name,
            'mode': prompt_config.get('mode', 'zero-shot'),
            'template': template,
            'placeholder_count': placeholder_count,
            'question_field_paths': prompt_fields_config.get('question_fields', []),
            'answer_field_path': prompt_fields_config.get('answer_field', ''),
            'few_shot_template': prompt_config.get('few_shot_example', '') if prompt_config.get('mode') == 'few-shot' else None
        }