import pandas as pd
from typing import Dict, List, Any, Optional, Union
import random

from config import Config
from utils import setup_logging
from field_resolver import FieldPathResolver

logger = setup_logging("prompt_manager")

class PromptManager:
    """
    Centralized prompt management with mode-based prompting support and JSON field path resolution.
    
    Handles:
    - Loading prompt templates with mode support (zero-shot/few-shot)
    - Generic dataset field mapping using nested configuration structure
    - JSON field path resolution via FieldPathResolver utility
    - Few-shot example generation with field paths
    - Template formatting and concatenation
    """
    
    def __init__(self):
        """Initialize prompt manager with configuration from JSON"""
        self.prompts_config = Config.load_prompts_config()
        self.setups_config = Config.load_setups_config()
        logger.info(f"PromptManager initialized with {len(self.prompts_config)} prompt templates")
    
    # =============================================================================
    # FIELD PATH RESOLUTION (delegated to FieldPathResolver)
    # =============================================================================
    
    def resolve_field_path(self, data: Union[Dict, pd.Series, Any], path: str) -> Any:
        """
        Resolve a nested JSON path like 'context.questions[0].text' against data structure.
        
        Args:
            data: The data structure to navigate (dict, pd.Series, or primitive)
            path: The field path to resolve
            
        Returns:
            The value at the specified path, or None if path doesn't exist
        """
        return FieldPathResolver.resolve_field_path(data, path)
    
    def extract_field_values(self, row: Union[Dict, pd.Series], field_paths: List[str]) -> List[str]:
        """
        Extract values from multiple field paths in a data row.
        
        Args:
            row: Data row (dict for JSON data, pd.Series for CSV/Parquet data)
            field_paths: List of field paths to extract
            
        Returns:
            List of string values extracted from the specified paths
        """
        return FieldPathResolver.extract_field_values(row, field_paths)
    
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
            if config.get('compatible_setup') != setup_name:
                continue
            
            if mode is not None:
                prompt_mode = config.get('mode', 'zero-shot')
                if prompt_mode != mode:
                    continue
            
            compatible.append(prompt_name)
        
        return compatible
    
    # =============================================================================
    # ENHANCED FEW-SHOT EXAMPLE GENERATION
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
        
        if row_index is not None:
            if row_index < 0 or row_index >= len(dataset):
                raise ValueError(f"Row index {row_index} out of bounds for dataset with {len(dataset)} rows")
            row = dataset.iloc[row_index]
            logger.debug(f"Using specified row {row_index} for few-shot example")
        else:
            row = dataset.sample(n=1, random_state=Config.RANDOM_SEED).iloc[0]
            logger.debug("Using random row for few-shot example")
        
        question_values = self.extract_field_values(row, question_field_paths)
        
        answer_value = self.resolve_field_path(row, answer_field_path)
        if answer_value is None or (isinstance(answer_value, float) and pd.isna(answer_value)):
            answer_value = 'N/A'
        else:
            answer_value = str(answer_value)
        
        if answer_value.lower().strip() in ['na', 'n/a', 'not applicable', 'not annotatable', '', 'none']:
            logger.warning("Selected few-shot example has NA answer, may affect prompt quality")
        
        return {
            'question_values': question_values,
            'answer_value': answer_value,
            'row_index': row_index if row_index is not None else row.name
        }
    
    # =============================================================================
    # ENHANCED TEMPLATE FORMATTING
    # =============================================================================
    
    def format_prompt(self, prompt_name: str, field_values: List[str]) -> str:
        """Format a prompt template with field values and enhanced error handling"""
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        prompt_config = self.prompts_config[prompt_name]
        
        if 'template' not in prompt_config:
            raise ValueError(f"Prompt '{prompt_name}' missing required 'template' field")
        
        template = prompt_config['template']
        
        placeholder_count = template.count('{}')
        if len(field_values) != placeholder_count:
            raise ValueError(
                f"Template mismatch for prompt '{prompt_name}': "
                f"expected {placeholder_count} values, got {len(field_values)}. "
                f"Template: {template[:100]}..."
            )
        
        try:
            formatted_prompt = template.format(*field_values)
            logger.debug(f"Formatted template for prompt '{prompt_name}' with {len(field_values)} values")
            return formatted_prompt
        except Exception as e:
            logger.error(f"Template formatting error for prompt '{prompt_name}': {e}")
            logger.error(f"Template: {template}")
            logger.error(f"Values: {field_values}")
            raise
    
    def format_few_shot_example(self, prompt_name: str, question_values: List[str], answer_value: str) -> str:
        """Format a few-shot example template with enhanced validation"""
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        prompt_config = self.prompts_config[prompt_name]
        
        if prompt_config.get('mode') != 'few-shot':
            raise ValueError(f"Prompt '{prompt_name}' is not a few-shot prompt")
        
        if 'few_shot_example' not in prompt_config:
            raise ValueError(f"Few-shot prompt '{prompt_name}' missing required 'few_shot_example' field")
        
        few_shot_template = prompt_config['few_shot_example']
        all_values = question_values + [answer_value]
        
        placeholder_count = few_shot_template.count('{}')
        if len(all_values) != placeholder_count:
            raise ValueError(
                f"Few-shot template mismatch for prompt '{prompt_name}': "
                f"expected {placeholder_count} values, got {len(all_values)}. "
                f"Template: {few_shot_template[:100]}..."
            )
        
        try:
            formatted_example = few_shot_template.format(*all_values)
            logger.debug(f"Formatted few-shot example for prompt '{prompt_name}'")
            return formatted_example
        except Exception as e:
            logger.error(f"Few-shot example formatting error for prompt '{prompt_name}': {e}")
            logger.error(f"Template: {few_shot_template}")
            logger.error(f"Values: {all_values}")
            raise
    
    # =============================================================================
    # ENHANCED COMPLETE PROMPT PREPARATION
    # =============================================================================
    
    def prepare_prompt_for_row(self, prompt_name: str, row: Union[pd.Series, Dict], setup_name: str, 
                              mode: str, dataset: pd.DataFrame = None, 
                              few_shot_row: Optional[int] = None) -> str:
        """
        Prepare a complete formatted prompt for a dataset row with enhanced validation.
        
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
        
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        if setup_name not in self.setups_config:
            raise ValueError(f"Unknown setup: {setup_name}")
        
        prompt_config = self.prompts_config[prompt_name]
        setup_config = self.setups_config[setup_name]
        
        if 'compatible_setup' not in prompt_config:
            raise ValueError(f"Prompt '{prompt_name}' missing required 'compatible_setup' field")
        
        if 'mode' not in prompt_config:
            raise ValueError(f"Prompt '{prompt_name}' missing required 'mode' field")
        
        if 'prompt_fields' not in setup_config:
            raise ValueError(f"Setup '{setup_name}' missing required 'prompt_fields' configuration")
        
        prompt_fields_config = setup_config['prompt_fields']
        
        if 'question_fields' not in prompt_fields_config:
            raise ValueError(f"Setup '{setup_name}' missing 'question_fields' in prompt_fields configuration")
        
        if prompt_config['compatible_setup'] != setup_name:
            raise ValueError(
                f"Prompt '{prompt_name}' is compatible with setup '{prompt_config['compatible_setup']}', "
                f"not '{setup_name}'"
            )
        
        if prompt_config['mode'] != mode:
            raise ValueError(
                f"Prompt '{prompt_name}' is configured for '{prompt_config['mode']}' mode, "
                f"but '{mode}' mode was requested"
            )
        
        question_field_paths = prompt_fields_config['question_fields']
        current_question_values = self.extract_field_values(row, question_field_paths)
        
        for i, (path, value) in enumerate(zip(question_field_paths, current_question_values)):
            if not value:
                logger.warning(f"Empty value for field path '{path}' in current row")
        
        if mode == 'zero-shot':
            return self.format_prompt(prompt_name, current_question_values)
        
        elif mode == 'few-shot':
            if dataset is None:
                raise ValueError("Dataset is required for few-shot prompting")
            
            example_data = self.generate_few_shot_example(dataset, setup_name, few_shot_row)
            
            few_shot_example = self.format_few_shot_example(
                prompt_name, 
                example_data['question_values'], 
                example_data['answer_value']
            )
            
            current_template = self.format_prompt(prompt_name, current_question_values)
            
            final_prompt = few_shot_example + current_template
            
            logger.debug(f"Created few-shot prompt with example from row {example_data['row_index']}")
            return final_prompt
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes are 'zero-shot' and 'few-shot'")
    
    # =============================================================================
    # ENHANCED VALIDATION AND DEBUGGING UTILITIES
    # =============================================================================
    
    def validate_prompt_field_paths(self, prompt_name: str, setup_name: str, 
                                   sample_row: Union[pd.Series, Dict]) -> Dict[str, Any]:
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
        
        all_field_paths = question_field_paths + ([answer_field_path] if answer_field_path else [])
        
        return FieldPathResolver.validate_field_paths(sample_row, all_field_paths)
    
    def validate_prompt_template_compatibility(self, prompt_name: str, setup_name: str) -> Dict[str, Any]:
        """
        Validate that a prompt template is compatible with a setup configuration.
        
        Args:
            prompt_name: Name of prompt to validate
            setup_name: Name of setup to validate against
            
        Returns:
            Dict containing compatibility validation results
        """
        validation_result = {
            'compatible': True,
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        if prompt_name not in self.prompts_config:
            validation_result['compatible'] = False
            validation_result['errors'].append(f"Unknown prompt: {prompt_name}")
            return validation_result
        
        if setup_name not in self.setups_config:
            validation_result['compatible'] = False
            validation_result['errors'].append(f"Unknown setup: {setup_name}")
            return validation_result
        
        prompt_config = self.prompts_config[prompt_name]
        setup_config = self.setups_config[setup_name]
        
        prompt_setup = prompt_config.get('compatible_setup')
        if prompt_setup != setup_name:
            validation_result['compatible'] = False
            validation_result['errors'].append(
                f"Prompt '{prompt_name}' is compatible with setup '{prompt_setup}', not '{setup_name}'"
            )
        
        prompt_fields_config = setup_config.get('prompt_fields', {})
        question_fields = prompt_fields_config.get('question_fields', [])
        
        template = prompt_config.get('template', '')
        expected_placeholders = template.count('{}')
        available_fields = len(question_fields)
        
        if expected_placeholders != available_fields:
            validation_result['compatible'] = False
            validation_result['errors'].append(
                f"Template placeholder mismatch: template expects {expected_placeholders} values, "
                f"but setup provides {available_fields} question fields"
            )
        
        prompt_mode = prompt_config.get('mode', 'zero-shot')
        if prompt_mode == 'few-shot':
            if 'few_shot_example' not in prompt_config:
                validation_result['compatible'] = False
                validation_result['errors'].append(f"Few-shot prompt missing 'few_shot_example' template")
            else:
                few_shot_template = prompt_config['few_shot_example']
                few_shot_placeholders = few_shot_template.count('{}')
                answer_field = prompt_fields_config.get('answer_field', '')
                
                expected_few_shot = available_fields + (1 if answer_field else 0)
                if few_shot_placeholders != expected_few_shot:
                    validation_result['compatible'] = False
                    validation_result['errors'].append(
                        f"Few-shot template mismatch: expects {few_shot_placeholders} values, "
                        f"but setup provides {expected_few_shot} fields (questions + answer)"
                    )
        
        validation_result['details'] = {
            'prompt_mode': prompt_mode,
            'compatible_setup': prompt_setup,
            'template_placeholders': expected_placeholders,
            'available_question_fields': available_fields,
            'question_field_paths': question_fields,
            'answer_field_path': prompt_fields_config.get('answer_field', '')
        }
        
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
        Get field requirements for a specific prompt with enhanced error handling.
        
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
        
        result = {
            'prompt_name': prompt_name,
            'setup_name': setup_name,
            'mode': prompt_config.get('mode', 'zero-shot'),
            'template': template,
            'placeholder_count': placeholder_count,
            'question_field_paths': prompt_fields_config.get('question_fields', []),
            'answer_field_path': prompt_fields_config.get('answer_field', ''),
            'few_shot_template': prompt_config.get('few_shot_example', '') if prompt_config.get('mode') == 'few-shot' else None
        }
        
        compatibility_check = self.validate_prompt_template_compatibility(prompt_name, setup_name)
        result['compatibility'] = compatibility_check
        
        return result
    
    # =============================================================================
    # BATCH PROCESSING AND UTILITIES
    # =============================================================================
    
    def prepare_batch_prompts(self, dataset: pd.DataFrame, prompt_name: str, 
                             setup_name: str, mode: str, max_rows: Optional[int] = None,
                             few_shot_row: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Prepare prompts for multiple rows efficiently with batch processing.
        
        Args:
            dataset: Dataset to process
            prompt_name: Name of prompt template
            setup_name: Name of setup
            mode: Prompting mode
            max_rows: Maximum number of rows to process
            few_shot_row: Specific few-shot row index
            
        Returns:
            List of prompt preparation results
        """
        logger.info(f"Preparing batch prompts: {len(dataset)} rows, mode: {mode}")
        
        if max_rows:
            dataset = dataset.head(max_rows)
        
        results = []
        few_shot_example_data = None
        
        if mode == 'few-shot':
            try:
                few_shot_example_data = self.generate_few_shot_example(dataset, setup_name, few_shot_row)
                logger.info(f"Generated few-shot example from row {few_shot_example_data['row_index']}")
            except Exception as e:
                logger.error(f"Failed to generate few-shot example: {e}")
                return []
        
        for idx, row in dataset.iterrows():
            try:
                prompt = self.prepare_prompt_for_row(
                    prompt_name=prompt_name,
                    row=row,
                    setup_name=setup_name,
                    mode=mode,
                    dataset=dataset,
                    few_shot_row=few_shot_row
                )
                
                results.append({
                    'row_index': idx,
                    'prompt': prompt,
                    'success': True,
                    'error': None
                })
                
            except Exception as e:
                logger.error(f"Error preparing prompt for row {idx}: {e}")
                results.append({
                    'row_index': idx,
                    'prompt': '',
                    'success': False,
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"Batch prompt preparation: {success_count}/{len(results)} successful")
        
        return results