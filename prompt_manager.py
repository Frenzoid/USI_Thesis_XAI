import pandas as pd
from typing import Dict, List, Any, Optional
import random

from config import Config
from utils import setup_logging

logger = setup_logging("prompt_manager")

class PromptManager:
    """
    Centralized prompt management with mode-based prompting support.
    
    Handles:
    - Loading prompt templates with mode support (zero-shot/few-shot)
    - Generic dataset field mapping using nested configuration structure
    - Few-shot example generation
    - Template formatting and concatenation
    """
    
    def __init__(self):
        """Initialize prompt manager with configuration from JSON"""
        self.prompts_config = Config.load_prompts_config()
        self.setups_config = Config.load_setups_config()
        logger.info(f"PromptManager initialized with {len(self.prompts_config)} prompt templates")
    
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
    
    def generate_few_shot_example(self, dataset: pd.DataFrame, setup_name: str, 
                                 row_index: Optional[int] = None) -> Dict[str, Any]:
        """Generate a few-shot example from dataset row"""
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
        
        question_fields = prompt_fields_config['question_fields']
        answer_field = prompt_fields_config['answer_field']
        
        if not answer_field:
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
        
        # Extract question values
        question_values = []
        for field in question_fields:
            if field in row and not pd.isna(row[field]):
                question_values.append(str(row[field]))
            else:
                question_values.append('N/A')
        
        # Extract answer value
        if answer_field in row and not pd.isna(row[answer_field]):
            answer_value = str(row[answer_field])
        else:
            answer_value = 'N/A'
        
        # Warn about NA answers
        if answer_value.lower().strip() in ['na', 'n/a', 'not applicable', 'not annotatable', '', 'none']:
            logger.warning("Selected few-shot example has NA answer, may affect prompt quality")
        
        return {
            'question_values': question_values,
            'answer_value': answer_value,
            'row_index': row_index if row_index is not None else row.name
        }
    
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
    
    def prepare_prompt_for_row(self, prompt_name: str, row: pd.Series, setup_name: str, 
                              mode: str, dataset: pd.DataFrame = None, 
                              few_shot_row: Optional[int] = None) -> str:
        """Prepare a complete formatted prompt for a dataset row"""
        
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
        
        # Extract current row values
        question_fields = prompt_fields_config['question_fields']
        current_question_values = []
        
        for field in question_fields:
            if field in row and not pd.isna(row[field]):
                current_question_values.append(str(row[field]))
            else:
                current_question_values.append("")
                logger.warning(f"Missing field '{field}' in row, using empty string")
        
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