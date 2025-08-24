import pandas as pd
from typing import Dict, List, Any
import re

from config import Config
from utils import setup_logging

logger = setup_logging("prompt_manager")

class PromptManager:
    """Centralized prompt management with JSON configuration support"""
    
    def __init__(self):
        self.prompts_config = Config.load_prompts_config()
        logger.info(f"PromptManager initialized with {len(self.prompts_config)} prompt templates")
    
    def get_prompt_info(self, prompt_name: str = None) -> Dict[str, Any]:
        """Get information about prompts"""
        if prompt_name:
            if prompt_name not in self.prompts_config:
                return {"error": f"Unknown prompt: {prompt_name}"}
            
            config = self.prompts_config[prompt_name]
            info = dict(config)
            
            # Extract required template variables
            template = config['template']
            template_vars = set(re.findall(r'\{(\w+)\}', template))
            info['required_variables'] = list(template_vars)
            
            return info
        else:
            # Return info for all prompts
            all_info = {}
            for prompt_key in self.prompts_config:
                all_info[prompt_key] = self.get_prompt_info(prompt_key)
            return all_info
    
    def validate_prompt_dataset_compatibility(self, prompt_name: str, dataset_name: str) -> bool:
        """Check if prompt is compatible with dataset"""
        if prompt_name not in self.prompts_config:
            logger.error(f"Unknown prompt: {prompt_name}")
            return False
        
        prompt_config = self.prompts_config[prompt_name]
        compatible_dataset = prompt_config.get('compatible_dataset', '')
        
        is_compatible = compatible_dataset == dataset_name
        
        if not is_compatible:
            logger.warning(f"Prompt '{prompt_name}' is designed for dataset '{compatible_dataset}', not '{dataset_name}'")
            return is_compatible
        
        prompt_config = self.prompts_config[prompt_name]
        compatible_dataset = prompt_config.get('compatible_dataset', '')
        
        is_compatible = compatible_dataset == dataset_name
        
        if not is_compatible:
            logger.warning(f"Prompt '{prompt_name}' is designed for dataset '{compatible_dataset}', not '{dataset_name}'")
        
        return is_compatible
    
    def get_compatible_prompts(self, dataset_name: str) -> List[str]:
        """Get list of prompts compatible with a dataset"""
        compatible = []
        
        for prompt_name, config in self.prompts_config.items():
            if config.get('compatible_dataset') == dataset_name:
                compatible.append(prompt_name)
        
        return compatible
    
    def format_prompt(self, prompt_name: str, template_vars: Dict[str, Any]) -> str:
        """Format a prompt template with variables"""
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        prompt_config = self.prompts_config[prompt_name]
        template = prompt_config['template']
        
        # Extract required variables from template
        required_vars = set(re.findall(r'\{(\w+)\}', template))
        provided_vars = set(template_vars.keys())
        
        # Check for missing variables
        missing_vars = required_vars - provided_vars
        if missing_vars:
            logger.error(f"Missing template variables for prompt '{prompt_name}': {missing_vars}")
            raise ValueError(f"Missing template variables: {missing_vars}")
        
        try:
            formatted_prompt = template.format(**template_vars)
            logger.debug(f"Formatted prompt '{prompt_name}' with length {len(formatted_prompt)}")
            return formatted_prompt
            
        except KeyError as e:
            logger.error(f"Template formatting error for prompt '{prompt_name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error formatting prompt '{prompt_name}': {e}")
            raise
    
    def prepare_prompt_for_row(self, prompt_name: str, row: pd.Series, dataset_name: str, 
                              additional_vars: Dict[str, Any] = None) -> str:
        """Prepare a formatted prompt for a specific dataset row"""
        if dataset_name not in ['gmeg']:  # Add other datasets as they're supported
            raise ValueError(f"Dataset '{dataset_name}' not supported yet")
        
        # Validate compatibility
        if not self.validate_prompt_dataset_compatibility(prompt_name, dataset_name):
            logger.warning(f"Using potentially incompatible prompt '{prompt_name}' with dataset '{dataset_name}'")
        
        # Prepare template variables based on dataset
        template_vars = additional_vars.copy() if additional_vars else {}
        
        if dataset_name == 'gmeg':
            # GMEG dataset uses 'original' and 'revised' fields
            template_vars.update({
                'original_text': str(row.get('original', '')),
                'revised_text': str(row.get('revised', ''))
            })
            
            # Handle few-shot prompts
            if 'few_shot' in prompt_name and 'few_shot_examples' not in template_vars:
                # This should be provided by the caller, but we can create a placeholder
                template_vars['few_shot_examples'] = "Few-shot examples would be provided here."
                logger.warning("Few-shot examples not provided for few-shot prompt")
        
        # Format the prompt
        return self.format_prompt(prompt_name, template_vars)
    
    def generate_few_shot_examples(self, dataset: pd.DataFrame, dataset_name: str, 
                                 n_examples: int = 3, exclude_indices: List[int] = None) -> str:
        """Generate few-shot examples from dataset"""
        logger.info(f"Generating {n_examples} few-shot examples for dataset {dataset_name}")
        
        if dataset_name != 'gmeg':
            raise ValueError(f"Few-shot generation not implemented for dataset: {dataset_name}")
        
        # Filter out excluded indices
        if exclude_indices:
            available_data = dataset[~dataset.index.isin(exclude_indices)]
        else:
            available_data = dataset
        
        if len(available_data) < n_examples:
            logger.warning(f"Not enough data for {n_examples} examples, using {len(available_data)}")
            n_examples = len(available_data)
        
        # Sample examples
        examples = available_data.sample(n=n_examples, random_state=Config.RANDOM_SEED)
        
        formatted_examples = []
        
        for _, row in examples.iterrows():
            original = str(row.get('original', 'N/A'))
            revised = str(row.get('revised', 'N/A'))
            answer_field = 'please_explain_the_revisions_write_na_if_not_annotatable'  # GMEG specific
            explanation = str(row.get(answer_field, 'N/A'))
            
            # Skip examples marked as 'NA' or 'not annotatable'
            if explanation.lower().strip() in ['na', 'n/a', 'not applicable', 'not annotatable', '']:
                continue
            
            example_text = f"Original: {original}\nRevised: {revised}\nAnalysis: {explanation}"
            formatted_examples.append(example_text)
        
        # If we don't have enough valid examples after filtering NA, take what we have
        if not formatted_examples and len(examples) > 0:
            # Fallback: use first example even if it's NA
            row = examples.iloc[0]
            original = str(row.get('original', 'N/A'))
            revised = str(row.get('revised', 'N/A'))
            example_text = f"Original: {original}\nRevised: {revised}\nAnalysis: (Example explanation would go here)"
            formatted_examples.append(example_text)
        
        result = "\n\n".join(formatted_examples[:n_examples])
        logger.debug(f"Generated few-shot examples with {len(formatted_examples)} examples")
        
        return result
    
    def list_prompts(self) -> Dict[str, str]:
        """List all available prompts with descriptions"""
        return {name: config['description'] for name, config in self.prompts_config.items()}
    
    def list_prompts_by_type(self, prompt_type: str) -> List[str]:
        """List prompts of a specific type"""
        prompts = []
        
        for name, config in self.prompts_config.items():
            if config.get('type', '') == prompt_type:
                prompts.append(name)
        
        return prompts
    
    def list_prompts_by_dataset(self, dataset_name: str) -> List[str]:
        """List prompts compatible with a specific dataset"""
        return self.get_compatible_prompts(dataset_name)
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about available prompts"""
        stats = {
            'total_prompts': len(self.prompts_config),
            'prompts_by_type': {},
            'prompts_by_dataset': {},
            'avg_template_length': 0
        }
        
        total_length = 0
        
        for name, config in self.prompts_config.items():
            # Count by type
            prompt_type = config.get('type', 'unknown')
            if prompt_type not in stats['prompts_by_type']:
                stats['prompts_by_type'][prompt_type] = 0
            stats['prompts_by_type'][prompt_type] += 1
            
            # Count by dataset
            compatible_dataset = config.get('compatible_dataset', 'unknown')
            if compatible_dataset not in stats['prompts_by_dataset']:
                stats['prompts_by_dataset'][compatible_dataset] = 0
            stats['prompts_by_dataset'][compatible_dataset] += 1
            
            # Template length
            total_length += len(config.get('template', ''))
        
        if stats['total_prompts'] > 0:
            stats['avg_template_length'] = total_length / stats['total_prompts']
        
        return stats
    
    def validate_prompt_template(self, prompt_name: str) -> Dict[str, Any]:
        """Validate a prompt template"""
        if prompt_name not in self.prompts_config:
            return {"valid": False, "error": f"Unknown prompt: {prompt_name}"}
        
        config = self.prompts_config[prompt_name]
        template = config.get('template', '')
        
        validation_result = {
            "valid": True,
            "prompt_name": prompt_name,
            "template_length": len(template),
            "required_variables": [],
            "issues": []
        }
        
        # Extract template variables
        try:
            template_vars = set(re.findall(r'\{(\w+)\}', template))
            validation_result["required_variables"] = list(template_vars)
        except Exception as e:
            validation_result["issues"].append(f"Error parsing template variables: {e}")
            validation_result["valid"] = False
        
        # Check for common issues
        if not template:
            validation_result["issues"].append("Empty template")
            validation_result["valid"] = False
        
        if len(template) < 50:
            validation_result["issues"].append("Template seems very short")
        
        if '{' in template and not re.search(r'\{\w+\}', template):
            validation_result["issues"].append("Template contains malformed variables")
            validation_result["valid"] = False
        
        return validation_result