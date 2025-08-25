import pandas as pd
from typing import Dict, List, Any
import re

from config import Config
from utils import setup_logging

logger = setup_logging("prompt_manager")

class PromptManager:
    """
    Centralized prompt management with JSON configuration support.
    
    This class handles:
    1. Loading and validating prompt templates
    2. Formatting templates with variables
    3. Checking prompt-dataset compatibility
    4. Generating few-shot examples
    5. Providing prompt statistics and information
    """
    
    def __init__(self):
        """Initialize prompt manager with configuration from JSON"""
        self.prompts_config = Config.load_prompts_config()
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
            dict: Prompt information including required variables
        """
        if prompt_name:
            # Information for specific prompt
            if prompt_name not in self.prompts_config:
                return {"error": f"Unknown prompt: {prompt_name}"}
            
            config = self.prompts_config[prompt_name]
            info = dict(config)
            
            # Extract required template variables using regex
            template = config['template']
            template_vars = set(re.findall(r'\{(\w+)\}', template))
            info['required_variables'] = list(template_vars)
            
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
        
        Prompts are designed for specific datasets and may not work correctly
        with others due to different field names and structures.
        
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
    
    def format_prompt(self, prompt_name: str, template_vars: Dict[str, Any]) -> str:
        """
        Format a prompt template with provided variables.
        
        Performs template variable substitution and validates that all
        required variables are provided.
        
        Args:
            prompt_name: Name of prompt template to format
            template_vars: Dictionary of variables to substitute
            
        Returns:
            str: Formatted prompt ready for model input
            
        Raises:
            ValueError: If prompt unknown or variables missing
            KeyError: If template formatting fails
        """
        if prompt_name not in self.prompts_config:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        prompt_config = self.prompts_config[prompt_name]
        template = prompt_config['template']
        
        # Extract required variables from template using regex
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
        """
        Prepare a formatted prompt for a specific dataset row.
        
        This is the main method for converting dataset rows into model prompts.
        It handles dataset-specific field mapping and variable preparation.
        
        Args:
            prompt_name: Name of prompt template to use
            row: Dataset row containing the data
            dataset_name: Name of the dataset (for field mapping)
            additional_vars: Extra variables to include (e.g., few-shot examples)
            
        Returns:
            str: Formatted prompt ready for model input
            
        Raises:
            ValueError: If dataset not supported or compatibility issues
        """
        if dataset_name not in ['gmeg']:  # Add other datasets as they're supported
            raise ValueError(f"Dataset '{dataset_name}' not supported yet")
        
        # Validate compatibility (with warning, not hard failure)
        if not self.validate_prompt_dataset_compatibility(prompt_name, dataset_name):
            logger.warning(f"Using potentially incompatible prompt '{prompt_name}' with dataset '{dataset_name}'")
        
        # Start with additional variables if provided
        template_vars = additional_vars.copy() if additional_vars else {}
        
        # Dataset-specific field mapping
        if dataset_name == 'gmeg':
            # GMEG dataset has 'original' and 'revised' fields for text comparison
            template_vars.update({
                'original_text': str(row.get('original', '')),
                'revised_text': str(row.get('revised', ''))
            })
            
            # Handle few-shot prompts specially
            if 'few_shot' in prompt_name and 'few_shot_examples' not in template_vars:
                # This should normally be provided by the caller, but provide fallback
                template_vars['few_shot_examples'] = "Few-shot examples would be provided here."
                logger.warning("Few-shot examples not provided for few-shot prompt")
        
        # Format the prompt with all variables
        return self.format_prompt(prompt_name, template_vars)
    
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
        
        if dataset_name != 'gmeg':
            raise ValueError(f"Few-shot generation not implemented for dataset: {dataset_name}")
        
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
            original = str(row.get('original', 'N/A'))
            revised = str(row.get('revised', 'N/A'))
            
            # Get the answer field (dataset-specific)
            answer_field = 'please_explain_the_revisions_write_na_if_not_annotatable'  # GMEG specific
            explanation = str(row.get(answer_field, 'N/A'))
            
            # Skip examples marked as 'NA' or 'not annotatable'
            na_patterns = ['na', 'n/a', 'not applicable', 'not annotatable', '', 'none']
            if explanation.lower().strip() in na_patterns:
                continue
            
            # Format as input-output example
            example_text = f"Original: {original}\nRevised: {revised}\nAnalysis: {explanation}"
            formatted_examples.append(example_text)
        
        # If we don't have enough valid examples after filtering NA, use fallback
        if not formatted_examples and len(examples) > 0:
            row = examples.iloc[0]
            original = str(row.get('original', 'N/A'))
            revised = str(row.get('revised', 'N/A'))
            example_text = f"Original: {original}\nRevised: {revised}\nAnalysis: (Example explanation would go here)"
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
    
    def list_prompts_by_type(self, prompt_type: str) -> List[str]:
        """
        List prompts of a specific type (e.g., 'baseline', 'few_shot').
        
        Args:
            prompt_type: Type of prompt to filter by
            
        Returns:
            list: Names of prompts matching the type
        """
        prompts = []
        
        for name, config in self.prompts_config.items():
            if config.get('type', '') == prompt_type:
                prompts.append(name)
        
        return prompts
    
    def list_prompts_by_dataset(self, dataset_name: str) -> List[str]:
        """
        List prompts compatible with a specific dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            list: Names of compatible prompts
        """
        return self.get_compatible_prompts(dataset_name)
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the available prompts.
        
        Returns:
            dict: Statistics including counts by type, dataset, and average lengths
        """
        stats = {
            'total_prompts': len(self.prompts_config),
            'prompts_by_type': {},
            'prompts_by_dataset': {},
            'avg_template_length': 0
        }
        
        total_length = 0
        
        # Analyze each prompt
        for name, config in self.prompts_config.items():
            # Count by type
            prompt_type = config.get('type', 'unknown')
            if prompt_type not in stats['prompts_by_type']:
                stats['prompts_by_type'][prompt_type] = 0
            stats['prompts_by_type'][prompt_type] += 1
            
            # Count by compatible dataset
            compatible_dataset = config.get('compatible_dataset', 'unknown')
            if compatible_dataset not in stats['prompts_by_dataset']:
                stats['prompts_by_dataset'][compatible_dataset] = 0
            stats['prompts_by_dataset'][compatible_dataset] += 1
            
            # Track template length
            total_length += len(config.get('template', ''))
        
        # Calculate average template length
        if stats['total_prompts'] > 0:
            stats['avg_template_length'] = total_length / stats['total_prompts']
        
        return stats
    
    # =============================================================================
    # PROMPT VALIDATION
    # =============================================================================
    
    def validate_prompt_template(self, prompt_name: str) -> Dict[str, Any]:
        """
        Validate a prompt template for common issues and correctness.
        
        Checks for:
        - Template existence and basic structure
        - Variable syntax correctness
        - Reasonable template length
        - Common formatting issues
        
        Args:
            prompt_name: Name of prompt to validate
            
        Returns:
            dict: Validation results with issues and recommendations
        """
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
        
        # Check for malformed variables (has { but not proper {var} format)
        if '{' in template and not re.search(r'\{\w+\}', template):
            validation_result["issues"].append("Template contains malformed variables")
            validation_result["valid"] = False
        
        # Check for unmatched braces
        open_braces = template.count('{')
        close_braces = template.count('}')
        if open_braces != close_braces:
            validation_result["issues"].append("Unmatched braces in template")
            validation_result["valid"] = False
        
        return validation_result