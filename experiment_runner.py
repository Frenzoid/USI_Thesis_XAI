import os
import time
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from config import Config
from utils import setup_logging, clear_gpu_memory
from models import ModelManager
from dataset_manager import DatasetManager
from prompt_manager import PromptManager

logger = setup_logging("experiment_runner")

class ExperimentRunner:
    """
    Handles running inference experiments with different models and configurations.
    
    This class orchestrates the entire experiment pipeline using generic field mapping:
    1. Validates configurations and field compatibility
    2. Prepares datasets and prompts using generic field mapping
    3. Loads and queries models
    4. Saves results with metadata
    """
    
    def __init__(self):
        """Initialize all component managers"""
        self.model_manager = ModelManager()
        self.prompt_manager = PromptManager()
        self.dataset_manager = DatasetManager()
        
        # Load all configuration files at startup
        self.prompts_config = Config.load_prompts_config()
        self.datasets_config = Config.load_datasets_config()
        self.models_config = Config.load_models_config()
        
        # Always initialize API clients (safe for all systems)
        self.model_manager.setup_api_clients()
        
        logger.info("ExperimentRunner initialized")
    
    def validate_experiment_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate experiment configuration for required fields and compatibility.
        
        Checks:
        - All required fields are present
        - Experiment type is valid
        - Model, dataset, and prompt exist in configurations
        - Mode is valid and matches prompt mode
        - Few-shot row is valid if specified
        - Prompt is compatible with the dataset
        - Prompt placeholder count matches dataset question field count
        
        Args:
            config: Experiment configuration dictionary
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        required_fields = ['experiment_type', 'model', 'dataset', 'prompt', 'mode', 'size', 'temperature']
        
        # Check all required fields are present
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate experiment type
        if not Config.validate_experiment_type(config['experiment_type']):
            logger.error(f"Invalid experiment type: {config['experiment_type']}")
            return False
        
        # Validate mode
        if config['mode'] not in ['zero-shot', 'few-shot']:
            logger.error(f"Invalid mode: {config['mode']}")
            return False
        
        # Validate model exists
        if config['model'] not in self.models_config:
            logger.error(f"Unknown model: {config['model']}")
            return False
        
        # Validate dataset exists
        if config['dataset'] not in self.datasets_config:
            logger.error(f"Unknown dataset: {config['dataset']}")
            return False
        
        # Validate prompt exists
        if config['prompt'] not in self.prompts_config:
            logger.error(f"Unknown prompt: {config['prompt']}")
            return False
        
        # Validate mode compatibility with prompt
        if not self.prompt_manager.validate_mode_compatibility(config['prompt'], config['mode']):
            return False
        
        # Validate prompt structure if needed
        if not self.prompt_manager.validate_few_shot_prompt_structure(config['prompt']):
            return False
        
        # Check prompt-dataset compatibility
        if not self.prompt_manager.validate_prompt_dataset_compatibility(config['prompt'], config['dataset']):
            logger.error(f"Prompt '{config['prompt']}' not compatible with dataset '{config['dataset']}'")
            return False
        
        # Validate prompt placeholder count matches dataset question fields
        try:
            self.prompt_manager.validate_prompt_field_count(config['prompt'], config['dataset'])
        except ValueError as e:
            logger.error(f"Field count validation failed: {e}")
            return False
        
        # Validate few-shot row if specified
        if config.get('few_shot_row') is not None:
            if config['mode'] != 'few-shot':
                logger.error("Cannot specify few_shot_row for non-few-shot experiments")
                return False
            if config['few_shot_row'] < 0:
                logger.error(f"Few-shot row must be non-negative: {config['few_shot_row']}")
                return False
        
        return True
    
    def prepare_dataset(self, dataset_name: str, size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare dataset for experiment.
        
        Downloads dataset if necessary, loads it, and samples to the requested size.
        Returns both the full dataset (for few-shot generation) and the sample (for experiment).
        
        Args:
            dataset_name: Name of dataset to prepare
            size: Number of samples to return
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (full_dataset, sampled_dataset)
            
        Raises:
            Exception: If dataset cannot be loaded or prepared
        """
        logger.info(f"Preparing dataset: {dataset_name}")
        
        dataset_config = self.datasets_config[dataset_name]
        
        # Construct path to dataset file
        dataset_path = os.path.join(Config.DATA_DIR, dataset_config['download_path'], dataset_config['csv_file'])
        
        # Download if not exists
        if not os.path.exists(dataset_path):
            logger.info(f"Dataset not found, downloading: {dataset_name}")
            self.download_dataset(dataset_name)
        
        # Load full dataset
        try:
            full_df = pd.read_csv(dataset_path)
            logger.info(f"Loaded full dataset with {len(full_df)} rows")
            
            # Sample data if requested size is smaller than dataset
            if size < len(full_df):
                sampled_df = full_df.sample(n=size, random_state=Config.RANDOM_SEED).reset_index(drop=True)
                logger.info(f"Sampled {size} rows from dataset for experiment")
            else:
                sampled_df = full_df.copy()
                logger.info(f"Using all {len(full_df)} rows (requested size >= dataset size)")
            
            return full_df, sampled_df
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise
    
    def download_dataset(self, dataset_name: str):
        """Use DatasetManager's download method"""
        return self.dataset_manager.download_dataset(dataset_name)
    
    def load_model(self, model_name: str):
        """
        Load the specified model for inference.
        
        Handles both local models (via Unsloth) and API models (validation only).
        For local models, supports both base and finetuned variants.
        
        Args:
            model_name: Name of model to load
            
        Raises:
            ValueError: If model configuration is invalid or API keys missing
        """
        model_config = self.models_config[model_name]
        
        if model_config['type'] == 'local':
            # Handle local models with Unsloth
            if model_config.get('finetuned', False):
                # Finetuned model: look in finetuned models directory
                model_path = os.path.join(Config.FINETUNED_MODELS_DIR, 
                                        model_config['model_path'].split('/')[-1] + '_finetuned')
                logger.info(f"Loading finetuned model from: {model_path}")
                self.model_manager.load_finetuned_model(model_path)
            else:
                # Base model: use configured path
                logger.info(f"Loading base model: {model_config['model_path']}")
                self.model_manager.load_open_source_model(model_name, model_config['model_path'])
        
        elif model_config['type'] == 'api':
            # API models: validate credentials only
            provider = model_config['provider']
            if provider == 'openai' and not Config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not configured")
            elif provider == 'google' and not Config.GENAI_API_KEY:
                raise ValueError("Google GenAI API key not configured")
            elif provider == 'anthropic' and not Config.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not configured")
            
            logger.info(f"Using API model: {model_name}")
        
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
    
    def generate_responses(self, prompts: List[str], expected_outputs: List[str], 
                          model_name: str, temperature: float, question_values_list: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Generate responses using the loaded model.
        
        Handles both local and API models, with comprehensive error handling
        and performance tracking.
        
        Args:
            prompts: List of prompts to send to model
            expected_outputs: List of expected outputs for evaluation
            model_name: Name of model to use
            temperature: Temperature parameter for generation
            question_values_list: List of question field values used to populate each prompt
            
        Returns:
            List[Dict]: List of response objects with metadata
        """
        logger.info(f"Generating {len(prompts)} responses using {model_name}")
        
        model_config = self.models_config[model_name]
        max_tokens = model_config.get('max_tokens', Config.MAX_NEW_TOKENS)
        
        responses = []
        processing_times = []
        
        for i, prompt in enumerate(tqdm(prompts, desc=f"Generating ({model_name})")):
            start_time = time.time()
            
            # Get the corresponding expected output and question values
            expected_output = expected_outputs[i]
            question_values = question_values_list[i] if i < len(question_values_list) else []
            
            try:
                # Route to appropriate model type
                if model_config['type'] == 'local':
                    response = self.model_manager.query_open_source(
                        prompt, max_tokens=max_tokens, temperature=temperature
                    )
                elif model_config['type'] == 'api':
                    # Route to appropriate API provider
                    provider = model_config['provider']
                    api_model_name = model_config['model_name']
                    
                    if provider == 'openai':
                        response = self.model_manager.query_openai(
                            prompt, model=api_model_name, max_tokens=max_tokens, temperature=temperature
                        )
                    elif provider == 'google':
                        response = self.model_manager.query_genai(
                            prompt, model=api_model_name, max_tokens=max_tokens, temperature=temperature
                        )
                    elif provider == 'anthropic':
                        response = self.model_manager.query_anthropic(
                            prompt, model=api_model_name, max_tokens=max_tokens, temperature=temperature
                        )
                    else:
                        raise ValueError(f"Unknown API provider: {provider}")
                else:
                    raise ValueError(f"Unknown model type: {model_config['type']}")
                
                processing_time = time.time() - start_time
                
                # Clean response (remove prompt if model echoed it)
                if response and response.startswith(prompt):
                    response = response[len(prompt):].strip()
                
                # Store successful response with metadata including question values
                responses.append({
                    'prompt': prompt,
                    'response': response,
                    'expected_output': expected_output,
                    'question_values': question_values,
                    'processing_time': processing_time,
                    'success': True,
                    'error': None
                })
                processing_times.append(processing_time)
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = str(e)
                
                logger.error(f"Error generating response {i+1}/{len(prompts)}: {error_msg}")
                
                # Store failed response with error info including question values
                responses.append({
                    'prompt': prompt,
                    'response': f"Error: {error_msg}",
                    'expected_output': expected_output,
                    'question_values': question_values,
                    'processing_time': processing_time,
                    'success': False,
                    'error': error_msg
                })
                processing_times.append(processing_time)
                continue
        
        # Log processing statistics
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            total_time = sum(processing_times)
            success_rate = sum(1 for r in responses if r['success']) / len(responses) * 100
            
            logger.info(f"Processing complete - Avg: {avg_time:.2f}s, Total: {total_time:.2f}s, Success: {success_rate:.1f}%")
        
        return responses
    
    def save_experiment_results(self, experiment_config: Dict[str, Any], 
                              df: pd.DataFrame, responses: List[Dict[str, Any]]) -> str:
        """
        Save experiment results to JSON file with comprehensive metadata.
        
        Creates a structured output file containing all experiment data,
        configuration, and processing statistics.
        
        Args:
            experiment_config: Configuration used for the experiment
            df: Original dataset DataFrame (this should be the sampled dataset for the experiment)
            responses: Generated responses with metadata
            
        Returns:
            str: Path to saved file
            
        Raises:
            Exception: If file cannot be saved
        """
        
        # Generate experiment name with few-shot row information
        experiment_name = Config.generate_experiment_name(
            experiment_config['experiment_type'],
            experiment_config['dataset'],
            experiment_config['model'],
            experiment_config['mode'],
            experiment_config['prompt'],
            experiment_config['size'],
            experiment_config['temperature'],
            few_shot_row=experiment_config.get('few_shot_row')
        )
        
        file_paths = Config.generate_file_paths(experiment_config['experiment_type'], experiment_name)
        output_file = file_paths['inference']
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare comprehensive results data
        results = {
            'experiment_name': experiment_name,
            'experiment_config': experiment_config,
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'name': experiment_config['dataset'],
                'size': len(df),
                'columns': list(df.columns)
            },
            'model_info': self.models_config[experiment_config['model']],
            'prompt_info': self.prompts_config[experiment_config['prompt']],
            'responses': responses,
            'expected_outputs': [],
            'processing_stats': {}
        }
        
        # Add expected outputs for evaluation using generic method
        for idx, row in df.iterrows():
            expected = self.dataset_manager.get_expected_answer(row, experiment_config['dataset'])
            results['expected_outputs'].append(expected)
        
        # Calculate comprehensive processing statistics
        processing_times = [r['processing_time'] for r in responses]
        if processing_times:
            results['processing_stats'] = {
                'total_time': sum(processing_times),
                'avg_time': sum(processing_times) / len(processing_times),
                'min_time': min(processing_times),
                'max_time': max(processing_times),
                'success_count': sum(1 for r in responses if r['success']),
                'error_count': sum(1 for r in responses if not r['success'])
            }
        
        # Save to file
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Experiment results saved: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving experiment results: {e}")
            raise
    
    def run_baseline_experiment(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Run a complete baseline experiment with mode-based prompting support.
        
        This is the main experiment execution method that:
        1. Validates configuration including mode and few-shot compatibility
        2. Prepares dataset and validates few-shot row if specified
        3. Loads model
        4. Generates prompts using mode-based system
        5. Generates responses and saves results
        
        Args:
            config: Complete experiment configuration
            
        Returns:
            Dict containing experiment results and metadata, or None if failed
        """
        logger.info(f"Starting {config['mode']} baseline experiment: {config}")
        
        # Validate configuration including mode compatibility
        if not self.validate_experiment_config(config):
            logger.error("Experiment configuration validation failed")
            return None
        
        try:
            # Step 1: Prepare dataset - returns both full dataset and sample
            full_df, sampled_df = self.prepare_dataset(config['dataset'], config['size'])

            # Ensure we only process the requested number of samples
            df_slice = sampled_df.head(config['size'])
            
            # Validate dataset has required fields
            if not self.dataset_manager.validate_dataset_fields(config['dataset']):
                logger.error(f"Dataset validation failed for {config['dataset']}")
                return None
            
            # Step 2: Determine few-shot row to use for consistency across all prompts
            actual_few_shot_row = None
            if config['mode'] == 'few-shot':
                if config.get('few_shot_row') is not None:
                    # User specified a row
                    if config['few_shot_row'] >= len(full_df):
                        logger.error(f"Few-shot row {config['few_shot_row']} out of bounds for full dataset with {len(full_df)} rows")
                        return None
                    actual_few_shot_row = config['few_shot_row']
                    logger.info(f"Using specified few-shot row {actual_few_shot_row} from full dataset ({len(full_df)} rows)")
                else:
                    # Generate a random row once for consistency across all prompts
                    import random
                    random.seed(Config.RANDOM_SEED)
                    actual_few_shot_row = random.randint(0, len(full_df) - 1)
                    logger.info(f"Generated random few-shot row {actual_few_shot_row} from full dataset ({len(full_df)} rows)")
                
                # Update config with the actual row used for saving and naming
                config['few_shot_row'] = actual_few_shot_row
            
            # Step 3: Load model
            self.load_model(config['model'])
            
            # Step 4: Prepare prompts using mode-based system
            prompts = []
            expected_outputs = []
            question_values_list = []
            
            logger.info(f"Preparing {config['mode']} prompts for {len(df_slice)} samples")
            
            for idx, row in df_slice.iterrows():
                try:
                    # Extract question field values for this row
                    dataset_config = self.datasets_config[config['dataset']]
                    question_fields = dataset_config.get('question_fields', [])
                    
                    current_question_values = []
                    for field in question_fields:
                        if field in row and not pd.isna(row[field]):
                            current_question_values.append(str(row[field]))
                        else:
                            current_question_values.append("")
                            logger.warning(f"Missing or null field '{field}' in row {idx}, using empty string")
                    
                    question_values_list.append(current_question_values)
                    
                    # Use the mode-based prompt preparation with FULL dataset for few-shot generation
                    prompt = self.prompt_manager.prepare_prompt_for_row(
                        prompt_name=config['prompt'],
                        row=row,
                        dataset_name=config['dataset'],
                        mode=config['mode'],
                        dataset=full_df,  # Use full dataset for few-shot generation
                        few_shot_row=actual_few_shot_row  # Use the determined row for consistency
                    )
                    prompts.append(prompt)
                    
                    # Get expected answer using generic dataset manager method
                    expected_output = self.dataset_manager.get_expected_answer(row, config['dataset'])
                    expected_outputs.append(expected_output)
                    
                except Exception as e:
                    logger.error(f"Error preparing {config['mode']} prompt for row {idx}: {e}")
                    prompts.append("")  # Fallback empty prompt
                    expected_outputs.append("")
                    question_values_list.append([])  # Fallback empty question values
            
            logger.info(f"Prepared {len(prompts)} {config['mode']} prompts")
            
            # Step 5: Generate responses
            responses = self.generate_responses(prompts, expected_outputs, config['model'], config['temperature'], question_values_list)
            
            # Step 6: Save results (use sampled dataset for result metadata)
            output_file = self.save_experiment_results(config, df_slice, responses)
            
            # Clean up GPU memory if using local models
            if self.models_config[config['model']]['type'] == 'local':
                clear_gpu_memory()
            
            # Generate experiment name with few-shot row information
            experiment_name = Config.generate_experiment_name(
                config['experiment_type'], 
                config['dataset'],
                config['model'],
                config['mode'],
                config['prompt'],
                config['size'],
                config['temperature'],
                few_shot_row=actual_few_shot_row
            )
            
            return {
                'experiment_name': experiment_name,
                'config': config,
                'output_file': output_file,
                'success': True,
                'num_responses': len(responses),
                'success_rate': sum(1 for r in responses if r['success']) / len(responses) if responses else 0
            }
            
        except Exception as e:
            logger.error(f"{config['mode']} baseline experiment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None