import os
import time
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from config import Config
from utils import setup_logging, download_dataset, clear_gpu_memory
from models import ModelManager
from dataset_manager import DatasetManager
from prompt_manager import PromptManager

logger = setup_logging("experiment_runner")

class ExperimentRunner:
    """
    Handles running inference experiments with different models and configurations.
    
    This class orchestrates the entire experiment pipeline:
    1. Validates configurations
    2. Prepares datasets and prompts
    3. Loads and queries models
    4. Saves results with metadata
    """
    
    def __init__(self):
        """Initialize all component managers and load configurations"""
        self.model_manager = ModelManager()
        self.prompt_manager = PromptManager()
        self.dataset_manager = DatasetManager()
        
        # Load all configuration files at startup
        self.prompts_config = Config.load_prompts_config()
        self.datasets_config = Config.load_datasets_config()
        self.models_config = Config.load_models_config()
        
        # Initialize system components
        self.model_manager.setup_api_clients()
        self.model_manager.load_embedding_model()
        
        logger.info("ExperimentRunner initialized")
    
    def validate_experiment_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate experiment configuration for required fields and compatibility.
        
        Checks:
        - All required fields are present
        - Experiment type is valid
        - Model, dataset, and prompt exist in configurations
        - Prompt is compatible with the dataset
        
        Args:
            config: Experiment configuration dictionary
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        required_fields = ['experiment_type', 'model', 'dataset', 'prompt', 'size', 'temperature']
        
        # Check all required fields are present
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate experiment type
        if not Config.validate_experiment_type(config['experiment_type']):
            logger.error(f"Invalid experiment type: {config['experiment_type']}")
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
        
        # Check prompt-dataset compatibility
        prompt_config = self.prompts_config[config['prompt']]
        if prompt_config['compatible_dataset'] != config['dataset']:
            logger.error(f"Prompt '{config['prompt']}' not compatible with dataset '{config['dataset']}'")
            return False
        
        return True
    
    def prepare_dataset(self, dataset_name: str, size: int) -> pd.DataFrame:
        """
        Load and prepare dataset for experiment.
        
        Downloads dataset if necessary, loads it, and samples to the requested size.
        
        Args:
            dataset_name: Name of dataset to prepare
            size: Number of samples to return
            
        Returns:
            pandas.DataFrame: Prepared dataset sample
            
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
        
        # Load and sample dataset
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded dataset with {len(df)} rows")
            
            # Sample data if requested size is smaller than dataset
            if size < len(df):
                df = df.sample(n=size, random_state=Config.RANDOM_SEED).reset_index(drop=True)
                logger.info(f"Sampled {size} rows from dataset")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise
    
    def download_dataset(self, dataset_name: str):
        """
        Download dataset using the existing download infrastructure.
        
        Creates a temporary configuration file and uses the legacy download function.
        This maintains compatibility with the existing download system.
        
        Args:
            dataset_name: Name of dataset to download
        """
        dataset_config = self.datasets_config[dataset_name]
        
        download_path = os.path.join(Config.DATA_DIR, dataset_config['download_path'])
        os.makedirs(download_path, exist_ok=True)
        
        # Create temporary configuration for legacy download function
        temp_config = [{
            'name': dataset_name,
            'link': dataset_config['download_link'],
            'storage_folder': dataset_config['download_path']
        }]
        
        temp_file = 'temp_dataset_config.json'
        with open(temp_file, 'w') as f:
            json.dump(temp_config, f)
        
        try:
            download_dataset(temp_file, Config.DATA_DIR)
        finally:
            # Always clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def prepare_prompts(self, df: pd.DataFrame, dataset_name: str, prompt_name: str) -> List[str]:
        """
        Prepare prompts for each row in the dataset.
        
        This is a legacy method maintained for compatibility. The actual prompt preparation
        is now done in run_baseline_experiment with better error handling.
        
        Args:
            df: Dataset DataFrame
            dataset_name: Name of the dataset
            prompt_name: Name of the prompt template
            
        Returns:
            List[str]: List of formatted prompts
        """
        logger.info(f"Preparing prompts using: {prompt_name}")
        
        prompts = []
        for idx, row in df.iterrows():
            try:
                # Use prompt manager to prepare prompt for this row
                prompt = self.prompt_manager.prepare_prompt_for_row(prompt_name, row, dataset_name)
                prompts.append(prompt)
                
            except Exception as e:
                logger.error(f"Error preparing prompt for row {idx}: {e}")
                # Add empty prompt as fallback
                prompts.append("")
        
        logger.info(f"Prepared {len(prompts)} prompts")
        return prompts
    
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
            # API models: validate credentials
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
                          model_name: str, temperature: float) -> List[Dict[str, Any]]:
        """
        Generate responses using the loaded model.
        
        Handles both local and API models, with comprehensive error handling
        and performance tracking.
        
        Args:
            prompts: List of prompts to send to model
            expected_outputs: List of expected outputs for evaluation
            model_name: Name of model to use
            temperature: Temperature parameter for generation
            
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
            
            # Get the corresponding expected output
            expected_output = expected_outputs[i]
            
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
                
                # Store successful response with metadata
                responses.append({
                    'prompt': prompt,
                    'response': response,
                    'expected_output': expected_output,
                    'processing_time': processing_time,
                    'success': True,
                    'error': None
                })
                processing_times.append(processing_time)
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = str(e)
                
                logger.error(f"Error generating response {i+1}/{len(prompts)}: {error_msg}")
                
                # Store failed response with error info
                responses.append({
                    'prompt': prompt,
                    'response': f"Error: {error_msg}",
                    'expected_output': expected_output,
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
            df: Original dataset DataFrame
            responses: Generated responses with metadata
            
        Returns:
            str: Path to saved file
            
        Raises:
            Exception: If file cannot be saved
        """
        
        # Generate experiment name and file paths
        experiment_name = Config.generate_experiment_name(
            experiment_config['experiment_type'],
            experiment_config['dataset'],
            experiment_config['model'],
            experiment_config['prompt'],
            experiment_config['size'],
            experiment_config['temperature']
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
        
        # Add expected outputs for evaluation
        dataset_config = self.datasets_config[experiment_config['dataset']]
        answer_field = dataset_config['answer_field']
        
        for idx, row in df.iterrows():
            expected = str(row[answer_field]) if answer_field in row and not pd.isna(row[answer_field]) else ""
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
        Run a complete baseline experiment with the given configuration.
        
        This is the main experiment execution method that:
        1. Validates configuration
        2. Prepares dataset and prompts
        3. Loads model
        4. Generates responses
        5. Saves results
        
        Args:
            config: Complete experiment configuration
            
        Returns:
            Dict containing experiment results and metadata, or None if failed
        """
        logger.info(f"Starting baseline experiment: {config}")
        
        # Validate configuration
        if not self.validate_experiment_config(config):
            logger.error("Experiment configuration validation failed")
            return None
        
        try:
            # Step 1: Prepare dataset
            df = self.prepare_dataset(config['dataset'], config['size'])
            
            # Ensure we only process the requested number of samples
            df_slice = df.head(config['size'])
            
            # Validate dataset has required fields
            if not self.dataset_manager.validate_dataset_fields(config['dataset']):
                logger.error(f"Dataset validation failed for {config['dataset']}")
                return None
            
            # Step 2: Load model
            self.load_model(config['model'])
            
            # Step 3: Prepare prompts and expected outputs
            prompts = []
            expected_outputs = []
            
            for idx, row in df_slice.iterrows():
                try:
                    # Handle few-shot prompts specially
                    if 'few_shot' in config['prompt']:
                        # Generate few-shot examples from other samples
                        few_shot_examples = self.prompt_manager.generate_few_shot_examples(
                            df, config['dataset'], n_examples=3, exclude_indices=[idx]
                        )
                        additional_vars = {'few_shot_examples': few_shot_examples}
                    else:
                        additional_vars = {}
                    
                    # Prepare prompt for this row
                    prompt = self.prompt_manager.prepare_prompt_for_row(
                        config['prompt'], row, config['dataset'], additional_vars
                    )
                    prompts.append(prompt)
                    
                    # Get expected answer using dataset manager
                    expected_output = self.dataset_manager.get_expected_answer(row, config['dataset'])
                    expected_outputs.append(expected_output)
                    
                except Exception as e:
                    logger.error(f"Error preparing prompt for row {idx}: {e}")
                    prompts.append("")  # Fallback empty prompt
                    expected_outputs.append("")
            
            logger.info(f"Prepared {len(prompts)} prompts")
            
            # Step 4: Generate responses
            responses = self.generate_responses(prompts, expected_outputs, config['model'], config['temperature'])
            
            # Step 5: Save results
            output_file = self.save_experiment_results(config, df, responses)
            
            # Clean up GPU memory if using local models
            if self.models_config[config['model']]['type'] == 'local':
                clear_gpu_memory()
            
            # Return success summary
            experiment_name = Config.generate_experiment_name(
                config['experiment_type'], config['dataset'], config['model'], 
                config['prompt'], config['size'], config['temperature']
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
            logger.error(f"Baseline experiment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None