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
    
    Orchestrates the complete experiment pipeline:
    1. Prepares datasets and prompts with row pruning
    2. Loads and queries models (local or API) with authentication handling
    3. Saves results with comprehensive metadata
    """
    
    def __init__(self):
        """Initialize all component managers"""
        self.model_manager = ModelManager()
        self.prompt_manager = PromptManager()
        self.dataset_manager = DatasetManager()
        
        # Load configurations
        self.prompts_config = Config.load_prompts_config()
        self.datasets_config = Config.load_datasets_config()
        self.models_config = Config.load_models_config()
        
        # Initialize API clients
        self.model_manager.setup_api_clients()
        
        logger.info("ExperimentRunner initialized")
    
    def validate_experiment_config(self, config: Dict[str, Any]) -> bool:
        """Validate experiment configuration for required fields"""
        required_fields = ['experiment_type', 'model', 'dataset', 'prompt', 'mode', 'size', 'temperature']
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate values exist in configs
        if config['experiment_type'] not in Config.EXPERIMENT_TYPES:
            logger.error(f"Invalid experiment type: {config['experiment_type']}")
            return False
        
        if config['model'] not in self.models_config:
            logger.error(f"Unknown model: {config['model']}")
            return False
        
        if config['dataset'] not in self.datasets_config:
            logger.error(f"Unknown dataset: {config['dataset']}")
            return False
        
        if config['prompt'] not in self.prompts_config:
            logger.error(f"Unknown prompt: {config['prompt']}")
            return False
        
        if config['mode'] not in ['zero-shot', 'few-shot']:
            logger.error(f"Invalid mode: {config['mode']}")
            return False
        
        # Validate prompt-dataset-mode compatibility
        prompt_config = self.prompts_config[config['prompt']]
        
        if prompt_config.get('compatible_dataset') != config['dataset']:
            logger.error(f"Prompt '{config['prompt']}' not compatible with dataset '{config['dataset']}'")
            return False
        
        if prompt_config.get('mode', 'zero-shot') != config['mode']:
            logger.error(f"Prompt '{config['prompt']}' is {prompt_config.get('mode', 'zero-shot')} but {config['mode']} requested")
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
    
    def check_model_accessibility(self, model_name: str) -> Tuple[bool, str]:
        """
        Check if a model is accessible before trying to load it.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Tuple[bool, str]: (is_accessible, reason_if_not)
        """
        if model_name not in self.models_config:
            return False, f"Model '{model_name}' not found in configuration"
        
        model_config = self.models_config[model_name]
        
        # API models are generally accessible if we have API keys
        if model_config['type'] == 'api':
            provider = model_config.get('provider', '')
            if provider == 'openai' and not Config.OPENAI_API_KEY:
                return False, "OpenAI API key not configured"
            elif provider == 'google' and not Config.GENAI_API_KEY:
                return False, "Google GenAI API key not configured"
            elif provider == 'anthropic' and not Config.ANTHROPIC_API_KEY:
                return False, "Anthropic API key not configured"
            return True, ""
        
        # For local models, check accessibility
        elif model_config['type'] == 'local':
            model_path = model_config.get('model_path', '')
            if not model_path:
                return False, "No model_path specified for local model"
            
            # Use the model manager's accessibility check
            try:
                is_accessible = self.model_manager._check_model_accessibility(model_path, model_name)
                if not is_accessible:
                    if not Config.HF_ACCESS_TOKEN:
                        return False, "Model may be gated/restricted - HF_ACCESS_TOKEN required"
                    else:
                        return False, "Model not accessible with current HF_ACCESS_TOKEN"
                return True, ""
            except Exception as e:
                return False, f"Error checking accessibility: {e}"
        
        return False, f"Unknown model type: {model_config.get('type', 'unknown')}"
    
    def prepare_dataset(self, dataset_name: str, size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, List[str]]:
        """Load and prepare dataset with row pruning"""
        logger.info(f"Preparing dataset: {dataset_name}")
        
        dataset_config = self.datasets_config[dataset_name]
        
        # Check for either csv_file or parquet_file
        if 'csv_file' in dataset_config:
            file_path = dataset_config['csv_file']
        elif 'parquet_file' in dataset_config:
            file_path = dataset_config['parquet_file']
        else:
            raise ValueError(f"Dataset '{dataset_name}' must specify either 'csv_file' or 'parquet_file'")
        
        dataset_path = os.path.join(Config.DATA_DIR, dataset_config['download_path'], file_path)
        
        # Download if not exists
        if not os.path.exists(dataset_path):
            logger.info(f"Dataset not found, downloading: {dataset_name}")
            if not self.dataset_manager.download_dataset(dataset_name):
                raise ValueError(f"Failed to download dataset: {dataset_name}")
        
        try:
            # Load original dataset - detect file type and use appropriate loader
            if 'csv_file' in dataset_config:
                original_df = pd.read_csv(dataset_path)
                logger.info(f"Loaded CSV dataset with {len(original_df)} rows")
            elif 'parquet_file' in dataset_config:
                original_df = pd.read_parquet(dataset_path)
                logger.info(f"Loaded Parquet dataset with {len(original_df)} rows")
            else:
                raise ValueError(f"Dataset '{dataset_name}' must specify either 'csv_file' or 'parquet_file'")
            
            # Apply row pruning
            filtered_df, pruned_count, prune_reasons = self.dataset_manager.filter_dataset_rows(
                original_df, dataset_name
            )
            logger.info(f"After pruning: {len(filtered_df)} rows remaining, {pruned_count} rows pruned")
            
            if len(filtered_df) == 0:
                raise ValueError(f"All rows in dataset {dataset_name} were pruned")
            
            # Sample data
            if size < len(filtered_df):
                sampled_df = filtered_df.sample(n=size, random_state=Config.RANDOM_SEED).reset_index(drop=True)
                logger.info(f"Sampled {size} rows from filtered dataset")
            else:
                sampled_df = filtered_df.copy()
                logger.info(f"Using all {len(filtered_df)} filtered rows")
            
            return original_df, filtered_df, sampled_df, pruned_count, prune_reasons
            
        except Exception as e:
            logger.error(f"Error preparing dataset {dataset_name}: {e}")
            raise
    
    def load_model_with_error_handling(self, model_name: str) -> Tuple[bool, str]:
        """
        Load a model with comprehensive error handling for authentication issues.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple[bool, str]: (success, error_message_if_failed)
        """
        if model_name not in self.models_config:
            return False, f"Unknown model: {model_name}"
        
        model_config = self.models_config[model_name]
        
        # Check accessibility first
        is_accessible, reason = self.check_model_accessibility(model_name)
        if not is_accessible:
            logger.warning(f"Model {model_name} not accessible: {reason}")
            return False, reason
        
        try:
            if model_config['type'] == 'local':
                if model_config.get('finetuned', False):
                    model_path = os.path.join(Config.FINETUNED_MODELS_DIR, 
                                            model_config['model_path'].split('/')[-1] + '_finetuned')
                    self.model_manager.load_finetuned_model(model_path)
                else:
                    self.model_manager.load_open_source_model(model_name, model_config['model_path'])
                
                logger.info(f"Successfully loaded local model: {model_name}")
                return True, ""
                
            elif model_config['type'] == 'api':
                # API models don't need explicit loading, just verify client setup
                provider = model_config.get('provider', '')
                if provider == 'openai' and not self.model_manager.openai_client:
                    return False, "OpenAI client not initialized"
                elif provider == 'google' and not self.model_manager.genai_client:
                    return False, "Google GenAI client not initialized"
                elif provider == 'anthropic' and not self.model_manager.anthropic_client:
                    return False, "Anthropic client not initialized"
                
                logger.info(f"API model {model_name} ready")
                return True, ""
            
            else:
                return False, f"Unknown model type: {model_config.get('type', 'unknown')}"
                
        except ValueError as e:
            # Handle authentication and accessibility errors
            error_msg = str(e)
            if any(keyword in error_msg.lower() for keyword in ['authentication', 'not accessible', 'gated', 'restricted']):
                logger.error(f"Authentication/access error for model {model_name}: {error_msg}")
                if not Config.HF_ACCESS_TOKEN:
                    return False, f"Model requires authentication - add HF_ACCESS_TOKEN to .env file"
                else:
                    return False, f"Model not accessible with current credentials: {error_msg}"
            else:
                logger.error(f"Model loading error for {model_name}: {error_msg}")
                return False, f"Model loading failed: {error_msg}"
                
        except Exception as e:
            logger.error(f"Unexpected error loading model {model_name}: {e}")
            return False, f"Unexpected error: {e}"
    
    def generate_responses(self, prompts: List[str], expected_outputs: List[str], 
                          model_name: str, temperature: float, question_values_list: List[List[str]]) -> List[Dict[str, Any]]:
        """Generate responses using the loaded model with enhanced error handling"""
        logger.info(f"Generating {len(prompts)} responses using {model_name}")
        
        model_config = self.models_config[model_name]
        max_tokens = model_config.get('max_tokens', Config.MAX_NEW_TOKENS)
        
        responses = []
        processing_times = []
        
        for i, prompt in enumerate(tqdm(prompts, desc=f"Generating ({model_name})")):
            start_time = time.time()
            
            expected_output = expected_outputs[i] if i < len(expected_outputs) else ""
            question_values = question_values_list[i] if i < len(question_values_list) else []
            
            try:
                # Route to appropriate model type
                if model_config['type'] == 'local':
                    response = self.model_manager.query_open_source(
                        prompt, max_tokens=max_tokens, temperature=temperature
                    )
                elif model_config['type'] == 'api':
                    response = self.model_manager.query_api_model(
                        prompt, 
                        provider=model_config['provider'],
                        model=model_config['model_name'],
                        max_tokens=max_tokens, 
                        temperature=temperature
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_config['type']}")
                
                processing_time = time.time() - start_time
                
                # Clean response (remove prompt if echoed)
                if response and response.startswith(prompt):
                    response = response[len(prompt):].strip()
                
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
                
                # Check for authentication-related errors
                if any(keyword in error_msg.lower() for keyword in ['authentication', 'unauthorized', 'forbidden', 'gated', 'restricted']):
                    logger.error(f"Authentication error for response {i+1}/{len(prompts)}: {error_msg}")
                    error_msg = f"Authentication/Access Error: {error_msg}"
                else:
                    logger.error(f"Error generating response {i+1}/{len(prompts)}: {error_msg}")
                
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
        
        # Log statistics
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            success_rate = sum(1 for r in responses if r['success']) / len(responses) * 100
            failed_auth = sum(1 for r in responses if not r['success'] and 'Authentication' in r.get('error', ''))
            
            logger.info(f"Processing complete - Avg: {avg_time:.2f}s, Success: {success_rate:.1f}%")
            if failed_auth > 0:
                logger.warning(f"Authentication failures: {failed_auth} responses")
        
        return responses
    
    def save_experiment_results(self, experiment_config: Dict[str, Any], 
                              df: pd.DataFrame, responses: List[Dict[str, Any]],
                              pruning_stats: Dict[str, Any]) -> str:
        """Save experiment results to JSON file with comprehensive metadata"""
        
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
        
        # Prepare results
        results = {
            'experiment_name': experiment_name,
            'experiment_config': experiment_config,
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'name': experiment_config['dataset'],
                'original_size': pruning_stats.get('original_size', 0),
                'pruned_rows': pruning_stats.get('pruned_count', 0),
                'final_size_before_sampling': pruning_stats.get('filtered_size', len(df)),
                'sampled_size': len(df),
                'columns': list(df.columns)
            },
            'pruning_stats': {
                'rows_pruned': pruning_stats.get('pruned_count', 0),
                'rows_kept_after_pruning': pruning_stats.get('filtered_size', len(df)),
                'prune_reasons_sample': pruning_stats.get('prune_reasons', [])[:10],
                'total_prune_reasons': len(pruning_stats.get('prune_reasons', []))
            },
            'model_info': self.models_config[experiment_config['model']],
            'prompt_info': self.prompts_config[experiment_config['prompt']],
            'responses': responses,
            'expected_outputs': [],
            'processing_stats': {},
            'system_info': {
                'hf_token_configured': bool(Config.HF_ACCESS_TOKEN),
                'gpu_available': False,
                'model_accessibility_checked': True
            }
        }
        
        # Add GPU info if available
        try:
            import torch
            results['system_info']['gpu_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                results['system_info']['gpu_count'] = torch.cuda.device_count()
                results['system_info']['gpu_name'] = torch.cuda.get_device_properties(0).name
        except:
            pass
        
        # Add expected outputs
        for idx, row in df.iterrows():
            expected = self.dataset_manager.get_expected_answer(row, experiment_config['dataset'])
            results['expected_outputs'].append(expected)
        
        # Calculate processing statistics
        processing_times = [r['processing_time'] for r in responses]
        if processing_times:
            auth_failures = sum(1 for r in responses if not r['success'] and 'Authentication' in r.get('error', ''))
            results['processing_stats'] = {
                'total_time': sum(processing_times),
                'avg_time': sum(processing_times) / len(processing_times),
                'min_time': min(processing_times),
                'max_time': max(processing_times),
                'success_count': sum(1 for r in responses if r['success']),
                'error_count': sum(1 for r in responses if not r['success']),
                'authentication_failures': auth_failures
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
        """Run a complete baseline experiment with mode-based prompting and authentication handling"""
        logger.info(f"Starting {config['mode']} baseline experiment: {config}")
        
        # Validate configuration
        if not self.validate_experiment_config(config):
            logger.error("Experiment configuration validation failed")
            return None
        
        # Check model accessibility before proceeding
        model_accessible, access_reason = self.check_model_accessibility(config['model'])
        if not model_accessible:
            logger.error(f"Cannot run experiment: {access_reason}")
            return {
                'experiment_name': 'failed_accessibility_check',
                'config': config,
                'success': False,
                'error': access_reason,
                'accessibility_issue': True
            }
        
        try:
            # Step 1: Prepare dataset with pruning
            original_df, filtered_df, sampled_df, pruned_count, prune_reasons = self.prepare_dataset(
                config['dataset'], config['size']
            )

            pruning_stats = {
                'original_size': len(original_df),
                'pruned_count': pruned_count,
                'filtered_size': len(filtered_df),
                'prune_reasons': prune_reasons
            }

            df_slice = sampled_df.head(config['size'])
            
            # Step 2: Determine few-shot row
            actual_few_shot_row = None
            if config['mode'] == 'few-shot':
                if config.get('few_shot_row') is not None:
                    if config['few_shot_row'] >= len(original_df):
                        logger.error(f"Few-shot row {config['few_shot_row']} out of bounds")
                        return None
                    actual_few_shot_row = config['few_shot_row']
                else:
                    import random
                    random.seed(Config.RANDOM_SEED)
                    actual_few_shot_row = random.randint(0, len(original_df) - 1)
                
                config['few_shot_row'] = actual_few_shot_row
                logger.info(f"Using few-shot row {actual_few_shot_row}")
            
            # Step 3: Load model with enhanced error handling
            model_loaded, load_error = self.load_model_with_error_handling(config['model'])
            if not model_loaded:
                logger.error(f"Failed to load model: {load_error}")
                return {
                    'experiment_name': 'failed_model_loading',
                    'config': config,
                    'success': False,
                    'error': load_error,
                    'model_loading_failed': True
                }
            
            # Step 4: Prepare prompts
            prompts = []
            expected_outputs = []
            question_values_list = []
            
            dataset_config = self.datasets_config[config['dataset']]
            question_fields = dataset_config.get('question_fields', [])
            
            for idx, row in df_slice.iterrows():
                # Extract question field values
                current_question_values = []
                for field in question_fields:
                    if field in row and not pd.isna(row[field]):
                        current_question_values.append(str(row[field]))
                    else:
                        current_question_values.append("")
                        logger.warning(f"Missing field '{field}' in row {idx}")
                
                question_values_list.append(current_question_values)
                
                # Prepare prompt
                try:
                    prompt = self.prompt_manager.prepare_prompt_for_row(
                        prompt_name=config['prompt'],
                        row=row,
                        dataset_name=config['dataset'],
                        mode=config['mode'],
                        dataset=original_df,  # Use original for few-shot
                        few_shot_row=actual_few_shot_row
                    )
                    prompts.append(prompt)
                    
                    # Get expected answer
                    expected_output = self.dataset_manager.get_expected_answer(row, config['dataset'])
                    expected_outputs.append(expected_output)
                    
                except Exception as e:
                    logger.error(f"Error preparing prompt for row {idx}: {e}")
                    prompts.append("")
                    expected_outputs.append("")
                    question_values_list.append([])
            
            logger.info(f"Prepared {len(prompts)} {config['mode']} prompts")
            
            # Step 5: Generate responses with enhanced error handling
            responses = self.generate_responses(prompts, expected_outputs, config['model'], config['temperature'], question_values_list)
            
            # Step 6: Save results
            output_file = self.save_experiment_results(config, df_slice, responses, pruning_stats)
            
            # Clean up GPU memory if using local models
            if self.models_config[config['model']]['type'] == 'local':
                clear_gpu_memory()
            
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
            
            # Check for authentication failures in responses
            auth_failures = sum(1 for r in responses if not r['success'] and 'Authentication' in r.get('error', ''))
            
            result = {
                'experiment_name': experiment_name,
                'config': config,
                'output_file': output_file,
                'success': True,
                'num_responses': len(responses),
                'success_rate': sum(1 for r in responses if r['success']) / len(responses) if responses else 0,
                'authentication_failures': auth_failures,
                'pruning_summary': {
                    'rows_pruned': pruned_count,
                    'rows_kept': len(filtered_df),
                    'rows_sampled': len(df_slice)
                }
            }
            
            if auth_failures > 0:
                logger.warning(f"Experiment completed with {auth_failures} authentication failures")
                result['partial_failure_reason'] = f"{auth_failures} authentication failures"
            
            return result
            
        except Exception as e:
            logger.error(f"{config['mode']} baseline experiment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Check if it's an authentication-related error
            error_msg = str(e)
            if any(keyword in error_msg.lower() for keyword in ['authentication', 'gated', 'restricted', 'unauthorized']):
                return {
                    'experiment_name': 'failed_authentication',
                    'config': config,
                    'success': False,
                    'error': f"Authentication error: {error_msg}",
                    'authentication_error': True
                }
            
            return None