import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from config import Config
from utils import setup_logging, experiment_context, save_results, load_results
from models import ModelManager
from prompts import PromptManager
from datasets_manager import DatasetManager
from evaluation import EvaluationFramework
from visualization import VisualizationFramework

logger = setup_logging("experiments")

class ExperimentRunner:
    """Main experiment runner that orchestrates all components"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.prompt_manager = PromptManager()
        self.dataset_manager = DatasetManager(self.prompt_manager)
        self.evaluator = EvaluationFramework()
        self.visualizer = VisualizationFramework()
        
        self.results_history = []
        
        logger.info("ExperimentRunner initialized")
    
    def initialize_system(self):
        """Initialize all system components"""
        logger.info("Initializing experiment system...")
        
        # Setup API clients
        self.model_manager.setup_api_clients()
        
        # Load embedding model
        self.model_manager.load_embedding_model()
        
        logger.info("System initialization completed")
    
    def generate_experiment_id(self, model_name: str, model_type: str, prompt_key: str, 
                              sample_size: int, dataset_type: str = 'gmeg') -> str:
        """Generate unique identifier for experiment configuration"""
        return f"{model_name}_{model_type}_{prompt_key}_{sample_size}_{dataset_type}"
    
    def experiment_exists(self, experiment_id: str) -> bool:
        """Check if experiment with same parameters already exists"""
        if not os.path.exists(Config.RESULTS_DIR):
            return False
        
        for filename in os.listdir(Config.RESULTS_DIR):
            if filename.endswith('.json') and experiment_id in filename:
                logger.info(f"Found existing experiment: {filename}")
                return True
        return False
    
    def load_existing_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Load existing experiment result if it exists"""
        if not os.path.exists(Config.RESULTS_DIR):
            return None
        
        for filename in os.listdir(Config.RESULTS_DIR):
            if filename.endswith('.json') and experiment_id in filename:
                filepath = os.path.join(Config.RESULTS_DIR, filename)
                try:
                    result = load_results(filepath)
                    logger.info(f"Loaded existing experiment from {filename}")
                    return result
                except Exception as e:
                    logger.error(f"Error loading existing experiment {filename}: {e}")
                    return None
        return None
    
    def run_single_experiment(self, model_name: str, model_type: str, prompt_key: str, 
                            sample_size: int = 50, experiment_name: str = None, 
                            dataset_type: str = 'gmeg', dataset_name: str = 'gmeg',
                            force_rerun: bool = False, **kwargs) -> Optional[Dict]:
        """
        Run a single experiment with the specified configuration
        
        Args:
            model_name: Name/key of the model to use
            model_type: 'open_source' or 'api'
            prompt_key: Key for the prompt template to use
            sample_size: Number of samples to evaluate
            experiment_name: Name for this experiment
            dataset_type: Type of dataset being used
            dataset_name: Name of the dataset to load
            force_rerun: If True, run experiment even if it already exists
            **kwargs: Additional arguments
        
        Returns:
            Dict: Experiment results or None if failed
        """
        
        # Generate experiment ID for checking duplicates
        experiment_id = self.generate_experiment_id(model_name, model_type, prompt_key, sample_size, dataset_type)
        
        if experiment_name is None:
            experiment_name = experiment_id
        
        # Check if experiment already exists
        if not force_rerun and self.experiment_exists(experiment_id):
            logger.info(f"Experiment {experiment_id} already exists. Skipping...")
            existing_result = self.load_existing_experiment(experiment_id)
            if existing_result:
                self.results_history.append(existing_result)
                return existing_result
            else:
                logger.warning("Could not load existing result. Running experiment...")
        
        with experiment_context(experiment_name):
            try:
                # Load dataset
                dataset = self.dataset_manager.load_dataset(dataset_name)
                if dataset is None:
                    logger.error(f"Failed to load dataset '{dataset_name}'")
                    return None
                
                # Validate dataset
                if not self.dataset_manager.validate_dataset(dataset, dataset_type):
                    logger.warning("Dataset validation failed. Continuing with available fields...")
                
                # Check if model is available
                if not self.model_manager.is_model_available(model_name, model_type):
                    logger.error(f"Model '{model_name}' of type '{model_type}' is not available")
                    return None
                
                # Load model if needed
                if model_type == 'open_source':
                    logger.info(f"Loading open-source model: {model_name}")
                    self.model_manager.load_open_source_model(model_name)
                
                # Prepare dataset for experiment
                logger.info(f"Preparing dataset for experiment with {sample_size} samples")
                prepared_data = self.dataset_manager.prepare_dataset_for_experiment(
                    dataset_name, prompt_key, sample_size
                )
                
                if not prepared_data['prompts']:
                    logger.error("No prompts prepared successfully")
                    return None
                
                # Generate responses
                logger.info(f"Generating responses using {model_name} ({model_type})")
                generated_responses = []
                processing_times = []
                
                for i, prompt in enumerate(tqdm(prepared_data['prompts'], desc=f"Generating responses ({model_name})")):
                    start_time = time.time()
                    
                    try:
                        if model_type == 'open_source':
                            response = self.model_manager.query_open_source(prompt)
                        elif model_type == 'api':
                            if 'gpt' in model_name:
                                response = self.model_manager.query_openai(prompt, model_name)
                            elif 'gemini' in model_name:
                                response = self.model_manager.query_genai(prompt, model_name)
                            else:
                                logger.error(f"Unknown API model: {model_name}")
                                response = f"Error: Unknown API model {model_name}"
                        else:
                            logger.error(f"Unknown model type: {model_type}")
                            response = f"Error: Unknown model type {model_type}"
                        
                        generated_responses.append(response)
                        processing_time = time.time() - start_time
                        processing_times.append(processing_time)
                        
                        logger.debug(f"Generated response {i+1}/{len(prepared_data['prompts'])}")
                        
                    except Exception as e:
                        logger.error(f"Error generating response for sample {i}: {e}")
                        generated_responses.append(f"Error: {str(e)}")
                        processing_times.append(time.time() - start_time)
                        continue
                
                # Evaluate responses
                logger.info("Evaluating generated responses...")
                evaluation_result = self.evaluator.evaluate_batch(
                    generated_responses, 
                    prepared_data['expected_outputs'], 
                    self.model_manager.embedding_model, 
                    batch_name=experiment_name, 
                    dataset_type=dataset_type
                )
                
                # Add detailed experiment metadata
                evaluation_result.update({
                    'experiment_id': experiment_id,
                    'model_name': model_name,
                    'model_type': model_type,
                    'prompt_key': prompt_key,
                    'sample_size': sample_size,
                    'dataset_type': dataset_type,
                    'dataset_name': dataset_name,
                    'experiment_details': {
                        'prompts': prepared_data['prompts'],
                        'generated_responses': generated_responses,
                        'expected_responses': prepared_data['expected_outputs'],
                        'sample_indices': prepared_data['sample_indices'],
                        'processing_times': processing_times
                    },
                    'processing_stats': {
                        'total_processing_time': sum(processing_times),
                        'avg_processing_time': np.mean(processing_times),
                        'min_processing_time': min(processing_times),
                        'max_processing_time': max(processing_times)
                    }
                })
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = save_results(evaluation_result, experiment_id, timestamp)
                
                # Log results summary
                self._log_experiment_results(evaluation_result)
                
                # Add to history
                self.results_history.append(evaluation_result)
                
                return evaluation_result
                
            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
    
    def run_experiment_batch(self, experiment_configs: List[Dict], 
                           auto_visualize: bool = True) -> List[Dict]:
        """
        Run a batch of experiments from configuration list
        
        Args:
            experiment_configs: List of experiment configuration dictionaries
            auto_visualize: Whether to automatically generate visualizations
        
        Returns:
            List of experiment results
        """
        logger.info(f"Starting experiment batch with {len(experiment_configs)} experiments")
        
        batch_results = []
        successful_experiments = 0
        failed_experiments = 0
        
        for i, config in enumerate(experiment_configs, 1):
            logger.info(f"Running experiment {i}/{len(experiment_configs)}")
            logger.info(f"Config: {config}")
            
            try:
                result = self.run_single_experiment(**config)
                if result:
                    batch_results.append(result)
                    successful_experiments += 1
                    logger.info(f"Experiment {i} completed successfully")
                else:
                    failed_experiments += 1
                    logger.error(f"Experiment {i} failed")
                    
            except Exception as e:
                failed_experiments += 1
                logger.error(f"Experiment {i} failed with error: {e}")
                continue
        
        logger.info(f"Batch completed: {successful_experiments} successful, {failed_experiments} failed")
        
        # Auto-generate visualizations if requested
        if auto_visualize and batch_results:
            logger.info("Generating automatic visualizations...")
            try:
                self.auto_generate_visualizations(batch_results)
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")
        
        return batch_results
    
    def auto_generate_visualizations(self, results: List[Dict]):
        """Automatically generate all visualizations for experiment results"""
        if not results:
            logger.warning("No results to visualize")
            return
        
        logger.info(f"Generating visualizations for {len(results)} experiments...")
        
        # Create main visualization directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = os.path.join(Config.PLOTS_DIR, f"experiment_analysis_{timestamp}")
        os.makedirs(viz_dir, exist_ok=True)
        
        try:
            # Generate comprehensive report
            plots, report_dir = self.visualizer.create_experiment_report(
                results, f"batch_analysis_{timestamp}"
            )
            
            # Create summary CSV
            csv_path = self.visualizer.create_summary_csv(
                results, 
                os.path.join(viz_dir, "experiment_summary.csv")
            )
            
            logger.info(f"Visualization complete!")
            logger.info(f"Main directory: {viz_dir}")
            logger.info(f"Report directory: {report_dir}")
            logger.info(f"Summary CSV: {csv_path}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _log_experiment_results(self, result: Dict):
        """Log experiment results summary"""
        experiment_name = result.get('batch_name', 'Unknown')
        agg_scores = result.get('aggregated_scores', {})
        proc_stats = result.get('processing_stats', {})
        
        logger.info(f"=== RESULTS FOR {experiment_name} ===")
        
        if agg_scores:
            for metric, stats in agg_scores.items():
                logger.info(f"{metric.replace('_', ' ').title()}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        else:
            logger.warning("No valid scores to aggregate (all samples may have been marked as NA)")
        
        if proc_stats:
            logger.info(f"=== PROCESSING STATS ===")
            logger.info(f"Total time: {proc_stats['total_processing_time']:.2f}s")
            logger.info(f"Average per sample: {proc_stats['avg_processing_time']:.2f}s")
            logger.info(f"Range: {proc_stats['min_processing_time']:.2f}s - {proc_stats['max_processing_time']:.2f}s")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all experiment results"""
        if not self.results_history:
            return {"message": "No experiments completed yet"}
        
        summary = {
            'total_experiments': len(self.results_history),
            'successful_experiments': len([r for r in self.results_history if r.get('aggregated_scores')]),
            'total_samples': sum(r.get('num_samples', 0) for r in self.results_history),
            'total_valid_evaluations': sum(r.get('num_valid_evaluations', 0) for r in self.results_history),
            'models_tested': list(set(r.get('model_name') for r in self.results_history if r.get('model_name'))),
            'prompts_tested': list(set(r.get('prompt_key') for r in self.results_history if r.get('prompt_key'))),
            'datasets_used': list(set(r.get('dataset_name') for r in self.results_history if r.get('dataset_name')))
        }
        
        # Find best performing experiment
        if self.results_history:
            valid_results = [r for r in self.results_history if r.get('aggregated_scores')]
            if valid_results:
                best_f1 = max(valid_results, 
                             key=lambda x: x.get('aggregated_scores', {}).get('f1_score', {}).get('mean', 0))
                summary['best_performance'] = {
                    'experiment': best_f1.get('batch_name'),
                    'f1_score': best_f1.get('aggregated_scores', {}).get('f1_score', {}).get('mean', 0),
                    'model': best_f1.get('model_name'),
                    'prompt': best_f1.get('prompt_key')
                }
        
        return summary
    
    def load_experiments_from_config(self, config_file: str) -> List[Dict]:
        """Load experiment configurations from JSON file"""
        logger.info(f"Loading experiment configurations from {config_file}")
        
        try:
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            experiments = config_data.get('experiments', [])
            logger.info(f"Loaded {len(experiments)} experiment configurations")
            
            return experiments
            
        except Exception as e:
            logger.error(f"Error loading experiment config from {config_file}: {e}")
            return []
    
    def validate_experiment_config(self, config: Dict) -> bool:
        """Validate a single experiment configuration"""
        required_fields = ['model_name', 'model_type', 'prompt_key']
        optional_fields = ['sample_size', 'dataset_type', 'dataset_name', 'experiment_name']
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field '{field}' in experiment config")
                return False
        
        # Validate model availability
        model_name = config['model_name']
        model_type = config['model_type']
        
        if not self.model_manager.is_model_available(model_name, model_type):
            logger.error(f"Model '{model_name}' of type '{model_type}' is not available")
            return False
        
        # Validate prompt key
        prompt_key = config['prompt_key']
        if prompt_key not in self.prompt_manager.prompts:
            logger.error(f"Unknown prompt key: {prompt_key}")
            return False
        
        return True