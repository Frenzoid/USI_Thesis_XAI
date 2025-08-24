import os
import json
import glob
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from config import Config
from utils import setup_logging
from evaluation import EvaluationFramework
from models import ModelManager

logger = setup_logging("evaluation_runner")

class EvaluationRunner:
    """Handles evaluating experiment results using various metrics"""
    
    def __init__(self):
        self.evaluation_framework = EvaluationFramework()
        self.model_manager = ModelManager()
        
        # Load embedding model for semantic similarity
        self.model_manager.load_embedding_model()
        
        logger.info("EvaluationRunner initialized")
    
    def find_experiment_files(self, experiment_type: Optional[str] = None) -> List[str]:
        """Find all experiment result files"""
        if experiment_type:
            if not Config.validate_experiment_type(experiment_type):
                raise ValueError(f"Invalid experiment type: {experiment_type}")
            
            search_dir = Config.get_output_dirs_for_experiment_type(experiment_type)['responses']
            pattern = os.path.join(search_dir, "inference_*.json")
        else:
            # Search all experiment types
            pattern = os.path.join(Config.RESPONSES_DIR, "**", "inference_*.json")
        
        files = glob.glob(pattern, recursive=True)
        logger.info(f"Found {len(files)} experiment files")
        return files
    
    def load_experiment_results(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load experiment results from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.debug(f"Loaded experiment results from: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading experiment results from {file_path}: {e}")
            return None
    
    def extract_responses_and_expected(self, experiment_data: Dict[str, Any]) -> tuple:
        """Extract generated responses and expected outputs from experiment data"""
        responses_data = experiment_data.get('responses', [])
        expected_outputs = experiment_data.get('expected_outputs', [])
        
        # Extract just the response text from response objects
        generated_responses = []
        for response_obj in responses_data:
            if isinstance(response_obj, dict):
                response_text = response_obj.get('response', '')
                generated_responses.append(response_text)
            else:
                generated_responses.append(str(response_obj))
        
        # Ensure both lists have the same length
        min_length = min(len(generated_responses), len(expected_outputs))
        if min_length != len(generated_responses) or min_length != len(expected_outputs):
            logger.warning(f"Mismatched lengths - Generated: {len(generated_responses)}, Expected: {len(expected_outputs)}")
            generated_responses = generated_responses[:min_length]
            expected_outputs = expected_outputs[:min_length]
        
        logger.debug(f"Extracted {len(generated_responses)} response pairs")
        return generated_responses, expected_outputs
    
    def determine_dataset_type(self, experiment_data: Dict[str, Any]) -> str:
        """Determine dataset type for evaluation metrics"""
        dataset_name = experiment_data.get('experiment_config', {}).get('dataset', 'general')
        
        # Map dataset names to evaluation types
        dataset_type_mapping = {
            'gmeg': 'gmeg',
            'xai_fungi': 'general',
            'hatebrxplain': 'general', 
            'explanationhardness': 'general',
            'reframinghumanai': 'general'
        }
        
        mapped_type = dataset_type_mapping.get(dataset_name, 'general')
        logger.debug(f"Mapped dataset '{dataset_name}' to evaluation type '{mapped_type}'")
        return mapped_type
    
    def evaluate_experiment_from_data(self, experiment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate experiment results from loaded data"""
        try:
            # Extract responses and expected outputs
            generated_responses, expected_outputs = self.extract_responses_and_expected(experiment_data)
            
            if not generated_responses or not expected_outputs:
                logger.error("No valid responses found for evaluation")
                return None
            
            # Determine dataset type for appropriate metrics
            dataset_type = self.determine_dataset_type(experiment_data)
            
            # Run evaluation
            experiment_name = experiment_data.get('experiment_name', 'unknown')
            logger.info(f"Evaluating experiment: {experiment_name}")
            
            evaluation_result = self.evaluation_framework.evaluate_batch(
                generated_responses=generated_responses,
                expected_responses=expected_outputs,
                embedding_model=self.model_manager.embedding_model,
                batch_name=experiment_name,
                dataset_type=dataset_type
            )
            
            # Add experiment metadata
            evaluation_result.update({
                'original_experiment_config': experiment_data.get('experiment_config', {}),
                'original_experiment_name': experiment_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'dataset_type': dataset_type
            })
            
            logger.info(f"Evaluation completed for: {experiment_name}")
            
            # Log key metrics
            if evaluation_result.get('aggregated_scores'):
                agg_scores = evaluation_result['aggregated_scores']
                f1_mean = agg_scores.get('f1_score', {}).get('mean', 0)
                sem_sim_mean = agg_scores.get('semantic_similarity', {}).get('mean', 0)
                exact_match_mean = agg_scores.get('exact_match', {}).get('mean', 0)
                
                logger.info(f"Results - F1: {f1_mean:.3f}, Semantic Similarity: {sem_sim_mean:.3f}, Exact Match: {exact_match_mean:.3f}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating experiment: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def save_evaluation_results(self, evaluation_data: Dict[str, Any], experiment_name: str, 
                              experiment_type: str) -> str:
        """Save evaluation results to file"""
        
        # Generate file path
        file_paths = Config.generate_file_paths(experiment_type, experiment_name)
        output_file = file_paths['evaluation']
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(evaluation_data, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            raise
    
    def evaluate_experiment(self, experiment_name: str, experiment_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Evaluate a specific experiment by name"""
        logger.info(f"Evaluating experiment: {experiment_name}")
        
        # Find the experiment file
        if experiment_type:
            search_dirs = [Config.get_output_dirs_for_experiment_type(experiment_type)['responses']]
        else:
            # Search all experiment types
            search_dirs = []
            for exp_type in Config.EXPERIMENT_TYPES:
                search_dirs.append(Config.get_output_dirs_for_experiment_type(exp_type)['responses'])
        
        experiment_file = None
        for search_dir in search_dirs:
            potential_file = os.path.join(search_dir, f"inference_{experiment_name}.json")
            if os.path.exists(potential_file):
                experiment_file = potential_file
                break
        
        if not experiment_file:
            logger.error(f"Experiment file not found: {experiment_name}")
            return None
        
        # Load experiment data
        experiment_data = self.load_experiment_results(experiment_file)
        if not experiment_data:
            return None
        
        # Evaluate
        evaluation_result = self.evaluate_experiment_from_data(experiment_data)
        if not evaluation_result:
            return None
        
        # Determine experiment type from file path if not provided
        if not experiment_type:
            for exp_type in Config.EXPERIMENT_TYPES:
                if exp_type in experiment_file:
                    experiment_type = exp_type
                    break
            else:
                experiment_type = 'baseline'  # Default
        
        # Save evaluation results
        output_file = self.save_evaluation_results(evaluation_result, experiment_name, experiment_type)
        
        return {
            'experiment_name': experiment_name,
            'experiment_type': experiment_type,
            'evaluation_file': output_file,
            'metrics': evaluation_result.get('aggregated_scores', {}),
            'num_samples': evaluation_result.get('num_samples', 0),
            'num_valid_evaluations': evaluation_result.get('num_valid_evaluations', 0)
        }
    
    def evaluate_all_experiments(self, experiment_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Evaluate all experiments of the specified type"""
        logger.info(f"Evaluating all experiments (type: {experiment_type or 'all'})")
        
        # Find all experiment files
        experiment_files = self.find_experiment_files(experiment_type)
        
        if not experiment_files:
            logger.warning("No experiment files found")
            return []
        
        results = []
        for file_path in tqdm(experiment_files, desc="Evaluating experiments"):
            try:
                # Extract experiment name from file path
                file_name = os.path.basename(file_path)
                if file_name.startswith('inference_'):
                    experiment_name = file_name[10:-5]  # Remove 'inference_' prefix and '.json' suffix
                else:
                    experiment_name = file_name[:-5]  # Remove '.json' suffix
                
                # Load and evaluate
                experiment_data = self.load_experiment_results(file_path)
                if not experiment_data:
                    continue
                
                evaluation_result = self.evaluate_experiment_from_data(experiment_data)
                if not evaluation_result:
                    continue
                
                # Determine experiment type from file path
                current_experiment_type = experiment_type
                if not current_experiment_type:
                    for exp_type in Config.EXPERIMENT_TYPES:
                        if exp_type in file_path:
                            current_experiment_type = exp_type
                            break
                    else:
                        current_experiment_type = 'baseline'  # Default
                
                # Save evaluation results
                output_file = self.save_evaluation_results(evaluation_result, experiment_name, current_experiment_type)
                
                results.append({
                    'experiment_name': experiment_name,
                    'experiment_type': current_experiment_type,
                    'evaluation_file': output_file,
                    'metrics': evaluation_result.get('aggregated_scores', {}),
                    'num_samples': evaluation_result.get('num_samples', 0),
                    'num_valid_evaluations': evaluation_result.get('num_valid_evaluations', 0)
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {file_path}: {e}")
                continue
        
        logger.info(f"Successfully evaluated {len(results)} out of {len(experiment_files)} experiments")
        return results
    
    def get_evaluation_summary(self, experiment_type: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of evaluation results"""
        # Find evaluation files
        if experiment_type:
            search_dir = Config.get_output_dirs_for_experiment_type(experiment_type)['evaluations']
            pattern = os.path.join(search_dir, "evaluation_*.json")
        else:
            pattern = os.path.join(Config.EVALUATIONS_DIR, "**", "evaluation_*.json")
        
        evaluation_files = glob.glob(pattern, recursive=True)
        
        if not evaluation_files:
            return {"message": "No evaluation results found"}
        
        # Load and summarize evaluations
        all_f1_scores = []
        all_semantic_similarities = []
        experiment_count = 0
        total_samples = 0
        
        for file_path in evaluation_files:
            try:
                with open(file_path, 'r') as f:
                    eval_data = json.load(f)
                
                agg_scores = eval_data.get('aggregated_scores', {})
                if 'f1_score' in agg_scores:
                    all_f1_scores.append(agg_scores['f1_score']['mean'])
                if 'semantic_similarity' in agg_scores:
                    all_semantic_similarities.append(agg_scores['semantic_similarity']['mean'])
                
                experiment_count += 1
                total_samples += eval_data.get('num_valid_evaluations', 0)
                
            except Exception as e:
                logger.warning(f"Error loading evaluation file {file_path}: {e}")
                continue
        
        summary = {
            'total_evaluations': experiment_count,
            'total_samples_evaluated': total_samples,
            'experiment_type_filter': experiment_type
        }
        
        if all_f1_scores:
            summary['f1_score_stats'] = {
                'mean': np.mean(all_f1_scores),
                'std': np.std(all_f1_scores),
                'min': np.min(all_f1_scores),
                'max': np.max(all_f1_scores)
            }
        
        if all_semantic_similarities:
            summary['semantic_similarity_stats'] = {
                'mean': np.mean(all_semantic_similarities),
                'std': np.std(all_semantic_similarities),
                'min': np.min(all_semantic_similarities),
                'max': np.max(all_semantic_similarities)
            }
        
        return summary