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
    """
    Handles evaluating experiment results using comprehensive metrics with metadata-based detection.
    
    This class orchestrates the evaluation pipeline:
    1. Finds and loads experiment result files
    2. Extracts metadata from file content (not filename parsing)
    3. Extracts model responses and expected outputs
    4. Runs evaluation using the EvaluationFramework with custom metrics
    5. Saves evaluation results with metadata
    6. Provides batch evaluation across multiple experiments
    """
    
    def __init__(self):
        """Initialize evaluation runner with framework and embedding model"""
        self.evaluation_framework = EvaluationFramework()
        self.model_manager = ModelManager()
        
        # Load embedding model for semantic similarity calculations
        self.model_manager.load_embedding_model()
        
        logger.info("EvaluationRunner initialized")
    
    # =============================================================================
    # METADATA EXTRACTION FROM FILE CONTENT
    # =============================================================================
    
    def extract_metadata_from_file_data(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract experiment metadata from loaded file data instead of filename parsing.
        
        Args:
            experiment_data: Loaded experiment result data
            
        Returns:
            dict: Extracted metadata including experiment_type, mode, model, etc.
        """
        # For inference result files, metadata is in 'experiment_config'
        metadata = experiment_data.get('experiment_config', {})
        
        if not metadata:
            # Fallback: check if it's an evaluation file with 'original_experiment_config'
            metadata = experiment_data.get('original_experiment_config', {})
        
        # Extract key fields with defaults
        extracted = {
            'experiment_type': metadata.get('experiment_type', 'baseline'),
            'mode': metadata.get('mode', 'zero-shot'),
            'model': metadata.get('model', 'unknown'),
            'dataset': metadata.get('dataset', 'unknown'),
            'prompt': metadata.get('prompt', 'unknown'),
            'size': metadata.get('size', 0),
            'temperature': metadata.get('temperature', 0.1),
            'few_shot_row': metadata.get('few_shot_row')
        }
        
        logger.debug(f"Extracted metadata from file: {extracted}")
        return extracted
    
    # =============================================================================
    # EXPERIMENT FILE DISCOVERY AND LOADING
    # =============================================================================
    
    def find_experiment_files(self, experiment_type: Optional[str] = None) -> List[str]:
        """
        Find all experiment result files matching the specified type.
        
        Args:
            experiment_type: Type of experiments to find (e.g., 'baseline')
                           If None, searches all experiment types
            
        Returns:
            list: Paths to found experiment files
            
        Raises:
            ValueError: If experiment type is invalid
        """
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
        """
        Load experiment results from a JSON file.
        
        Args:
            file_path: Path to experiment result file
            
        Returns:
            dict: Loaded experiment data, or None if failed
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.debug(f"Loaded experiment results from: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading experiment results from {file_path}: {e}")
            return None
    
    # =============================================================================
    # DATA EXTRACTION AND PROCESSING
    # =============================================================================
    
    def extract_responses_and_expected(self, experiment_data: Dict[str, Any]) -> tuple:
        """
        Extract generated responses and expected outputs from experiment data.
        
        Args:
            experiment_data: Loaded experiment result data
            
        Returns:
            tuple: (generated_responses, expected_outputs, response_data_list) 
                   as lists of strings and full response data with matching lengths
        """
        responses_data = experiment_data.get('responses', [])
        expected_outputs = experiment_data.get('expected_outputs', [])
        
        # Extract response text from response objects
        generated_responses = []
        full_response_data = []
        
        for response_obj in responses_data:
            if isinstance(response_obj, dict):
                response_text = response_obj.get('response', '')
                generated_responses.append(response_text)
                # Keep the full response data for custom metrics
                full_response_data.append(response_obj)
            else:
                generated_responses.append(str(response_obj))
                # Create minimal response data for legacy format
                full_response_data.append({
                    'response': str(response_obj),
                    'success': True,
                    'error': None,
                    'prompt': '',
                    'question_values': []
                })
        
        # Ensure all lists have the same length for proper evaluation
        min_length = min(len(generated_responses), len(expected_outputs), len(full_response_data))
        
        if min_length != len(generated_responses) or min_length != len(expected_outputs):
            logger.warning(f"Length mismatch detected - Generated: {len(generated_responses)}, Expected: {len(expected_outputs)}")
            logger.warning(f"Truncating to {min_length} pairs for evaluation")
            
            generated_responses = generated_responses[:min_length]
            expected_outputs = expected_outputs[:min_length]
            full_response_data = full_response_data[:min_length]
        
        # Add expected outputs to response data for custom metrics
        for i, response_data in enumerate(full_response_data):
            if i < len(expected_outputs):
                response_data['expected_output'] = expected_outputs[i]
            else:
                response_data['expected_output'] = ''
        
        logger.debug(f"Extracted {len(generated_responses)} response pairs for evaluation")
        return generated_responses, expected_outputs, full_response_data
    
    # =============================================================================
    # CORE EVALUATION METHODS
    # =============================================================================
    
    def evaluate_experiment_from_data(self, experiment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate experiment results from loaded data using metadata-based detection and custom metrics.
        """
        try:
            # Extract responses and expected outputs with full response data
            generated_responses, expected_outputs, response_data_list = self.extract_responses_and_expected(experiment_data)
            
            if not generated_responses or not expected_outputs:
                logger.error("No valid responses found for evaluation")
                return None
            
            # Extract metadata from file content
            metadata = self.extract_metadata_from_file_data(experiment_data)
            experiment_type = metadata['experiment_type']
            dataset_name = metadata['dataset']
            
            # Run comprehensive evaluation with custom metrics
            experiment_name = experiment_data.get('experiment_name', 'unknown')
            logger.info(f"Evaluating experiment: {experiment_name} (type: {experiment_type}, mode: {metadata['mode']}, dataset: {dataset_name})")
            
            evaluation_result = self.evaluation_framework.evaluate_batch(
                generated_responses=generated_responses,
                expected_responses=expected_outputs,
                embedding_model=self.model_manager.embedding_model,
                batch_name=experiment_name,
                dataset_name=dataset_name,
                response_data_list=response_data_list  # Pass full response data for custom metrics
            )
            
            # Add comprehensive metadata including extracted fields
            evaluation_result.update({
                'original_experiment_config': experiment_data.get('experiment_config', {}),
                'original_experiment_name': experiment_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'dataset_name': dataset_name,
                'extracted_metadata': metadata
            })
            
            logger.info(f"Evaluation completed for: {experiment_name}")
            
            # Log key performance metrics
            if evaluation_result.get('aggregated_scores'):
                agg_scores = evaluation_result['aggregated_scores']
                f1_mean = agg_scores.get('f1_score', {}).get('mean', 0)
                sem_sim_mean = agg_scores.get('semantic_similarity', {}).get('mean', 0)
                exact_match_mean = agg_scores.get('exact_match', {}).get('mean', 0)
                
                logger.info(f"Results - F1: {f1_mean:.3f}, Semantic Similarity: {sem_sim_mean:.3f}, Exact Match: {exact_match_mean:.3f}")
                
                # Log custom metrics if any were computed
                custom_metrics = [metric for metric in agg_scores.keys() 
                                if metric not in ['f1_score', 'semantic_similarity', 'exact_match', 'precision', 'recall', 'jaccard']]
                if custom_metrics:
                    logger.info(f"Custom metrics computed: {', '.join(custom_metrics)}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating experiment: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def save_evaluation_results(self, evaluation_data: Dict[str, Any], experiment_name: str, 
                              experiment_type: str) -> str:
        """
        Save evaluation results to JSON file with organized structure.
        
        Args:
            evaluation_data: Complete evaluation results
            experiment_name: Name of the original experiment
            experiment_type: Type of experiment (for directory organization)
            
        Returns:
            str: Path to saved evaluation file
        """
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
    
    # =============================================================================
    # SINGLE EXPERIMENT EVALUATION WITH METADATA-BASED DETECTION
    # =============================================================================
    
    def evaluate_experiment(self, experiment_name: str, experiment_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Evaluate a specific experiment by name with metadata-based detection.
        """
        logger.info(f"Evaluating experiment: {experiment_name}")
        
        # Search across all experiment types since we'll get the actual type from metadata
        experiment_file = None
        
        for exp_type in Config.EXPERIMENT_TYPES:
            search_dir = Config.get_output_dirs_for_experiment_type(exp_type)['responses']
            potential_file = os.path.join(search_dir, f"inference_{experiment_name}.json")
            if os.path.exists(potential_file):
                experiment_file = potential_file
                break
        
        if not experiment_file:
            logger.error(f"Experiment file not found: {experiment_name}")
            return None
        
        # Load and evaluate experiment data
        experiment_data = self.load_experiment_results(experiment_file)
        if not experiment_data:
            return None

        evaluation_result = self.evaluate_experiment_from_data(experiment_data)
        if not evaluation_result:
            return None
        
        # Extract actual metadata from the file content
        metadata = self.extract_metadata_from_file_data(experiment_data)
        final_experiment_type = metadata['experiment_type']
        
        # Save evaluation results using the correct type from metadata
        output_file = self.save_evaluation_results(evaluation_result, experiment_name, final_experiment_type)
        
        return {
            'experiment_name': experiment_name,
            'experiment_type': final_experiment_type,
            'mode': metadata['mode'],
            'model': metadata['model'],
            'dataset': metadata['dataset'],
            'evaluation_file': output_file,
            'metrics': evaluation_result.get('aggregated_scores', {}),
            'num_samples': evaluation_result.get('num_samples', 0),
            'num_valid_evaluations': evaluation_result.get('num_valid_evaluations', 0)
        }
    
    # =============================================================================
    # BATCH EVALUATION WITH METADATA-BASED DETECTION
    # =============================================================================
    
    def evaluate_all_experiments(self, experiment_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Evaluate all experiments with metadata-based filtering.
        """
        logger.info(f"Evaluating all experiments (type filter: {experiment_type or 'none'})")
        
        # Find all experiment files across all types since we'll filter by metadata
        all_experiment_files = []
        for exp_type in Config.EXPERIMENT_TYPES:
            search_dir = Config.get_output_dirs_for_experiment_type(exp_type)['responses']
            pattern = os.path.join(search_dir, "inference_*.json")
            files = glob.glob(pattern)
            all_experiment_files.extend(files)
        
        if not all_experiment_files:
            logger.warning("No experiment files found")
            return []
        
        results = []
        for file_path in tqdm(all_experiment_files, desc="Evaluating experiments"):
            try:
                # Extract experiment name from file path
                file_name = os.path.basename(file_path)
                if file_name.startswith('inference_'):
                    experiment_name = file_name[10:-5]  # Remove 'inference_' prefix and '.json' suffix
                else:
                    experiment_name = file_name[:-5]  # Remove '.json' suffix
                
                # Load and evaluate experiment
                experiment_data = self.load_experiment_results(file_path)
                if not experiment_data:
                    continue
                
                # Extract metadata from file content
                metadata = self.extract_metadata_from_file_data(experiment_data)
                file_experiment_type = metadata['experiment_type']
                
                # Apply experiment type filter if specified
                if experiment_type and file_experiment_type != experiment_type:
                    logger.debug(f"Skipping {experiment_name}: type {file_experiment_type} doesn't match filter {experiment_type}")
                    continue
                
                evaluation_result = self.evaluate_experiment_from_data(experiment_data)
                if not evaluation_result:
                    continue
                
                # Save evaluation results
                output_file = self.save_evaluation_results(evaluation_result, experiment_name, file_experiment_type)
                
                results.append({
                    'experiment_name': experiment_name,
                    'experiment_type': file_experiment_type,
                    'mode': metadata['mode'],
                    'model': metadata['model'],
                    'dataset': metadata['dataset'],
                    'evaluation_file': output_file,
                    'metrics': evaluation_result.get('aggregated_scores', {}),
                    'num_samples': evaluation_result.get('num_samples', 0),
                    'num_valid_evaluations': evaluation_result.get('num_valid_evaluations', 0)
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {file_path}: {e}")
                continue
        
        logger.info(f"Successfully evaluated {len(results)} out of {len(all_experiment_files)} experiments")
        return results
    
    # =============================================================================
    # EVALUATION SUMMARY AND ANALYTICS
    # =============================================================================
    
    def get_evaluation_summary(self, experiment_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive summary of evaluation results across experiments.
        """
        # Find evaluation files across all types since we'll filter by metadata if needed
        all_evaluation_files = []
        for exp_type in Config.EXPERIMENT_TYPES:
            search_dir = Config.get_output_dirs_for_experiment_type(exp_type)['evaluations']
            pattern = os.path.join(search_dir, "evaluation_*.json")
            files = glob.glob(pattern)
            all_evaluation_files.extend(files)
        
        if not all_evaluation_files:
            return {"message": "No evaluation results found"}
        
        # Load and filter evaluations by metadata
        valid_evaluations = []
        for file_path in all_evaluation_files:
            try:
                with open(file_path, 'r') as f:
                    eval_data = json.load(f)
                
                # Extract metadata and filter if experiment_type specified
                metadata = eval_data.get('original_experiment_config', {})
                file_experiment_type = metadata.get('experiment_type', 'baseline')
                
                if experiment_type and file_experiment_type != experiment_type:
                    continue
                
                valid_evaluations.append(eval_data)
                
            except Exception as e:
                logger.warning(f"Error loading evaluation file {file_path}: {e}")
                continue
        
        if not valid_evaluations:
            return {"message": f"No evaluation results found for type: {experiment_type}"}
        
        # Compute summary statistics
        all_f1_scores = []
        all_semantic_similarities = []
        all_exact_matches = []
        total_samples = 0
        
        for eval_data in valid_evaluations:
            agg_scores = eval_data.get('aggregated_scores', {})
            if 'f1_score' in agg_scores:
                all_f1_scores.append(agg_scores['f1_score']['mean'])
            if 'semantic_similarity' in agg_scores:
                all_semantic_similarities.append(agg_scores['semantic_similarity']['mean'])
            if 'exact_match' in agg_scores:
                all_exact_matches.append(agg_scores['exact_match']['mean'])
            
            total_samples += eval_data.get('num_valid_evaluations', 0)
        
        summary = {
            'total_evaluations': len(valid_evaluations),
            'total_samples_evaluated': total_samples,
            'experiment_type_filter': experiment_type
        }
        
        if all_f1_scores:
            summary['f1_score_stats'] = {
                'mean': np.mean(all_f1_scores),
                'std': np.std(all_f1_scores),
                'min': np.min(all_f1_scores),
                'max': np.max(all_f1_scores),
                'median': np.median(all_f1_scores)
            }
        
        if all_semantic_similarities:
            summary['semantic_similarity_stats'] = {
                'mean': np.mean(all_semantic_similarities),
                'std': np.std(all_semantic_similarities),
                'min': np.min(all_semantic_similarities),
                'max': np.max(all_semantic_similarities),
                'median': np.median(all_semantic_similarities)
            }
        
        if all_exact_matches:
            summary['exact_match_stats'] = {
                'mean': np.mean(all_exact_matches),
                'std': np.std(all_exact_matches),
                'min': np.min(all_exact_matches),
                'max': np.max(all_exact_matches),
                'median': np.median(all_exact_matches)
            }
        
        return summary
    
    # =============================================================================
    # EVALUATION COMPARISON AND ANALYSIS
    # =============================================================================
    
    def compare_experiments(self, experiment_names: List[str], experiment_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple experiments side by side with metadata-based detection.
        """
        logger.info(f"Comparing {len(experiment_names)} experiments")
        
        # Load evaluation results for all experiments
        evaluations = []
        for exp_name in experiment_names:
            # Search across all experiment types
            eval_file = None
            for exp_type in Config.EXPERIMENT_TYPES:
                search_dir = Config.get_output_dirs_for_experiment_type(exp_type)['evaluations']
                potential_file = os.path.join(search_dir, f"evaluation_{exp_name}.json")
                if os.path.exists(potential_file):
                    eval_file = potential_file
                    break
            
            if eval_file:
                try:
                    with open(eval_file, 'r') as f:
                        eval_data = json.load(f)
                    
                    # Apply experiment type filter if specified
                    if experiment_type:
                        metadata = eval_data.get('original_experiment_config', {})
                        file_experiment_type = metadata.get('experiment_type', 'baseline')
                        if file_experiment_type != experiment_type:
                            logger.info(f"Skipping {exp_name}: type {file_experiment_type} doesn't match filter {experiment_type}")
                            continue
                    
                    evaluations.append(eval_data)
                except Exception as e:
                    logger.error(f"Error loading evaluation for {exp_name}: {e}")
            else:
                logger.warning(f"Evaluation file not found for: {exp_name}")
        
        if len(evaluations) < 2:
            return {"error": "Need at least 2 valid evaluations for comparison"}
        
        # Create comparison matrix
        comparison = {
            'experiment_names': [eval_data.get('original_experiment_name', 'Unknown') for eval_data in evaluations],
            'metric_comparison': {},
            'rankings': {},
            'best_performers': {}
        }
        
        # Extract metrics for comparison
        common_metrics = set()
        for eval_data in evaluations:
            agg_scores = eval_data.get('aggregated_scores', {})
            common_metrics.update(agg_scores.keys())
        
        # Compare each metric across experiments
        for metric in common_metrics:
            values = []
            for eval_data in evaluations:
                agg_scores = eval_data.get('aggregated_scores', {})
                if metric in agg_scores:
                    values.append(agg_scores[metric]['mean'])
                else:
                    values.append(0.0)
            
            comparison['metric_comparison'][metric] = values
            
            # Rank experiments for this metric
            ranked_indices = np.argsort(values)[::-1]  # Descending order
            comparison['rankings'][metric] = [comparison['experiment_names'][i] for i in ranked_indices]
            
            # Best performer for this metric
            best_idx = np.argmax(values)
            comparison['best_performers'][metric] = {
                'experiment': comparison['experiment_names'][best_idx],
                'value': values[best_idx]
            }
        
        return comparison