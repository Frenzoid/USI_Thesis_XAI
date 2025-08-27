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
    Handles evaluating experiment results using comprehensive metrics.
    
    This class orchestrates the evaluation pipeline:
    1. Finds and loads experiment result files
    2. Extracts model responses and expected outputs
    3. Runs evaluation using the EvaluationFramework
    4. Saves evaluation results with metadata
    5. Provides batch evaluation across multiple experiments
    6. Auto-detects experiment types from names and file paths
    """
    
    def __init__(self):
        """Initialize evaluation runner with framework and embedding model"""
        self.evaluation_framework = EvaluationFramework()
        self.model_manager = ModelManager()
        
        # Load embedding model for semantic similarity calculations
        self.model_manager.load_embedding_model()
        
        logger.info("EvaluationRunner initialized")
    
    # =============================================================================
    # EXPERIMENT TYPE DETECTION
    # =============================================================================
    
    def extract_experiment_type_from_name(self, experiment_name: str) -> Optional[str]:
        """
        Extract experiment type from experiment name.
        
        Experiment names follow the pattern: {type}_{dataset}_{model}_{prompt}_{size}_{temp}
        For example: baseline_gmeg_gpt-4o-mini_gmeg_v1_basic_50_0p1
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            str: Experiment type if valid, None otherwise
        """
        if not experiment_name:
            return None
        
        # Extract the first part of the experiment name
        name_parts = experiment_name.split('_')
        if not name_parts:
            return None
        
        potential_type = name_parts[0]
        
        # Validate that it's a known experiment type
        if potential_type in Config.EXPERIMENT_TYPES:
            logger.debug(f"Extracted experiment type '{potential_type}' from name '{experiment_name}'")
            return potential_type
        
        return None
    
    def detect_experiment_type(self, experiment_name: str, experiment_file: str = None) -> str:
        """
        Detect experiment type using multiple methods with fallback chain.
        
        Detection priority:
        1. Extract from experiment name (most reliable)
        2. Extract from file path (fallback)
        3. Default to 'baseline' (ultimate fallback)
        
        Args:
            experiment_name: Name of the experiment
            experiment_file: Path to experiment file (optional)
            
        Returns:
            str: Detected experiment type
        """
        # Method 1: Extract from experiment name
        detected_type = self.extract_experiment_type_from_name(experiment_name)
        if detected_type:
            return detected_type
        
        # Method 2: Extract from file path
        if experiment_file:
            for exp_type in Config.EXPERIMENT_TYPES:
                if exp_type in experiment_file:
                    logger.debug(f"Detected experiment type '{exp_type}' from file path '{experiment_file}'")
                    return exp_type
        
        # Method 3: Ultimate fallback
        logger.warning(f"Could not detect experiment type for '{experiment_name}', using 'baseline' as fallback")
        return 'baseline'
    
    # =============================================================================
    # EXPERIMENT FILE DISCOVERY AND LOADING
    # =============================================================================
    
    def find_experiment_files(self, experiment_type: Optional[str] = None) -> List[str]:
        """
        Find all experiment result files matching the specified type.
        
        Searches for inference result files that contain model responses
        and can be evaluated against expected outputs.
        
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
        
        The length matching is critical because:
        1. Each generated response must have a corresponding expected output for evaluation
        2. Mismatched lengths indicate data corruption or incomplete experiments
        3. We need paired data for meaningful metric computation (precision, recall, etc.)
        4. Truncating to minimum length ensures we only evaluate complete pairs
        
        Args:
            experiment_data: Loaded experiment result data
            
        Returns:
            tuple: (generated_responses, expected_outputs) as lists of strings with matching lengths
        """
        responses_data = experiment_data.get('responses', [])
        expected_outputs = experiment_data.get('expected_outputs', [])
        
        # Extract response text from response objects (which may include metadata)
        generated_responses = []
        for response_obj in responses_data:
            if isinstance(response_obj, dict):
                response_text = response_obj.get('response', '')
                generated_responses.append(response_text)
            else:
                generated_responses.append(str(response_obj))
        
        # Critical: Ensure both lists have the same length for proper evaluation
        # If lengths don't match, we can only evaluate the overlapping portion
        min_length = min(len(generated_responses), len(expected_outputs))
        
        if min_length != len(generated_responses) or min_length != len(expected_outputs):
            logger.warning(f"Length mismatch detected - Generated: {len(generated_responses)}, Expected: {len(expected_outputs)}")
            logger.warning(f"This may indicate incomplete experiment or data corruption")
            logger.warning(f"Truncating to {min_length} pairs for evaluation")
            
            # Truncate both lists to the same length to ensure paired evaluation
            generated_responses = generated_responses[:min_length]
            expected_outputs = expected_outputs[:min_length]
        
        logger.debug(f"Extracted {len(generated_responses)} response pairs for evaluation")
        return generated_responses, expected_outputs
    
    def determine_dataset_type(self, experiment_data: Dict[str, Any]) -> str:
        """
        Determine the dataset type for appropriate evaluation metrics.
        
        Different datasets may require specialized evaluation metrics
        beyond the standard token-based and semantic similarity metrics.
        
        Args:
            experiment_data: Loaded experiment data
            
        Returns:
            str: Dataset type for evaluation (e.g., 'gmeg', 'general')
        """
        dataset_name = experiment_data.get('experiment_config', {}).get('dataset', 'general')
        
        # Map dataset names to evaluation types
        dataset_type_mapping = {
            'gmeg': 'gmeg',                    # Grammatical error correction explanations
            'xai_fungi': 'general',            # General XAI explanations
            'hatebrxplain': 'general',         # Hate speech explanations
            'explanationhardness': 'general',  # General explanation difficulty
            'reframinghumanai': 'general'      # Human-AI interaction explanations
        }
        
        mapped_type = dataset_type_mapping.get(dataset_name, 'general')
        logger.debug(f"Mapped dataset '{dataset_name}' to evaluation type '{mapped_type}'")
        return mapped_type
    
    # =============================================================================
    # CORE EVALUATION METHODS
    # =============================================================================
    
    def evaluate_experiment_from_data(self, experiment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate experiment results from loaded data.
        
        This is the core evaluation method that:
        1. Extracts responses and expected outputs
        2. Determines appropriate evaluation metrics
        3. Runs comprehensive evaluation
        4. Adds metadata and returns results
        
        Args:
            experiment_data: Complete experiment data dictionary
            
        Returns:
            dict: Evaluation results with metrics and metadata, or None if failed
        """
        try:
            # Extract responses and expected outputs
            generated_responses, expected_outputs = self.extract_responses_and_expected(experiment_data)
            
            if not generated_responses or not expected_outputs:
                logger.error("No valid responses found for evaluation")
                return None
            
            # Determine appropriate dataset type for specialized metrics
            dataset_type = self.determine_dataset_type(experiment_data)
            
            # Run comprehensive evaluation
            experiment_name = experiment_data.get('experiment_name', 'unknown')
            logger.info(f"Evaluating experiment: {experiment_name}")
            
            evaluation_result = self.evaluation_framework.evaluate_batch(
                generated_responses=generated_responses,
                expected_responses=expected_outputs,
                embedding_model=self.model_manager.embedding_model,
                batch_name=experiment_name,
                dataset_type=dataset_type
            )
            
            # Add comprehensive metadata
            evaluation_result.update({
                'original_experiment_config': experiment_data.get('experiment_config', {}),
                'original_experiment_name': experiment_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'dataset_type': dataset_type
            })
            
            logger.info(f"Evaluation completed for: {experiment_name}")
            
            # Log key performance metrics for quick assessment
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
        """
        Save evaluation results to JSON file with organized structure.
        
        Args:
            evaluation_data: Complete evaluation results
            experiment_name: Name of the original experiment
            experiment_type: Type of experiment (for directory organization)
            
        Returns:
            str: Path to saved evaluation file
            
        Raises:
            Exception: If file cannot be saved
        """
        
        # Generate organized file path
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
    # SINGLE EXPERIMENT EVALUATION WITH ENHANCED TYPE DETECTION
    # =============================================================================
    
    def evaluate_experiment(self, experiment_name: str, experiment_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Evaluate a specific experiment by name with enhanced type detection.
        
        This method now:
        1. Auto-detects experiment type from name if not provided
        2. Searches appropriate directories based on detected/provided type
        3. Falls back to searching all directories if needed
        
        Args:
            experiment_name: Name of experiment to evaluate
            experiment_type: Optional type hint (will auto-detect if None)
            
        Returns:
            dict: Evaluation summary with file paths and key metrics, or None if failed
        """
        logger.info(f"Evaluating experiment: {experiment_name}")
        
        # Step 1: Determine experiment type using enhanced detection
        if not experiment_type:
            experiment_type = self.extract_experiment_type_from_name(experiment_name)
            if experiment_type:
                logger.info(f"Auto-detected experiment type: {experiment_type}")
        
        # Step 2: Find the experiment file using detected/provided type
        experiment_file = None
        
        if experiment_type:
            # Search in the specific type directory first
            search_dir = Config.get_output_dirs_for_experiment_type(experiment_type)['responses']
            potential_file = os.path.join(search_dir, f"inference_{experiment_name}.json")
            if os.path.exists(potential_file):
                experiment_file = potential_file
        
        # Step 3: Fallback to searching all directories if not found
        if not experiment_file:
            logger.debug("Searching all experiment directories as fallback")
            for exp_type in Config.EXPERIMENT_TYPES:
                search_dir = Config.get_output_dirs_for_experiment_type(exp_type)['responses']
                potential_file = os.path.join(search_dir, f"inference_{experiment_name}.json")
                if os.path.exists(potential_file):
                    experiment_file = potential_file
                    # Update experiment_type based on where we found it
                    if not experiment_type:
                        experiment_type = exp_type
                        logger.info(f"Found experiment in {exp_type} directory")
                    break
        
        if not experiment_file:
            logger.error(f"Experiment file not found: {experiment_name}")
            return None
        
        # Step 4: Final type detection using all available information
        final_experiment_type = self.detect_experiment_type(experiment_name, experiment_file)
        
        # Step 5: Load and evaluate experiment data
        experiment_data = self.load_experiment_results(experiment_file)
        if not experiment_data:
            return None
        
        evaluation_result = self.evaluate_experiment_from_data(experiment_data)
        if not evaluation_result:
            return None
        
        # Step 6: Save evaluation results using detected type
        output_file = self.save_evaluation_results(evaluation_result, experiment_name, final_experiment_type)
        
        return {
            'experiment_name': experiment_name,
            'experiment_type': final_experiment_type,
            'evaluation_file': output_file,
            'metrics': evaluation_result.get('aggregated_scores', {}),
            'num_samples': evaluation_result.get('num_samples', 0),
            'num_valid_evaluations': evaluation_result.get('num_valid_evaluations', 0)
        }
    
    # =============================================================================
    # BATCH EVALUATION WITH ENHANCED TYPE DETECTION
    # =============================================================================
    
    def evaluate_all_experiments(self, experiment_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Evaluate all experiments of the specified type with enhanced detection.
        
        Processes multiple experiments in batch with progress tracking
        and comprehensive error handling. Now uses enhanced type detection
        for each individual experiment.
        
        Args:
            experiment_type: Type of experiments to evaluate (None for all types)
            
        Returns:
            list: List of evaluation summaries for each processed experiment
        """
        logger.info(f"Evaluating all experiments (type: {experiment_type or 'all'})")
        
        # Find all matching experiment files
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
                
                # Load and evaluate experiment using enhanced detection
                experiment_data = self.load_experiment_results(file_path)
                if not experiment_data:
                    continue
                
                evaluation_result = self.evaluate_experiment_from_data(experiment_data)
                if not evaluation_result:
                    continue
                
                # Use enhanced type detection
                detected_experiment_type = self.detect_experiment_type(experiment_name, file_path)
                
                # Save evaluation results
                output_file = self.save_evaluation_results(evaluation_result, experiment_name, detected_experiment_type)
                
                results.append({
                    'experiment_name': experiment_name,
                    'experiment_type': detected_experiment_type,
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
    
    # =============================================================================
    # EVALUATION SUMMARY AND ANALYTICS
    # =============================================================================
    
    def get_evaluation_summary(self, experiment_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive summary of evaluation results across experiments.
        
        Args:
            experiment_type: Type filter for evaluation files
            
        Returns:
            dict: Summary statistics and performance overview
        """
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
        all_exact_matches = []
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
                if 'exact_match' in agg_scores:
                    all_exact_matches.append(agg_scores['exact_match']['mean'])
                
                experiment_count += 1
                total_samples += eval_data.get('num_valid_evaluations', 0)
                
            except Exception as e:
                logger.warning(f"Error loading evaluation file {file_path}: {e}")
                continue
        
        # Compile comprehensive summary
        summary = {
            'total_evaluations': experiment_count,
            'total_samples_evaluated': total_samples,
            'experiment_type_filter': experiment_type
        }
        
        # Add F1 score statistics
        if all_f1_scores:
            summary['f1_score_stats'] = {
                'mean': np.mean(all_f1_scores),
                'std': np.std(all_f1_scores),
                'min': np.min(all_f1_scores),
                'max': np.max(all_f1_scores),
                'median': np.median(all_f1_scores)
            }
        
        # Add semantic similarity statistics
        if all_semantic_similarities:
            summary['semantic_similarity_stats'] = {
                'mean': np.mean(all_semantic_similarities),
                'std': np.std(all_semantic_similarities),
                'min': np.min(all_semantic_similarities),
                'max': np.max(all_semantic_similarities),
                'median': np.median(all_semantic_similarities)
            }
        
        # Add exact match statistics
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
        Compare multiple experiments side by side with enhanced type detection.
        
        Args:
            experiment_names: List of experiment names to compare
            experiment_type: Optional type filter for experiments
            
        Returns:
            dict: Comparison results with metric differences
        """
        logger.info(f"Comparing {len(experiment_names)} experiments")
        
        # Load evaluation results for all experiments
        evaluations = []
        for exp_name in experiment_names:
            # Use enhanced type detection if no type provided
            search_experiment_type = experiment_type
            if not search_experiment_type:
                search_experiment_type = self.extract_experiment_type_from_name(exp_name)
            
            # Find evaluation file
            eval_file = None
            if search_experiment_type:
                search_dirs = [Config.get_output_dirs_for_experiment_type(search_experiment_type)['evaluations']]
            else:
                search_dirs = []
                for exp_type in Config.EXPERIMENT_TYPES:
                    search_dirs.append(Config.get_output_dirs_for_experiment_type(exp_type)['evaluations'])
            
            for search_dir in search_dirs:
                potential_file = os.path.join(search_dir, f"evaluation_{exp_name}.json")
                if os.path.exists(potential_file):
                    eval_file = potential_file
                    break
            
            if eval_file:
                try:
                    with open(eval_file, 'r') as f:
                        eval_data = json.load(f)
                    evaluations.append(eval_data)
                except Exception as e:
                    logger.error(f"Error loading evaluation for {exp_name}: {e}")
            else:
                logger.warning(f"Evaluation file not found for: {exp_name}")
        
        if len(evaluations) < 2:
            return {"error": "Need at least 2 valid evaluations for comparison"}
        
        # Create comparison matrix
        comparison = {
            'experiment_names': experiment_names[:len(evaluations)],
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
            comparison['rankings'][metric] = [experiment_names[i] for i in ranked_indices]
            
            # Best performer for this metric
            best_idx = np.argmax(values)
            comparison['best_performers'][metric] = {
                'experiment': experiment_names[best_idx],
                'value': values[best_idx]
            }
        
        return comparison