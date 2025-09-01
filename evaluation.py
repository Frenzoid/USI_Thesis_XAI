import numpy as np
import torch
from torch import cosine_similarity
from typing import Dict, List, Any, Callable
from datetime import datetime
from tqdm import tqdm
import importlib
import os

from utils import setup_logging
from config import Config

logger = setup_logging("evaluation")

class EvaluationFramework:
    """
    Comprehensive evaluation system for XAI explanation quality assessment.
    
    This framework provides:
    1. Token-based metrics (F1, precision, recall, exact match)
    2. Semantic similarity using embeddings
    3. Dynamic custom metrics loaded from dataset-specific modules
    4. Custom metric registration system
    5. Batch processing with aggregation
    6. Evaluation history and comparison tools
    """
    
    def __init__(self):
        """Initialize evaluation framework with empty state"""
        self.metrics_cache = {}         # Cache for expensive computations
        self.evaluation_history = []    # History of all evaluations performed
        self.custom_metrics = {}        # Registry for custom evaluation metrics
        self.dataset_custom_metrics = {}  # Cache for loaded dataset-specific metrics
        
        logger.info("EvaluationFramework initialized")
    
    # =============================================================================
    # CUSTOM METRICS LOADING AND MANAGEMENT
    # =============================================================================
    
    def load_custom_metrics_for_dataset(self, dataset_name: str) -> Dict[str, Callable]:
        """
        Load custom metrics for a specific dataset from configured module.
        
        Args:
            dataset_name: Name of the dataset to load metrics for
            
        Returns:
            dict: Dictionary of metric_name -> metric_function
        """
        # Check cache first
        if dataset_name in self.dataset_custom_metrics:
            return self.dataset_custom_metrics[dataset_name]
        
        # Load datasets configuration
        try:
            datasets_config = Config.load_datasets_config()
        except Exception as e:
            logger.error(f"Failed to load datasets config: {e}")
            return {}
        
        if dataset_name not in datasets_config:
            logger.warning(f"Dataset {dataset_name} not found in configuration")
            return {}
        
        dataset_config = datasets_config[dataset_name]
        custom_metrics_config = dataset_config.get('custom_metrics', {})
        
        if not custom_metrics_config:
            logger.info(f"No custom metrics configured for dataset: {dataset_name}")
            self.dataset_custom_metrics[dataset_name] = {}
            return {}
        
        # Load the custom metrics module
        module_path = custom_metrics_config.get('module_path')
        metrics_registry = custom_metrics_config.get('metrics_registry')
        
        if not module_path or not metrics_registry:
            logger.warning(f"Incomplete custom metrics configuration for {dataset_name}")
            self.dataset_custom_metrics[dataset_name] = {}
            return {}
        
        try:
            logger.info(f"Loading custom metrics for {dataset_name} from {module_path}")
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the metrics registry
            if hasattr(module, metrics_registry):
                metrics_dict = getattr(module, metrics_registry)
                
                if isinstance(metrics_dict, dict):
                    # Validate that all values are callable
                    valid_metrics = {}
                    for metric_name, metric_func in metrics_dict.items():
                        if callable(metric_func):
                            valid_metrics[metric_name] = metric_func
                        else:
                            logger.warning(f"Metric {metric_name} is not callable, skipping")
                    
                    logger.info(f"Loaded {len(valid_metrics)} custom metrics for {dataset_name}: {list(valid_metrics.keys())}")
                    self.dataset_custom_metrics[dataset_name] = valid_metrics
                    return valid_metrics
                else:
                    logger.error(f"Metrics registry {metrics_registry} is not a dictionary")
            else:
                logger.error(f"Metrics registry {metrics_registry} not found in module {module_path}")
                
        except ImportError as e:
            logger.error(f"Failed to import custom metrics module {module_path}: {e}")
            logger.info("Make sure the custom_metrics directory exists and contains __init__.py")
        except Exception as e:
            logger.error(f"Error loading custom metrics for {dataset_name}: {e}")
        
        # Cache empty result on failure
        self.dataset_custom_metrics[dataset_name] = {}
        return {}
    
    def register_custom_metric(self, name: str, metric_func: Callable):
        """
        Register a custom metric function for use in evaluations.
        
        Args:
            name: Name of the metric
            metric_func: Function that takes response_data dict and returns float
        """
        if not callable(metric_func):
            raise ValueError(f"Metric function for {name} must be callable")
        
        self.custom_metrics[name] = metric_func
        logger.info(f"Registered custom metric: {name}")
    
    # =============================================================================
    # SEMANTIC SIMILARITY COMPUTATION
    # =============================================================================
    
    def compute_text_similarity(self, text1: str, text2: str, embedding_model) -> float:
        """
        Compute semantic similarity between two texts using embeddings.
        
        Uses cosine similarity between sentence embeddings to measure
        semantic closeness beyond just token overlap.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            embedding_model: Loaded embedding model from HuggingFace
            
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        try:
            # Get embeddings for both texts
            embeddings1 = torch.tensor(embedding_model.embed_query(text1)).reshape(1, -1)
            embeddings2 = torch.tensor(embedding_model.embed_query(text2)).reshape(1, -1)
            
            # Compute cosine similarity
            similarity = cosine_similarity(embeddings1, embeddings2).item()
            logger.debug(f"Computed similarity: {similarity:.4f}")
            return similarity
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    # =============================================================================
    # TOKEN-BASED METRICS
    # =============================================================================
    
    def compute_token_metrics(self, generated: str, expected: str) -> Dict[str, float]:
        """
        Compute token-level overlap metrics between generated and expected text.
        
        Computes standard NLP evaluation metrics based on token overlap:
        - Exact match: Perfect string equality
        - Precision: Fraction of generated tokens that appear in expected
        - Recall: Fraction of expected tokens that appear in generated
        - F1: Harmonic mean of precision and recall
        - Jaccard: Intersection over union of token sets
        
        Args:
            generated: Generated text from model
            expected: Expected/reference text
            
        Returns:
            dict: Dictionary of metric names to scores
        """
        # Normalize and tokenize texts
        gen_clean = generated.strip().lower()
        exp_clean = expected.strip().lower()
        
        # Handle edge cases where one or both texts are empty
        if not gen_clean and not exp_clean:
            # Both empty - perfect match
            return {'exact_match': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'jaccard': 1.0}
        elif not gen_clean or not exp_clean:
            # One empty, one not - no match
            return {'exact_match': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'jaccard': 0.0}
        
        # Tokenize into sets for overlap computation
        gen_tokens = set(gen_clean.split())
        exp_tokens = set(exp_clean.split())
        
        # Exact match check
        exact_match = float(gen_clean == exp_clean)
        
        # Token overlap metrics
        common_tokens = gen_tokens & exp_tokens
        precision = len(common_tokens) / len(gen_tokens) if gen_tokens else 0.0
        recall = len(common_tokens) / len(exp_tokens) if exp_tokens else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Jaccard similarity (intersection over union)
        all_tokens = gen_tokens | exp_tokens
        jaccard = len(common_tokens) / len(all_tokens) if all_tokens else 0.0
        
        return {
            'exact_match': exact_match,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'jaccard': jaccard
        }
    
    # =============================================================================
    # SINGLE RESPONSE EVALUATION
    # =============================================================================
    
    def evaluate_single_response(self, response_data: Dict[str, Any], embedding_model, dataset_name: str = 'general') -> Dict[str, float]:
        """
        Evaluate a single generated response against expected output.
        
        This is the core evaluation method that computes all metrics for
        one generated response. It handles edge cases and combines multiple
        metric types into a comprehensive evaluation.
        
        Args:
            response_data: Dictionary containing response fields (prompt, response, expected_output, etc.)
            embedding_model: Loaded embedding model for similarity
            dataset_name: Name of dataset for loading custom metrics
            
        Returns:
            dict: All computed metrics for this response
        """
        # Extract fields from response data
        generated = response_data.get('response', '')
        expected = response_data.get('expected_output', '')
        success = response_data.get('success', False)
        
        # Handle None or empty inputs by converting to strings
        if generated is None:
            generated = ""
        if expected is None:
            expected = ""
        
        generated = str(generated)
        expected = str(expected)
        
        # Skip evaluation if expected output indicates no annotation available
        # Common in datasets where some samples couldn't be annotated
        na_indicators = ['na', 'n/a', 'not applicable', 'not annotatable', '']
        if expected.lower().strip() in na_indicators:
            logger.debug("Skipping evaluation for NA annotation")
            return {
                'exact_match': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'jaccard': 0.0,
                'semantic_similarity': 0.0,
                'skipped_na': 1.0  # Flag to identify skipped items
            }
        
        # Skip evaluation if response generation failed
        if not success:
            logger.debug("Skipping evaluation for failed response generation")
            return {
                'exact_match': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'jaccard': 0.0,
                'semantic_similarity': 0.0,
                'skipped_na': 0.0,
                'generation_failed': 1.0
            }
        
        # Compute core token-based metrics
        metrics = self.compute_token_metrics(generated, expected)
        
        # Add semantic similarity using embeddings
        metrics['semantic_similarity'] = self.compute_text_similarity(generated, expected, embedding_model)
        
        # Load and apply dataset-specific custom metrics
        if dataset_name != 'general':
            custom_metrics = self.load_custom_metrics_for_dataset(dataset_name)
            for metric_name, metric_func in custom_metrics.items():
                try:
                    metric_value = metric_func(response_data)
                    # Ensure metric value is a float and within reasonable bounds
                    if isinstance(metric_value, (int, float)):
                        metrics[metric_name] = float(metric_value)
                    else:
                        logger.warning(f"Custom metric {metric_name} returned non-numeric value: {metric_value}")
                        metrics[metric_name] = 0.0
                except Exception as e:
                    logger.error(f"Error computing custom metric {metric_name}: {e}")
                    metrics[metric_name] = 0.0
        
        # Apply any registered custom metrics (global)
        for metric_name, metric_func in self.custom_metrics.items():
            try:
                metric_value = metric_func(response_data)
                if isinstance(metric_value, (int, float)):
                    metrics[metric_name] = float(metric_value)
                else:
                    logger.warning(f"Registered metric {metric_name} returned non-numeric value: {metric_value}")
                    metrics[metric_name] = 0.0
            except Exception as e:
                logger.error(f"Error computing registered metric {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        # Mark as valid evaluation (not skipped)
        metrics['skipped_na'] = 0.0
        metrics['generation_failed'] = 0.0
        
        logger.debug(f"Evaluated single response with {len(metrics)} metrics")
        return metrics
    
    # =============================================================================
    # BATCH EVALUATION AND AGGREGATION
    # =============================================================================
    
    def evaluate_batch(self, generated_responses: List[str], expected_responses: List[str],
                             embedding_model, batch_name: str = "batch", dataset_name: str = 'general',
                             response_data_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a batch of responses and compute aggregate statistics.
        
        Processes multiple response pairs efficiently and computes both
        individual scores and aggregate statistics across the batch.
        
        Args:
            generated_responses: List of generated responses from model
            expected_responses: List of expected responses from dataset
            embedding_model: Loaded embedding model for similarity computation
            batch_name: Name for this batch (for logging and identification)
            dataset_name: Name of dataset for loading custom metrics
            response_data_list: List of full response data dictionaries (with all fields)
            
        Returns:
            dict: Complete evaluation results with individual and aggregate scores
            
        Raises:
            ValueError: If response lists have different lengths
        """
        
        if len(generated_responses) != len(expected_responses):
            raise ValueError("Generated and expected responses must have same length")
        
        logger.info(f"Evaluating batch '{batch_name}' with {len(generated_responses)} responses")
        
        individual_scores = []
        skipped_count = 0
        failed_count = 0
        
        # Evaluate each response pair with progress tracking
        for i, (gen, exp) in enumerate(tqdm(zip(generated_responses, expected_responses), 
                                          total=len(generated_responses), 
                                          desc=f"Evaluating {batch_name}")):
            
            # Use full response data if available, otherwise create minimal data
            if response_data_list and i < len(response_data_list):
                response_data = response_data_list[i]
            else:
                # Fallback: create minimal response data dictionary
                response_data = {
                    'response': gen,
                    'expected_output': exp,
                    'success': True,
                    'error': None,
                    'prompt': '',
                    'question_values': []
                }
            
            scores = self.evaluate_single_response(response_data, embedding_model, dataset_name=dataset_name)
            individual_scores.append(scores)
            
            # Count items that were skipped due to NA annotations or failed generation
            if scores.get('skipped_na', 0) == 1.0:
                skipped_count += 1
            elif scores.get('generation_failed', 0) == 1.0:
                failed_count += 1
        
        # Filter out skipped items for meaningful aggregation
        valid_scores = [score for score in individual_scores 
                       if score.get('skipped_na', 0) == 0.0 and score.get('generation_failed', 0) == 0.0]
        
        # Compute aggregate statistics from valid scores
        if valid_scores:
            aggregated = self.aggregate_scores(valid_scores)
            logger.info(f"Aggregated scores from {len(valid_scores)} valid evaluations")
        else:
            aggregated = {}
            logger.warning("No valid scores to aggregate (all samples may have been marked as NA or failed)")
        
        # Compile complete results
        result = {
            'batch_name': batch_name,
            'dataset_name': dataset_name,
            'num_samples': len(generated_responses),
            'num_valid_evaluations': len(valid_scores),
            'num_skipped_na': skipped_count,
            'num_failed_generation': failed_count,
            'individual_scores': individual_scores,
            'aggregated_scores': aggregated,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log summary information
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} items marked as 'NA' or 'not annotatable'")
        if failed_count > 0:
            logger.info(f"Skipped {failed_count} items with failed response generation")
        
        logger.info(f"Valid evaluations: {len(valid_scores)} out of {len(generated_responses)}")
        
        # Log key performance metrics
        if aggregated:
            f1_mean = aggregated.get('f1_score', {}).get('mean', 0)
            sem_sim_mean = aggregated.get('semantic_similarity', {}).get('mean', 0)
            logger.info(f"Batch results - F1: {f1_mean:.4f}, Semantic Similarity: {sem_sim_mean:.4f}")
        
        # Add to evaluation history for later analysis
        self.evaluation_history.append(result)
        return result
    
    def aggregate_scores(self, individual_scores: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate individual scores into summary statistics.
        
        Computes descriptive statistics (mean, std, min, max, median) for each metric
        across all valid evaluations in the batch.
        
        Args:
            individual_scores: List of metric dictionaries from individual evaluations
            
        Returns:
            dict: Nested dictionary of metric names to statistic dictionaries
        """
        if not individual_scores:
            return {}
        
        metrics = list(individual_scores[0].keys())
        aggregated = {}
        
        for metric in metrics:
            # Skip flags in aggregation
            if metric in ['skipped_na', 'generation_failed']:
                continue
            
            values = [score[metric] for score in individual_scores]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        logger.debug(f"Aggregated {len(metrics)-2} metrics from {len(individual_scores)} scores")
        return aggregated
    
    # =============================================================================
    # EVALUATION HISTORY AND SUMMARY
    # =============================================================================
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all evaluations performed in this session.
        
        Returns:
            dict: Summary statistics and information about evaluation history
        """
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'total_samples_evaluated': sum(e['num_samples'] for e in self.evaluation_history),
            'total_valid_evaluations': sum(e['num_valid_evaluations'] for e in self.evaluation_history),
            'dataset_names': list(set(e['dataset_name'] for e in self.evaluation_history)),
            'batch_names': [e['batch_name'] for e in self.evaluation_history],
            'custom_metrics_registered': list(self.custom_metrics.keys()),
            'dataset_custom_metrics_loaded': list(self.dataset_custom_metrics.keys())
        }
        
        # Find best performing evaluation based on F1 score
        if self.evaluation_history:
            best_f1 = max(self.evaluation_history, 
                         key=lambda x: x.get('aggregated_scores', {}).get('f1_score', {}).get('mean', 0))
            summary['best_f1_evaluation'] = {
                'name': best_f1['batch_name'],
                'f1_score': best_f1.get('aggregated_scores', {}).get('f1_score', {}).get('mean', 0)
            }
        
        return summary