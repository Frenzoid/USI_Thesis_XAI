import numpy as np
import torch
from torch import cosine_similarity
from typing import Dict, List, Any
from datetime import datetime
from tqdm import tqdm

from utils import setup_logging

logger = setup_logging("evaluation")

class EvaluationFramework:
    """
    Comprehensive evaluation system for XAI explanation quality assessment.
    
    This framework provides:
    1. Token-based metrics (F1, precision, recall, exact match)
    2. Semantic similarity using embeddings
    3. Dataset-specific quality metrics
    4. Custom metric registration system
    5. Batch processing with aggregation
    6. Evaluation history and comparison tools
    """
    
    def __init__(self):
        """Initialize evaluation framework with empty state"""
        self.metrics_cache = {}         # Cache for expensive computations
        self.evaluation_history = []    # History of all evaluations performed
        self.custom_metrics = {}        # Registry for custom evaluation metrics
        
        logger.info("EvaluationFramework initialized")
    

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
    # DATASET-SPECIFIC METRICS
    # =============================================================================
    
    def compute_explanation_specific_metrics(self, generated: str, expected: str, 
                                           dataset_type: str = 'gmeg') -> Dict[str, float]:
        """
        Compute dataset-specific metrics for explanation quality assessment.
        
        Different datasets have different quality indicators and expected formats.
        This method implements specialized metrics for each dataset type.
        
        Args:
            generated: Generated explanation text
            expected: Expected explanation text
            dataset_type: Type of dataset (determines which metrics to compute)
            
        Returns:
            dict: Dataset-specific metrics
        """
        metrics = {}
        
        if dataset_type == 'gmeg':
            # GMEG-specific metrics for grammatical error correction explanations
            
            # 1. Bullet point format consistency
            # Many GMEG explanations use bullet points to list corrections
            gen_bullets = len([line for line in generated.split('\n') 
                              if line.strip().startswith(('-', '•', '*'))])
            exp_bullets = len([line for line in expected.split('\n') 
                              if line.strip().startswith(('-', '•', '*'))])
            
            # Ratio of bullet points (capped at 2.0 to avoid extreme values)
            bullet_ratio = gen_bullets / max(exp_bullets, 1)
            metrics['bullet_point_ratio'] = min(bullet_ratio, 2.0)
            
            # 2. Correction terminology usage
            # Check for words that indicate specific types of corrections
            correction_terms = [
                'spelling', 'grammar', 'punctuation', 'capitalization', 'word choice',
                'corrected', 'changed', 'replaced', 'added', 'removed', 'fixed',
                'error', 'mistake', 'wrong', 'incorrect'
            ]
            
            gen_lower = generated.lower()
            exp_lower = expected.lower()
            
            gen_correction_terms = sum(1 for term in correction_terms if term in gen_lower)
            exp_correction_terms = sum(1 for term in correction_terms if term in exp_lower)
            
            # Correction terminology recall (how well model uses correction vocabulary)
            if exp_correction_terms > 0:
                metrics['correction_terminology_recall'] = gen_correction_terms / exp_correction_terms
            else:
                metrics['correction_terminology_recall'] = 1.0 if gen_correction_terms == 0 else 0.0
            
            # 3. Structural format matching
            # Both should follow similar formatting (bullet points vs paragraphs)
            gen_has_structure = gen_bullets > 0
            exp_has_structure = exp_bullets > 0
            metrics['structural_format_match'] = float(gen_has_structure == exp_has_structure)
            
            logger.debug(f"GMEG-specific metrics computed: {metrics}")
        
        # Add support for other datasets here as they are implemented
        # elif dataset_type == 'other_dataset':
        #     metrics.update(compute_other_dataset_metrics(generated, expected))
        
        return metrics
    
    # =============================================================================
    # SINGLE RESPONSE EVALUATION
    # =============================================================================
    
    def evaluate_single_response(self, generated, expected, embedding_model, dataset_type) -> Dict[str, float]:
        """
        Evaluate a single generated response against expected output.
        
        This is the core evaluation method that computes all metrics for
        one generated response. It handles edge cases and combines multiple
        metric types into a comprehensive evaluation.
        
        Args:
            generated: Generated response from model (can be None/empty)
            expected: Expected response from dataset (can be None/empty)
            embedding_model: Loaded embedding model for similarity
            dataset_type: Type of dataset for specific metrics
            
        Returns:
            dict: All computed metrics for this response
        """
        
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
        
        # Compute core token-based metrics
        metrics = self.compute_token_metrics(generated, expected)
        
        # Add semantic similarity using embeddings
        metrics['semantic_similarity'] = self.compute_text_similarity(generated, expected, embedding_model)
        
        # Add dataset-specific metrics if not general dataset
        if dataset_type != 'general':
            specific_metrics = self.compute_explanation_specific_metrics(generated, expected, dataset_type)
            metrics.update(specific_metrics)
        
        # Apply any registered custom metrics
        for metric_name, metric_func in self.custom_metrics.items():
            try:
                metrics[metric_name] = metric_func(generated, expected)
            except Exception as e:
                logger.error(f"Error computing custom metric {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        # Mark as valid evaluation (not skipped)
        metrics['skipped_na'] = 0.0
        
        logger.debug(f"Evaluated single response with {len(metrics)} metrics")
        return metrics
    
    # =============================================================================
    # BATCH EVALUATION AND AGGREGATION
    # =============================================================================
    
    def evaluate_batch(self, generated_responses: List[str], expected_responses: List[str],
                             embedding_model, batch_name: str = "batch", dataset_type: str = 'general') -> Dict[str, Any]:
        """
        Evaluate a batch of responses and compute aggregate statistics.
        
        Processes multiple response pairs efficiently and computes both
        individual scores and aggregate statistics across the batch.
        
        Args:
            generated_responses: List of generated responses from model
            expected_responses: List of expected responses from dataset
            embedding_model: Loaded embedding model for similarity computation
            batch_name: Name for this batch (for logging and identification)
            dataset_type: Type of dataset for specialized metrics
            
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
        
        # Evaluate each response pair with progress tracking
        for gen, exp in tqdm(zip(generated_responses, expected_responses), 
                           total=len(generated_responses), 
                           desc=f"Evaluating {batch_name}"):
            scores = self.evaluate_single_response(gen, exp, embedding_model, dataset_type=dataset_type)
            individual_scores.append(scores)
            
            # Count items that were skipped due to NA annotations
            if scores.get('skipped_na', 0) == 1.0:
                skipped_count += 1
        
        # Filter out skipped items for meaningful aggregation
        valid_scores = [score for score in individual_scores if score.get('skipped_na', 0) == 0.0]
        
        # Compute aggregate statistics from valid scores
        if valid_scores:
            aggregated = self.aggregate_scores(valid_scores)
            logger.info(f"Aggregated scores from {len(valid_scores)} valid evaluations")
        else:
            aggregated = {}
            logger.warning("No valid scores to aggregate (all samples may have been marked as NA)")
        
        # Compile complete results
        result = {
            'batch_name': batch_name,
            'dataset_type': dataset_type,
            'num_samples': len(generated_responses),
            'num_valid_evaluations': len(valid_scores),
            'num_skipped_na': skipped_count,
            'individual_scores': individual_scores,
            'aggregated_scores': aggregated,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log summary information
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} items marked as 'NA' or 'not annotatable'")
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
            # Skip the skipped_na flag in aggregation
            if metric == 'skipped_na':
                continue
            
            values = [score[metric] for score in individual_scores]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        logger.debug(f"Aggregated {len(metrics)-1} metrics from {len(individual_scores)} scores")
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
            'dataset_types': list(set(e['dataset_type'] for e in self.evaluation_history)),
            'batch_names': [e['batch_name'] for e in self.evaluation_history],
            'custom_metrics_registered': list(self.custom_metrics.keys())
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