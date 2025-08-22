import numpy as np
import torch
from torch import cosine_similarity
from typing import Dict, List, Any
from datetime import datetime
from tqdm import tqdm

from utils import setup_logging

logger = setup_logging("evaluation")

class EvaluationFramework:
    """Comprehensive evaluation system for XAI explanation quality"""
    
    def __init__(self):
        self.metrics_cache = {}
        self.evaluation_history = []
        self.custom_metrics = {}  # Store dataset-specific metrics
        
        logger.info("EvaluationFramework initialized")
    
    def register_custom_metric(self, name: str, metric_function: callable):
        """Register a custom evaluation metric"""
        self.custom_metrics[name] = metric_function
        logger.info(f"Registered custom metric: {name}")
    
    def compute_text_similarity(self, text1: str, text2: str, embedding_model) -> float:
        """Compute semantic similarity between two texts"""
        try:
            embeddings1 = torch.tensor(embedding_model.embed_query(text1)).reshape(1, -1)
            embeddings2 = torch.tensor(embedding_model.embed_query(text2)).reshape(1, -1)
            similarity = cosine_similarity(embeddings1, embeddings2).item()
            logger.debug(f"Computed similarity: {similarity:.4f}")
            return similarity
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def compute_token_metrics(self, generated: str, expected: str) -> Dict[str, float]:
        """Compute token-based metrics"""
        # Clean and normalize text
        gen_clean = generated.strip().lower()
        exp_clean = expected.strip().lower()
        
        # Handle empty strings
        if not gen_clean and not exp_clean:
            return {'exact_match': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'jaccard': 1.0}
        elif not gen_clean or not exp_clean:
            return {'exact_match': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'jaccard': 0.0}
        
        gen_tokens = set(gen_clean.split())
        exp_tokens = set(exp_clean.split())
        
        # Exact match
        exact_match = float(gen_clean == exp_clean)
        
        # Token overlap metrics
        common_tokens = gen_tokens & exp_tokens
        precision = len(common_tokens) / len(gen_tokens) if gen_tokens else 0.0
        recall = len(common_tokens) / len(exp_tokens) if exp_tokens else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Jaccard similarity
        all_tokens = gen_tokens | exp_tokens
        jaccard = len(common_tokens) / len(all_tokens) if all_tokens else 0.0
        
        return {
            'exact_match': exact_match,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'jaccard': jaccard
        }
    
    def compute_explanation_specific_metrics(self, generated: str, expected: str, 
                                           dataset_type: str = 'gmeg') -> Dict[str, float]:
        """Compute dataset-specific metrics for explanation quality"""
        metrics = {}
        
        if dataset_type == 'gmeg':
            # GMEG-specific metrics for correction explanation quality
            
            # Count bullet points (expected format)
            gen_bullets = len([line for line in generated.split('\n') if line.strip().startswith(('-', '•', '*'))])
            exp_bullets = len([line for line in expected.split('\n') if line.strip().startswith(('-', '•', '*'))])
            
            # Bullet point ratio
            bullet_ratio = gen_bullets / max(exp_bullets, 1)
            metrics['bullet_point_ratio'] = min(bullet_ratio, 2.0)  # Cap at 2.0 to avoid extreme values
            
            # Check for key correction terms (words that indicate error types)
            correction_terms = [
                'spelling', 'grammar', 'punctuation', 'capitalization', 'word choice',
                'corrected', 'changed', 'replaced', 'added', 'removed', 'fixed'
            ]
            
            gen_lower = generated.lower()
            exp_lower = expected.lower()
            
            gen_correction_terms = sum(1 for term in correction_terms if term in gen_lower)
            exp_correction_terms = sum(1 for term in correction_terms if term in exp_lower)
            
            # Correction terminology overlap
            if exp_correction_terms > 0:
                metrics['correction_terminology_recall'] = gen_correction_terms / exp_correction_terms
            else:
                metrics['correction_terminology_recall'] = 1.0 if gen_correction_terms == 0 else 0.0
            
            # Structure similarity (both should be bullet-pointed lists)
            gen_has_structure = gen_bullets > 0
            exp_has_structure = exp_bullets > 0
            metrics['structural_format_match'] = float(gen_has_structure == exp_has_structure)
            
            logger.debug(f"GMEG-specific metrics computed: {metrics}")
        
        return metrics
    
    def evaluate_single_response(self, generated, expected, embedding_model, dataset_type) -> Dict[str, float]:
        """Evaluate a single generated response against expected output"""
        
        # Handle None or empty inputs
        if generated is None:
            generated = ""
        if expected is None:
            expected = ""
        
        generated = str(generated)
        expected = str(expected)
        
        # Skip evaluation if expected output indicates no annotation
        if expected.lower().strip() in ['na', 'n/a', 'not applicable', 'not annotatable']:
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
        
        # Compute basic metrics
        metrics = self.compute_token_metrics(generated, expected)
        
        # Add semantic similarity
        metrics['semantic_similarity'] = self.compute_text_similarity(generated, expected, embedding_model)
        
        # Add dataset-specific metrics
        if dataset_type != 'general':
            specific_metrics = self.compute_explanation_specific_metrics(generated, expected, dataset_type)
            metrics.update(specific_metrics)
        
        # Add custom metrics if registered
        for metric_name, metric_func in self.custom_metrics.items():
            try:
                metrics[metric_name] = metric_func(generated, expected)
            except Exception as e:
                logger.error(f"Error computing custom metric {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        # Flag for valid evaluation
        metrics['skipped_na'] = 0.0
        
        logger.debug(f"Evaluated single response with {len(metrics)} metrics")
        return metrics
    
    def evaluate_batch(self, generated_responses: List[str], expected_responses: List[str],
                             embedding_model, batch_name: str = "batch", dataset_type: str = 'general') -> Dict[str, Any]:
        """Evaluate a batch of responses"""
        
        if len(generated_responses) != len(expected_responses):
            raise ValueError("Generated and expected responses must have same length")
        
        logger.info(f"Evaluating batch '{batch_name}' with {len(generated_responses)} responses")
        
        individual_scores = []
        skipped_count = 0
        
        for gen, exp in tqdm(zip(generated_responses, expected_responses), 
                           total=len(generated_responses), 
                           desc=f"Evaluating {batch_name}"):
            scores = self.evaluate_single_response(gen, exp, embedding_model, dataset_type=dataset_type)
            individual_scores.append(scores)
            
            # Count skipped items (NA annotations)
            if scores.get('skipped_na', 0) == 1.0:
                skipped_count += 1
        
        # Filter out skipped items for aggregation
        valid_scores = [score for score in individual_scores if score.get('skipped_na', 0) == 0.0]
        
        # Aggregate scores (only from valid evaluations)
        if valid_scores:
            aggregated = self.aggregate_scores(valid_scores)
            logger.info(f"Aggregated scores from {len(valid_scores)} valid evaluations")
        else:
            aggregated = {}
            logger.warning("No valid scores to aggregate (all samples may have been marked as NA)")
        
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
        
        # Print summary with skipped items info
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} items marked as 'NA' or 'not annotatable'")
            logger.info(f"Valid evaluations: {len(valid_scores)} out of {len(generated_responses)}")
        
        # Log key metrics
        if aggregated:
            f1_mean = aggregated.get('f1_score', {}).get('mean', 0)
            sem_sim_mean = aggregated.get('semantic_similarity', {}).get('mean', 0)
            logger.info(f"Batch results - F1: {f1_mean:.4f}, Semantic Similarity: {sem_sim_mean:.4f}")
        
        self.evaluation_history.append(result)
        return result
    
    def aggregate_scores(self, individual_scores: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate individual scores into summary statistics"""
        if not individual_scores:
            return {}
        
        metrics = list(individual_scores[0].keys())
        aggregated = {}
        
        for metric in metrics:
            if metric == 'skipped_na':  # Skip aggregating this flag
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
    
    def compare_evaluations(self, eval1: Dict, eval2: Dict) -> Dict[str, Any]:
        """Compare two evaluation results"""
        logger.info(f"Comparing evaluations: {eval1.get('batch_name')} vs {eval2.get('batch_name')}")
        
        comparison = {
            'eval1_name': eval1.get('batch_name', 'eval1'),
            'eval2_name': eval2.get('batch_name', 'eval2'),
            'sample_sizes': {
                'eval1': eval1['num_samples'],
                'eval2': eval2['num_samples']
            },
            'metric_differences': {}
        }
        
        agg1 = eval1['aggregated_scores']
        agg2 = eval2['aggregated_scores']
        
        common_metrics = set(agg1.keys()) & set(agg2.keys())
        
        for metric in common_metrics:
            diff = agg2[metric]['mean'] - agg1[metric]['mean']
            pct_change = (diff / agg1[metric]['mean'] * 100) if agg1[metric]['mean'] != 0 else 0
            
            comparison['metric_differences'][metric] = {
                'absolute_difference': diff,
                'percentage_change': pct_change,
                'eval1_mean': agg1[metric]['mean'],
                'eval2_mean': agg2[metric]['mean']
            }
        
        logger.info(f"Comparison completed for {len(common_metrics)} common metrics")
        return comparison
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed"""
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
        
        # Find best performing evaluation
        if self.evaluation_history:
            best_f1 = max(self.evaluation_history, 
                         key=lambda x: x.get('aggregated_scores', {}).get('f1_score', {}).get('mean', 0))
            summary['best_f1_evaluation'] = {
                'name': best_f1['batch_name'],
                'f1_score': best_f1.get('aggregated_scores', {}).get('f1_score', {}).get('mean', 0)
            }
        
        return summary
