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
    2. Extracts metadata from file content
    3. Extracts model responses and expected outputs
    4. Runs evaluation using the EvaluationFramework with custom metrics
    5. Includes pruning statistics from inference stage
    6. Saves evaluation results with metadata
    7. Provides batch evaluation across multiple experiments
    8. Generates aggregated summaries comparing prompts and models by setup
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
        Extract experiment metadata from loaded file data.
        
        Args:
            experiment_data: Loaded experiment result data
            
        Returns:
            dict: Extracted metadata including mode, model, etc.
        """
        # For inference result files, metadata is in 'experiment_config'
        metadata = experiment_data.get('experiment_config', {})
        
        if not metadata:
            # Fallback: check if it's an evaluation file with 'original_experiment_config'
            metadata = experiment_data.get('original_experiment_config', {})
        
        # Extract key fields with defaults
        extracted = {
            'mode': metadata.get('mode', 'zero-shot'),
            'model': metadata.get('model', 'unknown'),
            'setup': metadata.get('setup', 'unknown'),
            'prompt': metadata.get('prompt', 'unknown'),
            'size': metadata.get('size', 0),
            'temperature': metadata.get('temperature', 0.1),
            'few_shot_row': metadata.get('few_shot_row')
        }
        
        logger.debug(f"Extracted metadata from file: {extracted}")
        return extracted
    
    def extract_pruning_stats_from_file_data(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract pruning statistics from loaded inference file data.
        
        Args:
            experiment_data: Loaded inference result data
            
        Returns:
            dict: Pruning statistics from the inference stage
        """
        pruning_stats = experiment_data.get('pruning_stats', {})
        
        # Also include dataset_info for additional context
        dataset_info = experiment_data.get('dataset_info', {})
        
        # Combine both sources of information
        combined_stats = {
            'rows_pruned': pruning_stats.get('rows_pruned', 0),
            'rows_kept_after_pruning': pruning_stats.get('rows_kept_after_pruning', 0),
            'prune_reasons_sample': pruning_stats.get('prune_reasons_sample', []),
            'total_prune_reasons': pruning_stats.get('total_prune_reasons', 0),
            'original_size': dataset_info.get('original_size', 0),
            'final_size_before_sampling': dataset_info.get('final_size_before_sampling', 0),
            'sampled_size': dataset_info.get('sampled_size', 0)
        }
        
        return combined_stats
    
    # =============================================================================
    # EXPERIMENT FILE DISCOVERY AND LOADING
    # =============================================================================
    
    def find_experiment_files(self) -> List[str]:
        """
        Find all experiment result files.
        
        Returns:
            list: Paths to found experiment files
        """
        pattern = os.path.join(Config.RESPONSES_DIR, "inference_*.json")
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
        Evaluate experiment results from loaded data using custom metrics.
        """
        try:
            # Extract responses and expected outputs with full response data
            generated_responses, expected_outputs, response_data_list = self.extract_responses_and_expected(experiment_data)
            
            if not generated_responses or not expected_outputs:
                logger.error("No valid responses found for evaluation")
                return None
            
            # Extract metadata from file content
            metadata = self.extract_metadata_from_file_data(experiment_data)
            setup_name = metadata['setup']
            
            # Extract pruning statistics from inference stage
            inference_pruning_stats = self.extract_pruning_stats_from_file_data(experiment_data)
            
            # Run comprehensive evaluation with custom metrics and pruning stats
            experiment_name = experiment_data.get('experiment_name', 'unknown')
            logger.info(f"Evaluating experiment: {experiment_name} (mode: {metadata['mode']}, setup: {setup_name})")
            
            # Log pruning summary from inference stage
            if inference_pruning_stats.get('rows_pruned', 0) > 0:
                logger.info(f"Inference pruning: {inference_pruning_stats['rows_pruned']} rows were pruned, "
                           f"{inference_pruning_stats['rows_kept_after_pruning']} rows kept")
            
            evaluation_result = self.evaluation_framework.evaluate_batch(
                generated_responses=generated_responses,
                expected_responses=expected_outputs,
                embedding_model=self.model_manager.embedding_model,
                batch_name=experiment_name,
                setup_name=setup_name,
                response_data_list=response_data_list,  # Pass full response data for custom metrics
                inference_pruning_stats=inference_pruning_stats  # Pass pruning stats from inference
            )
            
            # Add comprehensive metadata including extracted fields
            evaluation_result.update({
                'original_experiment_config': experiment_data.get('experiment_config', {}),
                'original_experiment_name': experiment_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'setup_name': setup_name,
                'extracted_metadata': metadata
            })
            
            logger.info(f"Evaluation completed for: {experiment_name}")
            
            # Log key performance metrics
            if evaluation_result.get('aggregated_scores'):
                agg_scores = evaluation_result['aggregated_scores']
                f1_mean = agg_scores.get('f1_score', {}).get('mean', 0)
                bleu_mean = agg_scores.get('bleu', {}).get('mean', 0)
                rouge1_mean = agg_scores.get('rouge1_f', {}).get('mean', 0)
                sem_sim_mean = agg_scores.get('semantic_similarity', {}).get('mean', 0)
                exact_match_mean = agg_scores.get('exact_match', {}).get('mean', 0)
                
                logger.info(f"Results - F1: {f1_mean:.3f}, BLEU: {bleu_mean:.3f}, ROUGE-1: {rouge1_mean:.3f}, "
                           f"Semantic Similarity: {sem_sim_mean:.3f}, Exact Match: {exact_match_mean:.3f}")
                
                # Log custom metrics if any were computed
                custom_metrics = [metric for metric in agg_scores.keys() 
                                if metric not in ['f1_score', 'bleu', 'rouge1_f', 'rouge2_f', 'rougeL_f',
                                                 'semantic_similarity', 'exact_match', 'precision', 'recall', 'jaccard']]
                if custom_metrics:
                    logger.info(f"Custom metrics computed: {', '.join(custom_metrics)}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating experiment: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def save_evaluation_results(self, evaluation_data: Dict[str, Any], experiment_name: str) -> str:
        """
        Save evaluation results to JSON file.
        
        Args:
            evaluation_data: Complete evaluation results
            experiment_name: Name of the original experiment
            
        Returns:
            str: Path to saved evaluation file
        """
        file_paths = Config.generate_file_paths(experiment_name)
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
    # SINGLE EXPERIMENT EVALUATION
    # =============================================================================
    
    def evaluate_experiment(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate a specific experiment by name.
        """
        logger.info(f"Evaluating experiment: {experiment_name}")
        
        # Find experiment file
        experiment_file = os.path.join(Config.RESPONSES_DIR, f"inference_{experiment_name}.json")
        
        if not os.path.exists(experiment_file):
            logger.error(f"Experiment file not found: {experiment_name}")
            return None
        
        # Load and evaluate experiment data
        experiment_data = self.load_experiment_results(experiment_file)
        if not experiment_data:
            return None

        evaluation_result = self.evaluate_experiment_from_data(experiment_data)
        if not evaluation_result:
            return None
        
        # Save evaluation results
        output_file = self.save_evaluation_results(evaluation_result, experiment_name)
        
        # Extract metadata for return
        metadata = self.extract_metadata_from_file_data(experiment_data)
        
        return {
            'experiment_name': experiment_name,
            'mode': metadata['mode'],
            'model': metadata['model'],
            'setup': metadata['setup'],
            'evaluation_file': output_file,
            'metrics': evaluation_result.get('aggregated_scores', {}),
            'num_samples': evaluation_result.get('num_samples', 0),
            'num_valid_evaluations': evaluation_result.get('num_valid_evaluations', 0),
            'pruning_summary': evaluation_result.get('inference_pruning_stats', {})
        }
    
    # =============================================================================
    # BATCH EVALUATION
    # =============================================================================
    
    def evaluate_all_experiments(self) -> List[Dict[str, Any]]:
        """
        Evaluate all experiments.
        """
        logger.info("Evaluating all experiments")
        
        # Find all experiment files
        all_experiment_files = self.find_experiment_files()
        
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
                
                evaluation_result = self.evaluate_experiment_from_data(experiment_data)
                if not evaluation_result:
                    continue
                
                # Save evaluation results
                output_file = self.save_evaluation_results(evaluation_result, experiment_name)
                
                # Extract metadata and pruning stats for summary
                metadata = self.extract_metadata_from_file_data(experiment_data)
                pruning_stats = evaluation_result.get('inference_pruning_stats', {})
                
                results.append({
                    'experiment_name': experiment_name,
                    'mode': metadata['mode'],
                    'model': metadata['model'],
                    'setup': metadata['setup'],
                    'prompt': metadata['prompt'],
                    'evaluation_file': output_file,
                    'metrics': evaluation_result.get('aggregated_scores', {}),
                    'num_samples': evaluation_result.get('num_samples', 0),
                    'num_valid_evaluations': evaluation_result.get('num_valid_evaluations', 0),
                    'pruning_summary': {
                        'rows_pruned': pruning_stats.get('rows_pruned', 0),
                        'rows_kept': pruning_stats.get('rows_kept_after_pruning', 0),
                        'original_size': pruning_stats.get('original_size', 0)
                    }
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {file_path}: {e}")
                continue
        
        logger.info(f"Successfully evaluated {len(results)} out of {len(all_experiment_files)} experiments")
        
        # Log overall pruning summary
        if results:
            total_pruned = sum(r['pruning_summary']['rows_pruned'] for r in results)
            total_kept = sum(r['pruning_summary']['rows_kept'] for r in results)
            total_original = sum(r['pruning_summary']['original_size'] for r in results)
            
            if total_pruned > 0:
                logger.info(f"Overall pruning summary: {total_pruned} rows pruned out of {total_original} original rows ({total_pruned/total_original*100:.1f}%)")
        
        # Generate aggregated summary comparing all prompts and models by setup
        if results:
            logger.info("Generating aggregated summary for all experiments...")
            self.generate_aggregated_summary(results)
        
        return results
    
    # =============================================================================
    # AGGREGATED SUMMARY GENERATION
    # =============================================================================
    
    def generate_aggregated_summary(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate aggregated summary comparing prompts and models organized by setup.
        
        Args:
            results: List of evaluation results from evaluate_all_experiments()
            
        Returns:
            str: Path to saved summary file
        """
        if not results:
            logger.warning("No results to aggregate")
            return ""
        
        logger.info(f"Generating aggregated summary from {len(results)} experiments")
        
        # Group results by setup -> prompt -> model
        by_setup = {}
        
        for result in results:
            setup = result.get('setup', 'unknown')
            prompt = result.get('prompt', 'unknown')
            model = result.get('model', 'unknown')
            
            # Initialize nested structure
            if setup not in by_setup:
                by_setup[setup] = {}
            if prompt not in by_setup[setup]:
                by_setup[setup][prompt] = {}
            if model not in by_setup[setup][prompt]:
                by_setup[setup][prompt][model] = []
            
            by_setup[setup][prompt][model].append(result)
        
        # Compute aggregate statistics for each setup -> prompt -> model combination
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'unique_setups': len(by_setup),
            'by_setup': {},
            'overall_statistics': {}
        }
        
        # Process each setup
        for setup_name, setup_data in by_setup.items():
            summary['by_setup'][setup_name] = {
                'unique_prompts': len(setup_data),
                'unique_models': len(set(model for prompt_data in setup_data.values() for model in prompt_data.keys())),
                'by_prompt': {}
            }
            
            # Process each prompt within the setup
            for prompt_name, prompt_data in setup_data.items():
                summary['by_setup'][setup_name]['by_prompt'][prompt_name] = {
                    'unique_models': len(prompt_data),
                    'by_model': {}
                }
                
                # Process each model within the prompt
                for model_name, model_results in prompt_data.items():
                    aggregated_metrics = self._aggregate_results_group(
                        model_results,
                        group_name=f"Setup: {setup_name}, Prompt: {prompt_name}, Model: {model_name}"
                    )
                    
                    summary['by_setup'][setup_name]['by_prompt'][prompt_name]['by_model'][model_name] = aggregated_metrics
        
        # Compute overall statistics across all experiments
        summary['overall_statistics'] = {
            'by_setup': self._compute_setup_statistics(by_setup, results),
            'by_prompt': self._compute_prompt_statistics(results),
            'by_model': self._compute_model_statistics(results),
            'best_performers': self._find_best_performers(results)
        }
        
        # Save summary file
        summary_file = os.path.join(Config.EVALUATIONS_DIR, "aggregated_summary.json")
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Aggregated summary saved: {summary_file}")
            
            # Log key findings
            self._log_summary_highlights(summary)
            
            return summary_file
        except Exception as e:
            logger.error(f"Error saving aggregated summary: {e}")
            return ""
    
    def _aggregate_results_group(self, results: List[Dict[str, Any]], group_name: str) -> Dict[str, Any]:
        """
        Aggregate metrics across a group of results.
        
        Args:
            results: List of evaluation results to aggregate
            group_name: Name of this group for logging
            
        Returns:
            dict: Aggregated statistics for this group
        """
        if not results:
            return {}
        
        # Collect all metric values across experiments
        all_metrics = {}
        
        for result in results:
            metrics = result.get('metrics', {})
            for metric_name, metric_stats in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                
                # Use mean value if available, otherwise the value itself
                if isinstance(metric_stats, dict) and 'mean' in metric_stats:
                    all_metrics[metric_name].append(metric_stats['mean'])
                elif isinstance(metric_stats, (int, float)):
                    all_metrics[metric_name].append(metric_stats)
        
        # Compute aggregate statistics
        aggregated = {
            'num_experiments': len(results),
            'experiments': [r['experiment_name'] for r in results],
            'metrics': {}
        }
        
        for metric_name, values in all_metrics.items():
            if values:
                aggregated['metrics'][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'values': [float(v) for v in values]
                }
        
        return aggregated
    
    def _compute_setup_statistics(self, by_setup: Dict, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics aggregated by setup."""
        setup_stats = {}
        
        for setup_name in by_setup.keys():
            setup_results = [r for r in results if r.get('setup') == setup_name]
            setup_stats[setup_name] = self._aggregate_results_group(
                setup_results,
                group_name=f"Setup: {setup_name}"
            )
        
        return setup_stats
    
    def _compute_prompt_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics aggregated by prompt across all setups."""
        by_prompt = {}
        
        for result in results:
            prompt = result.get('prompt', 'unknown')
            if prompt not in by_prompt:
                by_prompt[prompt] = []
            by_prompt[prompt].append(result)
        
        prompt_stats = {}
        for prompt_name, prompt_results in by_prompt.items():
            prompt_stats[prompt_name] = self._aggregate_results_group(
                prompt_results,
                group_name=f"Prompt: {prompt_name}"
            )
        
        return prompt_stats
    
    def _compute_model_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics aggregated by model across all setups."""
        by_model = {}
        
        for result in results:
            model = result.get('model', 'unknown')
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)
        
        model_stats = {}
        for model_name, model_results in by_model.items():
            model_stats[model_name] = self._aggregate_results_group(
                model_results,
                group_name=f"Model: {model_name}"
            )
        
        return model_stats
    
    def _find_best_performers(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find best performing setup-prompt-model combinations for each metric."""
        # Collect all unique metric names across results
        all_metric_names = set()
        for result in results:
            all_metric_names.update(result.get('metrics', {}).keys())
        
        best_performers = {}
        
        # Find best performer for each metric
        for metric_name in all_metric_names:
            best = max(results, key=lambda r: r.get('metrics', {}).get(metric_name, {}).get('mean', 0), default=None)
            if best:
                best_performers[f'by_{metric_name}'] = {
                    'experiment_name': best['experiment_name'],
                    'setup': best.get('setup', 'unknown'),
                    'prompt': best.get('prompt', 'unknown'),
                    'model': best.get('model', 'unknown'),
                    'score': best.get('metrics', {}).get(metric_name, {}).get('mean', 0)
                }
        
        return best_performers
    
    def _log_summary_highlights(self, summary: Dict[str, Any]):
        """Log key highlights from the aggregated summary."""
        logger.info("=" * 60)
        logger.info("AGGREGATED SUMMARY HIGHLIGHTS")
        logger.info("=" * 60)
        logger.info(f"Total experiments: {summary['total_experiments']}")
        logger.info(f"Unique setups: {summary['unique_setups']}")
        
        # Log setup breakdown
        for setup_name, setup_data in summary['by_setup'].items():
            logger.info(f"  Setup '{setup_name}': {setup_data['unique_prompts']} prompts, {setup_data['unique_models']} models")
        
        # Log best performers
        best = summary.get('overall_statistics', {}).get('best_performers', {})
        
        if best.get('by_f1_score'):
            logger.info(f"Best F1 Score: {best['by_f1_score']['score']:.3f} "
                       f"(Setup: {best['by_f1_score']['setup']}, "
                       f"Prompt: {best['by_f1_score']['prompt']}, "
                       f"Model: {best['by_f1_score']['model']})")
        
        if best.get('by_bleu'):
            logger.info(f"Best BLEU: {best['by_bleu']['score']:.3f} "
                       f"(Setup: {best['by_bleu']['setup']}, "
                       f"Prompt: {best['by_bleu']['prompt']}, "
                       f"Model: {best['by_bleu']['model']})")
        
        if best.get('by_rouge1_f'):
            logger.info(f"Best ROUGE-1: {best['by_rouge1_f']['score']:.3f} "
                       f"(Setup: {best['by_rouge1_f']['setup']}, "
                       f"Prompt: {best['by_rouge1_f']['prompt']}, "
                       f"Model: {best['by_rouge1_f']['model']})")
        
        if best.get('by_semantic_similarity'):
            logger.info(f"Best Semantic Similarity: {best['by_semantic_similarity']['score']:.3f} "
                       f"(Setup: {best['by_semantic_similarity']['setup']}, "
                       f"Prompt: {best['by_semantic_similarity']['prompt']}, "
                       f"Model: {best['by_semantic_similarity']['model']})")
        
        logger.info("=" * 60)
    
    # =============================================================================
    # EVALUATION SUMMARY AND ANALYTICS
    # =============================================================================
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of evaluation results across experiments.
        """
        # Find evaluation files
        all_evaluation_files = glob.glob(os.path.join(Config.EVALUATIONS_DIR, "evaluation_*.json"))
        
        if not all_evaluation_files:
            return {"message": "No evaluation results found"}
        
        # Load evaluations
        valid_evaluations = []
        total_pruned_across_all = 0
        total_original_across_all = 0
        
        for file_path in all_evaluation_files:
            try:
                with open(file_path, 'r') as f:
                    eval_data = json.load(f)
                
                valid_evaluations.append(eval_data)
                
                # Accumulate pruning statistics
                pruning_stats = eval_data.get('inference_pruning_stats', {})
                total_pruned_across_all += pruning_stats.get('rows_pruned', 0)
                total_original_across_all += pruning_stats.get('original_size', 0)
                
            except Exception as e:
                logger.warning(f"Error loading evaluation file {file_path}: {e}")
                continue
        
        if not valid_evaluations:
            return {"message": "No valid evaluation results found"}
        
        # Compute summary statistics for all default metrics
        all_f1_scores = []
        all_bleu_scores = []
        all_rouge1_scores = []
        all_rouge2_scores = []
        all_rougeL_scores = []
        all_semantic_similarities = []
        all_exact_matches = []
        total_samples = 0
        
        for eval_data in valid_evaluations:
            agg_scores = eval_data.get('aggregated_scores', {})
            if 'f1_score' in agg_scores:
                all_f1_scores.append(agg_scores['f1_score']['mean'])
            if 'bleu' in agg_scores:
                all_bleu_scores.append(agg_scores['bleu']['mean'])
            if 'rouge1_f' in agg_scores:
                all_rouge1_scores.append(agg_scores['rouge1_f']['mean'])
            if 'rouge2_f' in agg_scores:
                all_rouge2_scores.append(agg_scores['rouge2_f']['mean'])
            if 'rougeL_f' in agg_scores:
                all_rougeL_scores.append(agg_scores['rougeL_f']['mean'])
            if 'semantic_similarity' in agg_scores:
                all_semantic_similarities.append(agg_scores['semantic_similarity']['mean'])
            if 'exact_match' in agg_scores:
                all_exact_matches.append(agg_scores['exact_match']['mean'])
            
            total_samples += eval_data.get('num_valid_evaluations', 0)
        
        summary = {
            'total_evaluations': len(valid_evaluations),
            'total_samples_evaluated': total_samples,
            'overall_pruning_stats': {
                'total_rows_pruned': total_pruned_across_all,
                'total_original_rows': total_original_across_all,
                'pruning_rate': (total_pruned_across_all / total_original_across_all * 100) if total_original_across_all > 0 else 0
            }
        }
        
        # Add statistics for each metric
        if all_f1_scores:
            summary['f1_score_stats'] = {
                'mean': np.mean(all_f1_scores),
                'std': np.std(all_f1_scores),
                'min': np.min(all_f1_scores),
                'max': np.max(all_f1_scores),
                'median': np.median(all_f1_scores)
            }
        
        if all_bleu_scores:
            summary['bleu_stats'] = {
                'mean': np.mean(all_bleu_scores),
                'std': np.std(all_bleu_scores),
                'min': np.min(all_bleu_scores),
                'max': np.max(all_bleu_scores),
                'median': np.median(all_bleu_scores)
            }
        
        if all_rouge1_scores:
            summary['rouge1_f_stats'] = {
                'mean': np.mean(all_rouge1_scores),
                'std': np.std(all_rouge1_scores),
                'min': np.min(all_rouge1_scores),
                'max': np.max(all_rouge1_scores),
                'median': np.median(all_rouge1_scores)
            }
        
        if all_rouge2_scores:
            summary['rouge2_f_stats'] = {
                'mean': np.mean(all_rouge2_scores),
                'std': np.std(all_rouge2_scores),
                'min': np.min(all_rouge2_scores),
                'max': np.max(all_rouge2_scores),
                'median': np.median(all_rouge2_scores)
            }
        
        if all_rougeL_scores:
            summary['rougeL_f_stats'] = {
                'mean': np.mean(all_rougeL_scores),
                'std': np.std(all_rougeL_scores),
                'min': np.min(all_rougeL_scores),
                'max': np.max(all_rougeL_scores),
                'median': np.median(all_rougeL_scores)
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
    
    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments side by side.
        """
        logger.info(f"Comparing {len(experiment_names)} experiments")
        
        # Load evaluation results for all experiments
        evaluations = []
        for exp_name in experiment_names:
            eval_file = os.path.join(Config.EVALUATIONS_DIR, f"evaluation_{exp_name}.json")
            
            if os.path.exists(eval_file):
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
            'experiment_names': [eval_data.get('original_experiment_name', 'Unknown') for eval_data in evaluations],
            'metric_comparison': {},
            'rankings': {},
            'best_performers': {},
            'pruning_comparison': []
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
        
        # Add pruning statistics comparison
        for eval_data in evaluations:
            pruning_stats = eval_data.get('inference_pruning_stats', {})
            exp_name = eval_data.get('original_experiment_name', 'Unknown')
            
            comparison['pruning_comparison'].append({
                'experiment_name': exp_name,
                'rows_pruned': pruning_stats.get('rows_pruned', 0),
                'rows_kept': pruning_stats.get('rows_kept_after_pruning', 0),
                'original_size': pruning_stats.get('original_size', 0),
                'pruning_rate': (pruning_stats.get('rows_pruned', 0) / pruning_stats.get('original_size', 1) * 100) if pruning_stats.get('original_size', 0) > 0 else 0
            })
        
        return comparison