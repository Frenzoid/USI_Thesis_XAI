#!/usr/bin/env python3
"""
XAI Explanation Evaluation System - Comprehensive CLI

This script provides a command-line interface for running experiments to evaluate
Large Language Models' alignment with user study results in XAI explanation evaluation.
Handles multiple space-separated inputs with intelligent compatibility checking.
"""

import argparse
import sys
import os
import shutil
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass

# Load environment variables first
try:
    from dotenv import load_dotenv
    from pathlib import Path
    
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print("No .env file found. Create one from .env.template with your API keys")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")

from config import Config
from utils import setup_logging, initialize_system, print_system_status, validate_gpu_requirements_for_command
from experiment_runner import ExperimentRunner
from evaluator_runner import EvaluationRunner
from plotter import PlottingRunner
from dataset_manager import DatasetManager
from prompt_manager import PromptManager

# =============================================================================
# INTELLIGENT ARGUMENT RESOLVER
# =============================================================================

@dataclass
class ExperimentCombination:
    """A valid experiment combination"""
    model: str
    dataset: str
    prompt: str
    mode: str
    
    def __str__(self):
        return f"{self.model}+{self.dataset}+{self.prompt}+{self.mode}"

class ArgumentResolver:
    """
    Intelligent argument resolver that supports multiple space-separated inputs
    with compatibility checking and smart expansion of missing arguments.
    """
    
    def __init__(self, prompts_config: dict, datasets_config: dict, models_config: dict):
        self.prompts_config = prompts_config
        self.datasets_config = datasets_config
        self.models_config = models_config
        
        # Precompute all available options
        self.all_models = list(models_config.keys())
        self.all_datasets = list(datasets_config.keys())
        self.all_prompts = list(prompts_config.keys())
        self.all_modes = ['zero-shot', 'few-shot']
        
        # Build compatibility maps for efficient filtering
        self.dataset_to_prompts = self._build_dataset_to_prompts_map()
        self.prompt_to_dataset = self._build_prompt_to_dataset_map()
        self.mode_to_prompts = self._build_mode_to_prompts_map()
        self.prompt_to_mode = self._build_prompt_to_mode_map()
        
        self.logger = setup_logging("argument_resolver")
    
    def _build_dataset_to_prompts_map(self) -> Dict[str, List[str]]:
        """Build mapping from datasets to compatible prompts"""
        mapping = {}
        for dataset in self.all_datasets:
            mapping[dataset] = []
            for prompt_name, prompt_config in self.prompts_config.items():
                if prompt_config.get('compatible_dataset') == dataset:
                    mapping[dataset].append(prompt_name)
        return mapping
    
    def _build_prompt_to_dataset_map(self) -> Dict[str, str]:
        """Build mapping from prompts to their compatible dataset"""
        mapping = {}
        for prompt_name, prompt_config in self.prompts_config.items():
            mapping[prompt_name] = prompt_config.get('compatible_dataset', '')
        return mapping
    
    def _build_mode_to_prompts_map(self) -> Dict[str, List[str]]:
        """Build mapping from modes to compatible prompts"""
        mapping = {'zero-shot': [], 'few-shot': []}
        for prompt_name, prompt_config in self.prompts_config.items():
            mode = prompt_config.get('mode', 'zero-shot')
            if mode in mapping:
                mapping[mode].append(prompt_name)
        return mapping
    
    def _build_prompt_to_mode_map(self) -> Dict[str, str]:
        """Build mapping from prompts to their mode"""
        mapping = {}
        for prompt_name, prompt_config in self.prompts_config.items():
            mapping[prompt_name] = prompt_config.get('mode', 'zero-shot')
        return mapping
    
    def _parse_and_validate_list_arg(self, arg_value: Optional[str], all_options: List[str], 
                                   arg_name: str) -> List[str]:
        """
        Parse and validate space/comma separated argument values.
        
        Args:
            arg_value: Raw argument value from command line
            all_options: List of all valid options for this argument type
            arg_name: Name of argument for error messages
            
        Returns:
            List of validated argument values
            
        Raises:
            ValueError: If any provided values are invalid
        """
        if not arg_value:
            return []
        
        if arg_value.lower() == 'all':
            self.logger.info(f"Expanding '{arg_name}' to all {len(all_options)} available options")
            return all_options
        
        # Parse space or comma separated values
        if ',' in arg_value:
            parsed_values = [item.strip() for item in arg_value.split(',') if item.strip()]
        else:
            # Split on whitespace and filter empty strings
            parsed_values = [item.strip() for item in arg_value.split() if item.strip()]
        
        # Validate each parsed value
        invalid_values = []
        valid_values = []
        
        for value in parsed_values:
            if value in all_options:
                valid_values.append(value)
            else:
                invalid_values.append(value)
        
        # Report validation results
        if invalid_values:
            self.logger.error(f"Invalid {arg_name} values: {invalid_values}")
            self.logger.info(f"Valid {arg_name} options: {all_options}")
            raise ValueError(f"Invalid {arg_name} values: {invalid_values}")
        
        if valid_values:
            self.logger.info(f"Parsed {len(valid_values)} {arg_name}(s): {valid_values}")
        
        return valid_values
    
    def resolve_arguments(self, args, force: bool = False) -> List[ExperimentCombination]:
        """
        Main entry point for resolving arguments with intelligent compatibility.
        
        Args:
            args: Parsed command line arguments
            force: Whether to ignore compatibility errors
            
        Returns:
            List of valid experiment combinations
        """
        # Parse and validate user inputs
        try:
            specified_models = self._parse_and_validate_list_arg(
                args.model, self.all_models, "model"
            )
            specified_datasets = self._parse_and_validate_list_arg(
                args.dataset, self.all_datasets, "dataset"
            )
            specified_prompts = self._parse_and_validate_list_arg(
                args.prompt, self.all_prompts, "prompt"
            )
        except ValueError as e:
            self.logger.error(f"Argument validation failed: {e}")
            return []
        
        specified_mode = args.mode
        
        # Log what was specified
        self.logger.info("=== ARGUMENT RESOLUTION ===")
        self.logger.info(f"Models: {specified_models or 'None (will use all)'}")
        self.logger.info(f"Datasets: {specified_datasets or 'None (will infer/use all)'}")
        self.logger.info(f"Prompts: {specified_prompts or 'None (will use compatible)'}")
        self.logger.info(f"Mode: {specified_mode or 'None (will infer from prompts)'}")
        
        # Apply intelligent expansion and compatibility filtering
        return self._resolve_with_intelligent_expansion(
            specified_models, specified_datasets, specified_prompts, specified_mode, force
        )
    
    def _resolve_with_intelligent_expansion(self, models: List[str], datasets: List[str], 
                                          prompts: List[str], mode: Optional[str], 
                                          force: bool) -> List[ExperimentCombination]:
        """
        Systematically resolve all 16 possible argument combinations with proper constraint handling.
        
        Constraints:
        - Each prompt is compatible with exactly one dataset
        - Each prompt supports exactly one mode (zero-shot or few-shot)
        """
        final_combinations = []
        
        # Use specified models or default to all models
        if not models:
            models = self.all_models
            self.logger.info(f"No models specified - using all {len(models)} models")
        
        # Determine the universe of valid (dataset, prompt, mode) triplets based on constraints
        valid_triplets = self._get_valid_triplets(datasets, prompts, mode, force)
        
        if not valid_triplets:
            self.logger.error("No valid (dataset, prompt, mode) combinations found")
            return []
        
        # Create cartesian product of models × valid triplets
        for model in models:
            for dataset, prompt, prompt_mode in valid_triplets:
                final_combinations.append(ExperimentCombination(model, dataset, prompt, prompt_mode))
        
        # Log final summary
        self._log_combination_summary(final_combinations)
        
        return final_combinations
    
    def _get_valid_triplets(self, datasets: List[str], prompts: List[str], 
                           mode: Optional[str], force: bool) -> List[Tuple[str, str, str]]:
        """
        Get all valid (dataset, prompt, mode) triplets based on the specified constraints.
        
        Handles all 16 possible input combinations systematically.
        """
        valid_triplets = []
        
        # Determine which triplets to consider based on input combination
        has_datasets = bool(datasets)
        has_prompts = bool(prompts)  
        has_mode = bool(mode)
        
        self.logger.info(f"Resolving triplets - Datasets: {has_datasets}, Prompts: {has_prompts}, Mode: {has_mode}")
        
        if not has_datasets and not has_prompts and not has_mode:
            # Case 1: No constraints - all valid combinations
            self.logger.info("No constraints specified - using all compatible combinations")
            valid_triplets = self._build_all_valid_triplets()
            
        elif not has_datasets and not has_prompts and has_mode:
            # Case 2: Only mode specified - all prompts with that mode
            self.logger.info(f"Only mode '{mode}' specified - finding compatible prompts")
            compatible_prompts = self.mode_to_prompts.get(mode, [])
            for prompt in compatible_prompts:
                dataset = self.prompt_to_dataset[prompt]
                valid_triplets.append((dataset, prompt, mode))
                
        elif not has_datasets and has_prompts and not has_mode:
            # Case 3: Only prompts specified - infer datasets and modes
            self.logger.info("Only prompts specified - inferring datasets and modes")
            for prompt in prompts:
                dataset = self.prompt_to_dataset.get(prompt, '')
                prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')
                if dataset:
                    valid_triplets.append((dataset, prompt, prompt_mode))
                else:
                    self.logger.warning(f"Prompt '{prompt}' has no associated dataset")
                    
        elif not has_datasets and has_prompts and has_mode:
            # Case 4: Prompts + mode specified - validate compatibility
            self.logger.info(f"Prompts + mode '{mode}' specified - validating compatibility")
            for prompt in prompts:
                dataset = self.prompt_to_dataset.get(prompt, '')
                prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')
                
                if not dataset:
                    self.logger.warning(f"Prompt '{prompt}' has no associated dataset")
                    continue
                    
                if prompt_mode != mode:
                    if not force:
                        self.logger.error(f"Prompt '{prompt}' is {prompt_mode} but {mode} was requested")
                        continue
                    else:
                        self.logger.warning(f"Force mode: using {prompt_mode} prompt '{prompt}' despite {mode} request")
                        # Use the prompt's actual mode, not the requested mode
                        valid_triplets.append((dataset, prompt, prompt_mode))
                        continue
                
                valid_triplets.append((dataset, prompt, mode))
                
        elif has_datasets and not has_prompts and not has_mode:
            # Case 5: Only datasets specified - all prompts for those datasets
            self.logger.info("Only datasets specified - finding all compatible prompts")
            for dataset in datasets:
                compatible_prompts = self.dataset_to_prompts.get(dataset, [])
                for prompt in compatible_prompts:
                    prompt_mode = self.prompt_to_mode[prompt]
                    valid_triplets.append((dataset, prompt, prompt_mode))
                    
        elif has_datasets and not has_prompts and has_mode:
            # Case 6: Datasets + mode specified - prompts for datasets with that mode
            self.logger.info(f"Datasets + mode '{mode}' specified - finding compatible prompts")
            for dataset in datasets:
                compatible_prompts = self.dataset_to_prompts.get(dataset, [])
                mode_filtered_prompts = [p for p in compatible_prompts 
                                       if self.prompt_to_mode.get(p, 'zero-shot') == mode]
                for prompt in mode_filtered_prompts:
                    valid_triplets.append((dataset, prompt, mode))
                    
        elif has_datasets and has_prompts and not has_mode:
            # Case 7: Datasets + prompts specified - validate compatibility
            self.logger.info("Datasets + prompts specified - validating compatibility") 
            for prompt in prompts:
                expected_dataset = self.prompt_to_dataset.get(prompt, '')
                prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')
                
                if not expected_dataset:
                    self.logger.warning(f"Prompt '{prompt}' has no associated dataset")
                    continue
                    
                if expected_dataset not in datasets:
                    if not force:
                        self.logger.error(f"Prompt '{prompt}' expects dataset '{expected_dataset}' but only {datasets} specified")
                        continue
                    else:
                        self.logger.warning(f"Force mode: using prompt '{prompt}' despite dataset mismatch")
                        # Use the prompt's expected dataset
                        valid_triplets.append((expected_dataset, prompt, prompt_mode))
                        continue
                
                valid_triplets.append((expected_dataset, prompt, prompt_mode))
                
        elif has_datasets and has_prompts and has_mode:
            # Case 8: All specified - validate all constraints
            self.logger.info("All constraints specified - validating complete compatibility")
            for prompt in prompts:
                expected_dataset = self.prompt_to_dataset.get(prompt, '')
                prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')
                
                dataset_valid = not expected_dataset or expected_dataset in datasets
                mode_valid = prompt_mode == mode
                
                if not expected_dataset:
                    self.logger.warning(f"Prompt '{prompt}' has no associated dataset")
                    continue
                
                if not dataset_valid and not mode_valid:
                    if not force:
                        self.logger.error(f"Prompt '{prompt}' incompatible: expects '{expected_dataset}' (got {datasets}) and is {prompt_mode} (requested {mode})")
                        continue
                    else:
                        self.logger.warning(f"Force mode: using prompt '{prompt}' despite all mismatches")
                        valid_triplets.append((expected_dataset, prompt, prompt_mode))
                        continue
                        
                elif not dataset_valid:
                    if not force:
                        self.logger.error(f"Prompt '{prompt}' expects dataset '{expected_dataset}' but only {datasets} specified")
                        continue
                    else:
                        self.logger.warning(f"Force mode: using prompt '{prompt}' despite dataset mismatch")
                        valid_triplets.append((expected_dataset, prompt, prompt_mode))
                        continue
                        
                elif not mode_valid:
                    if not force:
                        self.logger.error(f"Prompt '{prompt}' is {prompt_mode} but {mode} was requested")
                        continue
                    else:
                        self.logger.warning(f"Force mode: using {prompt_mode} prompt '{prompt}' despite {mode} request")
                        valid_triplets.append((expected_dataset, prompt, prompt_mode))
                        continue
                
                # All constraints satisfied
                valid_triplets.append((expected_dataset, prompt, mode))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_triplets = []
        for triplet in valid_triplets:
            if triplet not in seen:
                seen.add(triplet)
                unique_triplets.append(triplet)
        
        self.logger.info(f"Found {len(unique_triplets)} valid (dataset, prompt, mode) combinations")
        return unique_triplets
    
    def _build_all_valid_triplets(self) -> List[Tuple[str, str, str]]:
        """Build all possible valid (dataset, prompt, mode) triplets"""
        triplets = []
        for dataset in self.all_datasets:
            for prompt in self.dataset_to_prompts.get(dataset, []):
                mode = self.prompt_to_mode[prompt]
                triplets.append((dataset, prompt, mode))
        return triplets
    
    def _log_combination_summary(self, combinations: List[ExperimentCombination]):
        """Log a summary of the final combinations"""
        self.logger.info(f"=== FINAL COMBINATIONS: {len(combinations)} ===")
        
        if not combinations:
            return
            
        # Group by mode for summary
        mode_counts = {}
        dataset_counts = {}
        model_counts = {}
        
        for combo in combinations:
            mode_counts[combo.mode] = mode_counts.get(combo.mode, 0) + 1
            dataset_counts[combo.dataset] = dataset_counts.get(combo.dataset, 0) + 1
            model_counts[combo.model] = model_counts.get(combo.model, 0) + 1
        
        self.logger.info(f"Breakdown by mode: {dict(mode_counts)}")
        self.logger.info(f"Breakdown by dataset: {dict(dataset_counts)}")  
        self.logger.info(f"Breakdown by model: {dict(model_counts)}")
        
        # Show sample combinations
        sample_count = min(5, len(combinations))
        if sample_count > 0:
            self.logger.info(f"Sample combinations (showing first {sample_count}):")
            for i in range(sample_count):
                combo = combinations[i]
                self.logger.info(f"  - {combo}")
            
            if len(combinations) > sample_count:
                self.logger.info(f"  ... and {len(combinations) - sample_count} more")
    
    def _build_all_compatible_combinations(self) -> List[ExperimentCombination]:
        """Build all possible valid combinations"""
        combinations = []
        for model in self.all_models:
            for dataset in self.all_datasets:
                for prompt in self.dataset_to_prompts.get(dataset, []):
                    mode = self.prompt_to_mode[prompt]
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        return combinations

# =============================================================================
# MAIN CLI FUNCTIONS
# =============================================================================

def setup_cli_logging():
    """Setup logging for CLI"""
    logger = setup_logging("main", "INFO")
    return logger

def validate_experiment_type(experiment_type: str):
    """Validate experiment type"""
    if not Config.validate_experiment_type(experiment_type):
        raise ValueError(f"Unsupported experiment type: {experiment_type}. Supported: {Config.EXPERIMENT_TYPES}")

def validate_local_model_capability(models_to_run: List[str], models_config: dict, force: bool = False) -> bool:
    """
    Validate that the system can run local models if they are requested.
    
    Args:
        models_to_run: List of model names to run
        models_config: Models configuration dictionary
        force: Whether to force run despite warnings
        
    Returns:
        bool: True if can proceed, False if should abort
    """
    # Check if any local models are being requested
    local_models_requested = []
    api_models_requested = []
    
    for model in models_to_run:
        if model in models_config:
            if models_config[model]['type'] == 'local':
                local_models_requested.append(model)
            else:
                api_models_requested.append(model)
    
    # If no local models requested, proceed
    if not local_models_requested:
        return True
    
    # Check GPU status
    from utils import check_gpu_availability
    gpu_status = check_gpu_availability()
    
    # Get logger
    logger = setup_logging("model_validation", "INFO")
    
    # Warn about local models on CPU or insufficient GPU
    if not gpu_status['cuda_available']:
        if not gpu_status['torch_available']:
            logger.error("Cannot run local models: PyTorch not available")
            logger.info("Available options:")
            logger.info("   - Install PyTorch with CUDA support for GPU acceleration")
            logger.info("   - Use API-based models only (GPT, Gemini, Claude)")
            return False
        else:
            logger.warning("Local models requested but no GPU available")
            logger.warning(f"   Requested local models: {', '.join(local_models_requested)}")
            logger.warning("   This will be VERY slow and may require >16GB RAM")
            
            if not force:
                response = input("Continue with CPU-only local model inference? (y/N): ")
                if response.lower() != 'y':
                    logger.info("Consider using API models for better performance:")
                    if api_models_requested:
                        logger.info(f"   Available API models in your selection: {', '.join(api_models_requested)}")
                    else:
                        api_models = [m for m, c in models_config.items() if c['type'] == 'api']
                        logger.info(f"   Available API models: {', '.join(api_models[:3])}...")
                    return False
    
    elif not gpu_status['can_run_local_models']:
        logger.warning(f"GPU memory may be insufficient for local models ({gpu_status['total_memory']:.1f} GB available)")
        logger.warning("   Consider using smaller models or API-based models for better reliability")
    
    return True

def run_baseline_experiment_command(args):
    """Run baseline experiments with intelligent argument resolution"""
    logger.info("Starting baseline experiment execution with intelligent argument resolution...")
    
    # Load configurations
    try:
        prompts_config = Config.load_prompts_config()
        datasets_config = Config.load_datasets_config()
        models_config = Config.load_models_config()
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        return False
    
    # Initialize the argument resolver
    resolver = ArgumentResolver(prompts_config, datasets_config, models_config)
    
    # Resolve all argument combinations
    try:
        combinations = resolver.resolve_arguments(args, force=args.force)
    except Exception as e:
        logger.error(f"Error resolving arguments: {e}")
        return False
    
    if not combinations:
        logger.error("No valid experiment combinations found")
        logger.info("Check your arguments and use --force to ignore compatibility warnings")
        logger.info("Use 'python main.py list-options' to see available options")
        return False
    
    # Group combinations by mode for organized execution
    combinations_by_mode = {}
    for combo in combinations:
        mode = combo.mode
        if mode not in combinations_by_mode:
            combinations_by_mode[mode] = []
        combinations_by_mode[mode].append(combo)
    
    # Validate few-shot row parameter
    if args.few_shot_row is not None:
        if args.few_shot_row < 0:
            logger.error(f"Few-shot row index must be non-negative, got: {args.few_shot_row}")
            return False
        if 'few-shot' not in combinations_by_mode:
            logger.error("Cannot specify --few-shot-row when no few-shot experiments will be run")
            return False
    
    # Validate model capability
    unique_models = list(set(combo.model for combo in combinations))
    if not validate_local_model_capability(unique_models, models_config, args.force):
        return False
    
    # Ask for confirmation if many experiments
    if len(combinations) > 10 and not args.force:
        print(f"\n⚠️  About to run {len(combinations)} experiments. This may take a long time.⚠️")
        print("📊 Breakdown:")
        for mode, mode_combos in combinations_by_mode.items():
            print(f"  {mode}: {len(mode_combos)} experiments")
        
        response = input("\nProceed? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled by user")
            return False
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Execute all combinations
    results = []
    success_count = 0
    failure_count = 0
    
    total_experiments = len(combinations)
    
    for i, combo in enumerate(combinations, 1):
        try:
            logger.info(f"Running experiment {i}/{total_experiments}: {combo}")
            
            experiment_config = {
                'experiment_type': 'baseline',
                'model': combo.model,
                'dataset': combo.dataset,
                'prompt': combo.prompt,
                'mode': combo.mode,
                'few_shot_row': args.few_shot_row,
                'size': args.size,
                'temperature': args.temperature
            }
            
            result = runner.run_baseline_experiment(experiment_config)
            
            if result:
                results.append(result)
                success_count += 1
                logger.info(f"Completed ({i}/{total_experiments}): {result['experiment_name']}")
            else:
                failure_count += 1
                logger.error(f"Failed ({i}/{total_experiments}): {combo}")
        
        except Exception as e:
            failure_count += 1
            logger.error(f"Error in experiment {i}/{total_experiments} ({combo}): {e}")
            continue
    
    # Final summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT EXECUTION COMPLETE")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {failure_count}")
    logger.info(f"Total: {total_experiments} combinations processed")
    
    if results:
        logger.info("Generated experiment files:")
        for result in results[:10]:  # Show first 10
            logger.info(f"  - {result['output_file']}")
        if len(results) > 10:
            logger.info(f"  ... and {len(results) - 10} more files")
    
    return success_count > 0

# =============================================================================
# OTHER COMMAND HANDLERS (unchanged)
# =============================================================================

def evaluate_command(args):
    """Evaluate experiment results"""
    logger.info("Starting evaluation...")
    
    # Validate experiment type
    if args.experiment_type:
        validate_experiment_type(args.experiment_type)
    
    # Initialize evaluation runner
    evaluator = EvaluationRunner()
    
    try:
        if args.experiment:
            # Evaluate specific experiment(s)
            experiments_to_evaluate = args.experiment.split(',')
            results = []
            
            for experiment_name in experiments_to_evaluate:
                result = evaluator.evaluate_experiment(experiment_name.strip(), args.experiment_type)
                if result:
                    results.append(result)
                    logger.info(f"Evaluated: {experiment_name}")
                else:
                    logger.error(f"Failed to evaluate: {experiment_name}")
            
        else:
            # Evaluate all experiments
            results = evaluator.evaluate_all_experiments(args.experiment_type)
        
        if results:
            logger.info(f"Successfully evaluated {len(results)} experiments")
            
            # Show summary
            for result in results:
                logger.info(f"Evaluation: {result['experiment_name']}")
                if 'metrics' in result:
                    metrics = result['metrics']
                    f1 = metrics.get('f1_score', {}).get('mean', 0)
                    similarity = metrics.get('semantic_similarity', {}).get('mean', 0)
                    logger.info(f"  F1: {f1:.3f}, Similarity: {similarity:.3f}")
            
            return True
        else:
            logger.error("No evaluations completed successfully")
            return False
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False

def plot_command(args):
    """Generate plots from evaluation results"""
    logger.info("Starting plot generation...")
    
    # Validate experiment type
    if args.experiment_type:
        validate_experiment_type(args.experiment_type)
    
    # Initialize plotting runner
    plotter = PlottingRunner()
    
    try:
        if args.experiment:
            # Plot specific experiment(s)
            experiments_to_plot = args.experiment.split(',')
            
            if args.compare:
                # Generate comparison plots
                results = plotter.create_comparison_plots(experiments_to_plot, args.experiment_type)
                if results:
                    logger.info(f"Generated comparison plots: {results}")
                else:
                    logger.error("Failed to generate comparison plots")
            else:
                # Generate individual plots
                success_count = 0
                for experiment_name in experiments_to_plot:
                    result = plotter.create_individual_plot(experiment_name.strip(), args.experiment_type)
                    if result:
                        success_count += 1
                        logger.info(f"Generated plot for: {experiment_name}")
                    else:
                        logger.error(f"Failed to generate plot for: {experiment_name}")
                
                return success_count > 0
        else:
            # Plot all experiments
            results = plotter.create_all_plots(args.experiment_type)
            if results:
                logger.info(f"Generated plots for {len(results)} experiments")
                return True
            else:
                logger.error("Failed to generate plots")
                return False
                
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        return False

def show_prompt_command(args):
    """Show a populated prompt template using dataset data"""
    logger.info("Showing populated prompt template...")
    
    # Load configurations
    try:
        prompts_config = Config.load_prompts_config()
        datasets_config = Config.load_datasets_config()
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        return False
    
    # Validate prompt exists
    if args.prompt not in prompts_config:
        logger.error(f"Unknown prompt: {args.prompt}")
        logger.info(f"Available prompts: {list(prompts_config.keys())}")
        return False
    
    # Initialize managers
    prompt_manager = PromptManager()
    dataset_manager = DatasetManager()
    
    try:
        # Get prompt configuration
        prompt_config = prompts_config[args.prompt]
        compatible_dataset = prompt_config.get('compatible_dataset', '')
        prompt_mode = prompt_config.get('mode', 'zero-shot')
        
        if not compatible_dataset:
            logger.error(f"Prompt '{args.prompt}' has no compatible dataset specified")
            return False
        
        # Load the compatible dataset
        logger.info(f"Loading dataset: {compatible_dataset}")
        dataset = dataset_manager.load_dataset(compatible_dataset)
        if dataset is None:
            logger.error(f"Failed to load dataset: {compatible_dataset}")
            return False
        
        # Validate dataset has required fields
        if not dataset_manager.validate_dataset_fields(compatible_dataset):
            logger.error(f"Dataset validation failed for {compatible_dataset}")
            return False
        
        # Determine which row to use
        if args.row is not None:
            if args.row < 0 or args.row >= len(dataset):
                logger.error(f"Row index {args.row} out of bounds for dataset with {len(dataset)} rows")
                return False
            row = dataset.iloc[args.row]
            logger.info(f"Using specified row {args.row}")
        else:
            # Use random row
            import random
            random.seed(Config.RANDOM_SEED)
            row_index = random.randint(0, len(dataset) - 1)
            row = dataset.iloc[row_index]
            logger.info(f"Using random row {row_index}")
        
        # Prepare the populated prompt
        populated_prompt = prompt_manager.prepare_prompt_for_row(
            prompt_name=args.prompt,
            row=row,
            dataset_name=compatible_dataset,
            mode=prompt_mode,
            dataset=dataset,  # Full dataset for few-shot
            few_shot_row=args.row if prompt_mode == 'few-shot' and args.row is not None else None
        )
        
        # Display results
        print("\n" + "=" * 80)
        print(f"POPULATED PROMPT TEMPLATE")
        print("=" * 80)
        print(f"Prompt: {args.prompt}")
        print(f"Mode: {prompt_mode}")
        print(f"Compatible Dataset: {compatible_dataset}")
        print(f"Row Used: {args.row if args.row is not None else row_index}")
        print(f"Dataset Size: {len(dataset)} rows")
        print("=" * 80)
        print(f"\nPOPULATED TEMPLATE:")
        print("-" * 40)
        print(populated_prompt)
        print("-" * 40)
        
        # Show the original data used
        dataset_config = datasets_config[compatible_dataset]
        question_fields = dataset_config.get('question_fields', [])
        answer_field = dataset_config.get('answer_field', '')
        
        print(f"\nORIGINAL DATA USED:")
        print("-" * 40)
        for field in question_fields:
            value = str(row.get(field, 'N/A')) if field in row else 'N/A'
            print(f"{field}: {value}")
        
        if answer_field and answer_field in row:
            answer_value = str(row.get(answer_field, 'N/A'))
            print(f"\nExpected Answer ({answer_field}): {answer_value}")
        
        print("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating populated prompt: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def download_datasets_command(args):
    """Download specified datasets"""
    logger.info("Starting dataset download...")
    
    manager = DatasetManager()
    all_datasets = list(manager.datasets_config.keys())
    
    # Parse dataset arguments properly
    if not args.dataset or args.dataset.lower() == 'all':
        datasets_to_download = all_datasets
    else:
        if ',' in args.dataset:
            datasets_to_download = [item.strip() for item in args.dataset.split(',')]
        else:
            datasets_to_download = args.dataset.split()
    
    success_count = 0
    failure_count = 0
    
    if not datasets_to_download:
        logger.warning("No datasets specified for download.")
        return False
        
    for dataset_name in datasets_to_download:
        if dataset_name not in manager.datasets_config:
            logger.error(f"Unknown dataset: {dataset_name}")
            failure_count += 1
            continue
            
        try:
            if manager.download_dataset(dataset_name):
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {e}")
            failure_count += 1
            
    logger.info(f"Dataset download completed: {success_count} successful, {failure_count} failed")
    
    return success_count > 0

def cleanup_command(args):
    """Clean up system files (datasets, logs, results, cache)"""
    logger.info("Starting system cleanup...")
    
    cleanup_types = {
        'datasets': Config.DATA_DIR,
        'logs': Config.LOGS_DIR,
        'results': Config.OUTPUTS_DIR,
        'cache': Config.CACHED_MODELS_DIR,
        'finetuned': Config.FINETUNED_MODELS_DIR
    }
    
    # Parse cleanup targets
    if args.target.lower() == 'all':
        targets = list(cleanup_types.keys())
    else:
        targets = args.target.split()
    
    cleaned_items = []
    errors = []
    
    for target in targets:
        if target not in cleanup_types:
            logger.warning(f"Unknown cleanup target: {target}")
            continue
        
        target_dir = cleanup_types[target]
        
        try:
            if os.path.exists(target_dir):
                if args.dry_run:
                    # Count items that would be deleted
                    item_count = len(os.listdir(target_dir)) if os.path.isdir(target_dir) else 1
                    logger.info(f"[DRY RUN] Would remove {item_count} items from {target_dir}")
                else:
                    # Actually delete
                    if os.path.isdir(target_dir):
                        shutil.rmtree(target_dir)
                        # Recreate empty directory
                        os.makedirs(target_dir, exist_ok=True)
                        logger.info(f"Cleaned directory: {target_dir}")
                    else:
                        os.remove(target_dir)
                        logger.info(f"Removed file: {target_dir}")
                    
                    cleaned_items.append(target)
            else:
                logger.info(f"Target does not exist: {target_dir}")
                
        except Exception as e:
            error_msg = f"Error cleaning {target}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Summary
    if args.dry_run:
        logger.info(f"[DRY RUN] Would clean {len(targets)} targets")
    else:
        logger.info(f"Cleanup completed: {len(cleaned_items)} targets cleaned")
        if errors:
            logger.warning(f"{len(errors)} errors occurred during cleanup")
    
    return len(errors) == 0

def list_available_options():
    """List available models, datasets, and prompts"""
    try:
        prompts_config = Config.load_prompts_config()
        datasets_config = Config.load_datasets_config()
        models_config = Config.load_models_config()
        
        print("\n=== AVAILABLE OPTIONS ===")
        
        print(f"\nModels ({len(models_config)}):")
        for model_name, config in models_config.items():
            model_type = config['type']
            finetuned = " (finetuned)" if config.get('finetuned', False) else ""
            print(f"  - {model_name}: {config['description']}{finetuned} [{model_type}]")
        
        print(f"\nDatasets ({len(datasets_config)}):")
        for dataset_name, config in datasets_config.items():
            print(f"  - {dataset_name}: {config['description']}")
        
        print(f"\nPrompts ({len(prompts_config)}):")
        for prompt_name, config in prompts_config.items():
            compatible = config['compatible_dataset']
            mode = config.get('mode', 'zero-shot')
            print(f"  - {prompt_name}: {config['description']} [for {compatible}, {mode}]")
        
        print(f"\nExperiment Types ({len(Config.EXPERIMENT_TYPES)}):")
        for exp_type in Config.EXPERIMENT_TYPES:
            print(f"  - {exp_type}")
        
    except Exception as e:
        print(f"Error loading configurations: {e}")

def list_commands_command():
    """List all available commands and their arguments with proper formatting"""
    print("\n=== AVAILABLE COMMANDS ===")
    
    print("\n1. run-baseline-exp")
    print("   Description: Run baseline inference experiments with intelligent argument resolution")
    print("   Arguments:")
    print("     --model MODELS           Model(s) to use (space-separated: 'gpt-4o-mini claude-3.5-sonnet' or 'all')")
    print("     --dataset DATASETS       Dataset(s) to use (space-separated: 'gmeg qald' or 'all')")
    print("     --prompt PROMPTS         Prompt(s) to use (space-separated: 'gmeg_v1 qald_v1' or 'all')")
    print("     --mode MODE              Prompting mode: zero-shot or few-shot")
    print("     --few-shot-row ROW       Specific row number for few-shot example (0-based, defaults to random)")
    print("     --size SIZE              Sample size (default: 50)")
    print("     --temperature TEMP       Temperature for generation (default: 0.1)")
    print("     --force                  Force run even with validation errors")
    
    print("\n   Intelligent Argument Resolution:")
    print("     • Specify any combination of arguments - system handles compatibility automatically")
    print("     • Unspecified arguments default to all compatible options")
    print("     • Dataset+prompt combinations are validated for compatibility")
    print("     • Mode filtering only uses prompts that support the specified mode")
    print("     • Multiple values supported: space-separated or comma-separated")
    
    print("\n2. evaluate")
    print("   Description: Evaluate experiment results using various metrics")
    print("   Arguments:")
    print("     --experiment NAMES       Specific experiment(s) to evaluate (comma-separated)")
    print("     --experiment-type TYPE   Filter by experiment type")
    
    print("\n3. plot")
    print("   Description: Generate plots and visualizations from evaluation results")
    print("   Arguments:")
    print("     --experiment NAMES       Specific experiment(s) to plot (comma-separated)")
    print("     --experiment-type TYPE   Filter by experiment type")
    print("     --compare               Generate comparison plots instead of individual")
    
    print("\n4. show-prompt")
    print("   Description: Display a populated prompt template using real dataset data")
    print("   Arguments:")
    print("     --prompt PROMPT_NAME     Name of prompt template to show (required)")
    print("     --row ROW_NUMBER         Specific row number to use from dataset (0-based, optional)")
    print("                              If not specified, uses a random row")
    
    print("\n5. download-datasets")
    print("   Description: Download specified datasets from configured sources")
    print("   Arguments:")
    print("     --dataset DATASETS       Dataset(s) to download (space-separated or 'all')")
    
    print("\n6. cleanup")
    print("   Description: Clean up system files and directories")
    print("   Arguments:")
    print("     --target TARGETS         What to clean: datasets, logs, results, cache, finetuned, all")
    print("     --dry-run               Show what would be cleaned without actually cleaning")
    
    print("\n7. list-options")
    print("   Description: List available models, datasets, prompts, and experiment types")
    print("   Arguments: None")
    
    print("\n8. list-commands / help / show-commands")
    print("   Description: Show this help message with all commands and their arguments")
    print("   Arguments: None")
    
    print("\n9. status")
    print("   Description: Check system status including configuration files and API keys")
    print("   Arguments: None")
    
    print("\n=== EXAMPLE USAGE ===")
    print("# Multiple models with space separation")
    print("python main.py run-baseline-exp --model 'gpt-4o-mini claude-3.5-sonnet'")
    print("")
    print("# Multiple datasets and automatic prompt selection")
    print("python main.py run-baseline-exp --dataset 'gmeg qald' --mode few-shot")
    print("")
    print("# Specific prompts with automatic dataset inference")
    print("python main.py run-baseline-exp --prompt 'gmeg_v1_basic qald_v1_basic'")
    print("")
    print("# Show populated prompt template with specific data")
    print("python main.py show-prompt --prompt gmeg_v1_basic --row 42")
    print("")
    print("# Show populated prompt template with random data")
    print("python main.py show-prompt --prompt gmeg_v2_enhanced")
    print("")
    print("# Mixed arguments with compatibility validation")
    print("python main.py run-baseline-exp --model 'gpt-4o-mini' --dataset gmeg --mode zero-shot")
    print("")
    print("# Force incompatible combinations")
    print("python main.py run-baseline-exp --prompt 'gmeg_v1_basic qald_v1_basic' --force")

def check_system_command(args):
    """Check system status"""
    logger.info("Checking system status...")
    
    # Use the enhanced system status display
    gpu_status, memory_status = print_system_status()
    
    # Check configuration files
    config_status = Config.validate_configuration_files()
    
    print("Configuration Files:")
    for config_type, exists in config_status.items():
        status = "✅" if exists else "❌"
        print(f"   {config_type}.json: {status}")
    
    # Check directories
    print("\nDirectories:")
    try:
        created_dirs = Config.create_directories()
        print(f"   Created/verified {len(created_dirs)} directories")
    except Exception as e:
        print(f"   Error creating directories: {e}")
    
    # Check API keys
    print("\nAPI Keys:")
    print(f"   OpenAI: {'✅' if Config.OPENAI_API_KEY else '❌'}")
    print(f"   Google GenAI: {'✅' if Config.GENAI_API_KEY else '❌'}")
    print(f"   Anthropic: {'✅' if Config.ANTHROPIC_API_KEY else '❌'}")
    
    # Recommendations based on system status
    print("\nRecommendations:")
    if not gpu_status['cuda_available']:
        if gpu_status['torch_available']:
            print("   - Consider API-based models for better performance")
            print("   - Local models will be very slow on CPU")
        else:
            print("   - Install PyTorch to enable local model support")
            print("   - Use API-based models (GPT, Gemini, Claude)")
    elif not gpu_status['can_run_local_models']:
        print(f"   - GPU memory ({gpu_status['total_memory']:.1f} GB) may limit local model size")
        print("   - Consider smaller models or API-based alternatives")
    else:
        print("   - System ready for both local and API-based models")
    
    return all(config_status.values())

def main():
    """Main CLI entry point"""
    # Initialize system
    try:
        initialize_system()
        global logger
        logger = setup_cli_logging()
        logger.info("XAI Explanation Evaluation System - Comprehensive CLI")
    except Exception as e:
        print(f"System initialization failed: {e}")
        return 1
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description="XAI Explanation Evaluation System - Multi-Input Argument Handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use 'python main.py list-commands' to see all available commands and examples."
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Baseline experiment command
    baseline_parser = subparsers.add_parser('run-baseline-exp', help='Run baseline inference experiments')
    baseline_parser.add_argument('--model', type=str,
                                help='Model(s) to use (space-separated: "gpt-4o-mini claude-3.5-sonnet" or "all")')
    baseline_parser.add_argument('--dataset', type=str,
                                help='Dataset(s) to use (space-separated: "gmeg qald" or "all")')
    baseline_parser.add_argument('--prompt', type=str,
                                help='Prompt(s) to use (space-separated: "gmeg_v1 qald_v1" or "all")')
    baseline_parser.add_argument('--mode', type=str, choices=['zero-shot', 'few-shot'],
                                help='Prompting mode: zero-shot or few-shot')
    baseline_parser.add_argument('--few-shot-row', type=int,
                                help='Specific row number to use for few-shot example (0-based indexing, defaults to random)')
    baseline_parser.add_argument('--size', type=int, default=Config.DEFAULT_SAMPLE_SIZE,
                                help=f'Sample size (default: {Config.DEFAULT_SAMPLE_SIZE})')
    baseline_parser.add_argument('--temperature', type=float, default=Config.DEFAULT_TEMPERATURE,
                                help=f'Temperature for generation (default: {Config.DEFAULT_TEMPERATURE})')
    baseline_parser.add_argument('--force', action='store_true',
                                help='Force run even with validation errors')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate experiment results')
    eval_parser.add_argument('--experiment', type=str,
                           help='Specific experiment(s) to evaluate (comma-separated)')
    eval_parser.add_argument('--experiment-type', type=str,
                           help='Filter by experiment type (baseline, masked, impersonation)')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Generate plots from evaluations')
    plot_parser.add_argument('--experiment', type=str,
                           help='Specific experiment(s) to plot (comma-separated)')
    plot_parser.add_argument('--experiment-type', type=str,
                           help='Filter by experiment type (baseline, masked, impersonation)')
    plot_parser.add_argument('--compare', action='store_true',
                           help='Generate comparison plots instead of individual plots')
    
    # Show prompt command
    show_prompt_parser = subparsers.add_parser('show-prompt', help='Display populated prompt template')
    show_prompt_parser.add_argument('--prompt', type=str, required=True,
                                   help='Name of prompt template to show')
    show_prompt_parser.add_argument('--row', type=int,
                                   help='Specific row number to use from dataset (0-based, uses random if not specified)')
    
    # Download datasets command
    download_parser = subparsers.add_parser('download-datasets', help='Download specified datasets')
    download_parser.add_argument('--dataset', type=str, default='all',
                                 help='Dataset(s) to download (space-separated, or "all" for all datasets)')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up system files and directories')
    cleanup_parser.add_argument('--target', type=str, default='all',
                               choices=['datasets', 'logs', 'results', 'cache', 'finetuned', 'all'],
                               help='What to clean: datasets, logs, results, cache, finetuned, all')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                               help='Show what would be cleaned without actually cleaning')
    
    # Utility commands
    list_parser = subparsers.add_parser('list-options', help='List available models, datasets, and prompts')
    list_commands_parser = subparsers.add_parser('list-commands', help='List all available commands and their arguments')
    help_parser = subparsers.add_parser('help', help='Show all available commands and their arguments')
    show_commands_parser = subparsers.add_parser('show-commands', help='Show all available commands and their arguments')
    status_parser = subparsers.add_parser('status', help='Check system status')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        if args.command == 'run-baseline-exp':
            success = run_baseline_experiment_command(args)
        elif args.command == 'evaluate':
            success = evaluate_command(args)
        elif args.command == 'plot':
            success = plot_command(args)
        elif args.command == 'show-prompt':
            success = show_prompt_command(args)
        elif args.command == 'download-datasets':
            success = download_datasets_command(args)
        elif args.command == 'cleanup':
            success = cleanup_command(args)
        elif args.command == 'list-options':
            list_available_options()
            success = True
        elif args.command == 'list-commands':
            list_commands_command()
            success = True
        elif args.command == 'help':
            list_commands_command()
            success = True
        elif args.command == 'show-commands':
            list_commands_command()
            success = True
        elif args.command == 'status':
            success = check_system_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            success = False
        
        if success:
            logger.info("Command completed successfully")
            return 0
        else:
            logger.error("Command failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())