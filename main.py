#!/usr/bin/env python3
"""
XAI Explanation Evaluation System CLI

This script provides a command-line interface for running experiments to evaluate
Large Language Models' alignment with user study results in XAI explanation evaluation.
Handles multiple space-separated inputs with intelligent compatibility checking.
Supports JSON/JSONL datasets with nested field path resolution.
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
    setup: str
    prompt: str
    mode: str
    
    def __str__(self):
        return f"{self.model}+{self.setup}+{self.prompt}+{self.mode}"

class ArgumentResolver:
    """
    Intelligent argument resolver that supports multiple space-separated inputs
    with compatibility checking and smart expansion of missing arguments.
    """
    
    def __init__(self, prompts_config: dict, setups_config: dict, models_config: dict):
        self.prompts_config = prompts_config
        self.setups_config = setups_config
        self.models_config = models_config
        
        # Precompute all available options
        self.all_models = list(models_config.keys())
        self.all_setups = list(setups_config.keys())
        self.all_prompts = list(prompts_config.keys())
        self.all_modes = ['zero-shot', 'few-shot']
        
        # Build compatibility maps for efficient filtering
        self.setup_to_prompts = self._build_setup_to_prompts_map()
        self.prompt_to_setup = self._build_prompt_to_setup_map()
        self.mode_to_prompts = self._build_mode_to_prompts_map()
        self.prompt_to_mode = self._build_prompt_to_mode_map()
        
        self.logger = setup_logging("argument_resolver")
    
    def _build_setup_to_prompts_map(self) -> Dict[str, List[str]]:
        """Build mapping from setups to compatible prompts"""
        mapping = {}
        for setup in self.all_setups:
            mapping[setup] = []
            for prompt_name, prompt_config in self.prompts_config.items():
                if prompt_config.get('compatible_setup') == setup:
                    mapping[setup].append(prompt_name)
        return mapping
    
    def _build_prompt_to_setup_map(self) -> Dict[str, str]:
        """Build mapping from prompts to their compatible setup"""
        mapping = {}
        for prompt_name, prompt_config in self.prompts_config.items():
            mapping[prompt_name] = prompt_config.get('compatible_setup', '')
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
            specified_setups = self._parse_and_validate_list_arg(
                args.setup, self.all_setups, "setup"
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
        self.logger.info(f"Setups: {specified_setups or 'None (will infer/use all)'}")
        self.logger.info(f"Prompts: {specified_prompts or 'None (will use compatible)'}")
        self.logger.info(f"Mode: {specified_mode or 'None (will infer from prompts)'}")
        
        # Apply intelligent expansion and compatibility filtering
        return self._resolve_with_intelligent_expansion(
            specified_models, specified_setups, specified_prompts, specified_mode, force
        )
    
    def _resolve_with_intelligent_expansion(self, models: List[str], setups: List[str], 
                                          prompts: List[str], mode: Optional[str], 
                                          force: bool) -> List[ExperimentCombination]:
        """
        Systematically resolve all 16 possible argument combinations with proper constraint handling.
        
        Constraints:
        - Each prompt is compatible with exactly one setup
        - Each prompt supports exactly one mode (zero-shot or few-shot)
        """
        final_combinations = []
        
        # Use specified models or default to all models
        if not models:
            models = self.all_models
            self.logger.info(f"No models specified - using all {len(models)} models")
        
        # Determine the universe of valid (setup, prompt, mode) triplets based on constraints
        valid_triplets = self._get_valid_triplets(setups, prompts, mode, force)
        
        if not valid_triplets:
            self.logger.error("No valid (setup, prompt, mode) combinations found")
            return []
        
        # Create cartesian product of models × valid triplets
        for model in models:
            for setup, prompt, prompt_mode in valid_triplets:
                final_combinations.append(ExperimentCombination(model, setup, prompt, prompt_mode))
        
        # Log final summary
        self._log_combination_summary(final_combinations)
        
        return final_combinations
    
    def _get_valid_triplets(self, setups: List[str], prompts: List[str], 
                           mode: Optional[str], force: bool) -> List[Tuple[str, str, str]]:
        """
        Get all valid (setup, prompt, mode) triplets based on the specified constraints.
        
        Handles all 16 possible input combinations systematically.
        """
        valid_triplets = []
        
        # Determine which triplets to consider based on input combination
        has_setups = bool(setups)
        has_prompts = bool(prompts)  
        has_mode = bool(mode)
        
        self.logger.info(f"Resolving triplets - Setups: {has_setups}, Prompts: {has_prompts}, Mode: {has_mode}")
        
        if not has_setups and not has_prompts and not has_mode:
            # Case 1: No constraints - all valid combinations
            self.logger.info("No constraints specified - using all compatible combinations")
            valid_triplets = self._build_all_valid_triplets()
            
        elif not has_setups and not has_prompts and has_mode:
            # Case 2: Only mode specified - all prompts with that mode
            self.logger.info(f"Only mode '{mode}' specified - finding compatible prompts")
            compatible_prompts = self.mode_to_prompts.get(mode, [])
            for prompt in compatible_prompts:
                setup = self.prompt_to_setup[prompt]
                valid_triplets.append((setup, prompt, mode))
                
        elif not has_setups and has_prompts and not has_mode:
            # Case 3: Only prompts specified - infer setups and modes
            self.logger.info("Only prompts specified - inferring setups and modes")
            for prompt in prompts:
                setup = self.prompt_to_setup.get(prompt, '')
                prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')
                if setup:
                    valid_triplets.append((setup, prompt, prompt_mode))
                else:
                    self.logger.warning(f"Prompt '{prompt}' has no associated setup")
                    
        elif not has_setups and has_prompts and has_mode:
            # Case 4: Prompts + mode specified - validate compatibility
            self.logger.info(f"Prompts + mode '{mode}' specified - validating compatibility")
            for prompt in prompts:
                setup = self.prompt_to_setup.get(prompt, '')
                prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')
                
                if not setup:
                    self.logger.warning(f"Prompt '{prompt}' has no associated setup")
                    continue
                    
                if prompt_mode != mode:
                    if not force:
                        self.logger.error(f"Prompt '{prompt}' is {prompt_mode} but {mode} was requested")
                        continue
                    else:
                        self.logger.warning(f"Force mode: using {prompt_mode} prompt '{prompt}' despite {mode} request")
                        # Use the prompt's actual mode, not the requested mode
                        valid_triplets.append((setup, prompt, prompt_mode))
                        continue
                
                valid_triplets.append((setup, prompt, mode))
                
        elif has_setups and not has_prompts and not has_mode:
            # Case 5: Only setups specified - all prompts for those setups
            self.logger.info("Only setups specified - finding all compatible prompts")
            for setup in setups:
                compatible_prompts = self.setup_to_prompts.get(setup, [])
                for prompt in compatible_prompts:
                    prompt_mode = self.prompt_to_mode[prompt]
                    valid_triplets.append((setup, prompt, prompt_mode))
                    
        elif has_setups and not has_prompts and has_mode:
            # Case 6: Setups + mode specified - prompts for setups with that mode
            self.logger.info(f"Setups + mode '{mode}' specified - finding compatible prompts")
            for setup in setups:
                compatible_prompts = self.setup_to_prompts.get(setup, [])
                mode_filtered_prompts = [p for p in compatible_prompts 
                                       if self.prompt_to_mode.get(p, 'zero-shot') == mode]
                for prompt in mode_filtered_prompts:
                    valid_triplets.append((setup, prompt, mode))
                    
        elif has_setups and has_prompts and not has_mode:
            # Case 7: Setups + prompts specified - validate compatibility
            self.logger.info("Setups + prompts specified - validating compatibility") 
            for prompt in prompts:
                expected_setup = self.prompt_to_setup.get(prompt, '')
                prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')
                
                if not expected_setup:
                    self.logger.warning(f"Prompt '{prompt}' has no associated setup")
                    continue
                    
                if expected_setup not in setups:
                    if not force:
                        self.logger.error(f"Prompt '{prompt}' expects setup '{expected_setup}' but only {setups} specified")
                        continue
                    else:
                        self.logger.warning(f"Force mode: using prompt '{prompt}' despite setup mismatch")
                        # Use the prompt's expected setup
                        valid_triplets.append((expected_setup, prompt, prompt_mode))
                        continue
                
                valid_triplets.append((expected_setup, prompt, prompt_mode))
                
        elif has_setups and has_prompts and has_mode:
            # Case 8: All specified - validate all constraints
            self.logger.info("All constraints specified - validating complete compatibility")
            for prompt in prompts:
                expected_setup = self.prompt_to_setup.get(prompt, '')
                prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')
                
                setup_valid = not expected_setup or expected_setup in setups
                mode_valid = prompt_mode == mode
                
                if not expected_setup:
                    self.logger.warning(f"Prompt '{prompt}' has no associated setup")
                    continue
                
                if not setup_valid and not mode_valid:
                    if not force:
                        self.logger.error(f"Prompt '{prompt}' incompatible: expects '{expected_setup}' (got {setups}) and is {prompt_mode} (requested {mode})")
                        continue
                    else:
                        self.logger.warning(f"Force mode: using prompt '{prompt}' despite all mismatches")
                        valid_triplets.append((expected_setup, prompt, prompt_mode))
                        continue
                        
                elif not setup_valid:
                    if not force:
                        self.logger.error(f"Prompt '{prompt}' expects setup '{expected_setup}' but only {setups} specified")
                        continue
                    else:
                        self.logger.warning(f"Force mode: using prompt '{prompt}' despite setup mismatch")
                        valid_triplets.append((expected_setup, prompt, prompt_mode))
                        continue
                        
                elif not mode_valid:
                    if not force:
                        self.logger.error(f"Prompt '{prompt}' is {prompt_mode} but {mode} was requested")
                        continue
                    else:
                        self.logger.warning(f"Force mode: using {prompt_mode} prompt '{prompt}' despite {mode} request")
                        valid_triplets.append((expected_setup, prompt, prompt_mode))
                        continue
                
                # All constraints satisfied
                valid_triplets.append((expected_setup, prompt, mode))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_triplets = []
        for triplet in valid_triplets:
            if triplet not in seen:
                seen.add(triplet)
                unique_triplets.append(triplet)
        
        self.logger.info(f"Found {len(unique_triplets)} valid (setup, prompt, mode) combinations")
        return unique_triplets
    
    def _build_all_valid_triplets(self) -> List[Tuple[str, str, str]]:
        """Build all possible valid (setup, prompt, mode) triplets"""
        triplets = []
        for setup in self.all_setups:
            for prompt in self.setup_to_prompts.get(setup, []):
                mode = self.prompt_to_mode[prompt]
                triplets.append((setup, prompt, mode))
        return triplets
    
    def _log_combination_summary(self, combinations: List[ExperimentCombination]):
        """Log a summary of the final combinations"""
        self.logger.info(f"=== FINAL COMBINATIONS: {len(combinations)} ===")
        
        if not combinations:
            return
            
        # Group by mode for summary
        mode_counts = {}
        setup_counts = {}
        model_counts = {}
        
        for combo in combinations:
            mode_counts[combo.mode] = mode_counts.get(combo.mode, 0) + 1
            setup_counts[combo.setup] = setup_counts.get(combo.setup, 0) + 1
            model_counts[combo.model] = model_counts.get(combo.model, 0) + 1
        
        self.logger.info(f"Breakdown by mode: {dict(mode_counts)}")
        self.logger.info(f"Breakdown by setup: {dict(setup_counts)}")  
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

# =============================================================================
# OPTIMIZED EXECUTION ORDER FUNCTIONS
# =============================================================================

def group_combinations_for_efficiency(combinations: List[ExperimentCombination]) -> Dict[str, Dict[str, List[ExperimentCombination]]]:
    """
    Group combinations by model first, then setup for optimal execution order.
    
    This minimizes expensive model loading/unloading operations by:
    1. Loading each model only once
    2. Processing all setups for that model
    3. Cleaning up model once before moving to next
    
    Args:
        combinations: List of experiment combinations
        
    Returns:
        dict: Nested structure {model_name: {setup_name: [combinations]}}
    """
    grouped = {}
    
    for combo in combinations:
        if combo.model not in grouped:
            grouped[combo.model] = {}
        if combo.setup not in grouped[combo.model]:
            grouped[combo.model][combo.setup] = []
        grouped[combo.model][combo.setup].append(combo)
    
    return grouped

def log_execution_plan(grouped_combinations: Dict[str, Dict[str, List[ExperimentCombination]]], models_config: dict):
    """Log the execution plan for user visibility"""
    total_experiments = sum(
        len(combos) 
        for model_groups in grouped_combinations.values() 
        for combos in model_groups.values()
    )
    
    logger.info("=" * 60)
    logger.info("EXECUTION PLAN")
    logger.info("=" * 60)
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Models to load: {len(grouped_combinations)}")
    
    for model_name, setups in grouped_combinations.items():
        model_type = models_config.get(model_name, {}).get('type', 'unknown')
        setup_count = len(setups)
        experiment_count = sum(len(combos) for combos in setups.values())
        
        logger.info(f"Model {model_name} ({model_type}): {experiment_count} experiments across {setup_count} setups")
        
        for setup_name, combos in setups.items():
            modes = list(set(combo.mode for combo in combos))
            prompts = list(set(combo.prompt for combo in combos))
            logger.info(f"   Setup {setup_name}: {len(combos)} experiments ({', '.join(modes)}) using {len(prompts)} prompts")
    
    # Calculate efficiency gain
    local_models = [model for model, config in models_config.items() 
                   if model in grouped_combinations and config.get('type') == 'local']
    
    if local_models:
        old_loads = total_experiments  # One load per experiment (old way)
        new_loads = len(local_models)  # One load per unique local model (new way)
        efficiency_gain = ((old_loads - new_loads) / old_loads) * 100 if old_loads > 0 else 0
        
        logger.info(f"Efficiency gain: {old_loads} → {new_loads} model loads ({efficiency_gain:.1f}% reduction)")
    
    logger.info("=" * 60)

# =============================================================================
# MAIN CLI FUNCTIONS
# =============================================================================

def setup_cli_logging():
    """Setup logging for CLI"""
    logger = setup_logging("main", "INFO")
    return logger

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
        logger.warning("   Consider using smaller models or API-based alternatives for better reliability")
    
    # Check for model accessibility
    try:
        from models import ModelManager
        model_manager = ModelManager()
        inaccessible_models = model_manager.get_inaccessible_models()
        
        # Filter for requested local models that are inaccessible
        requested_inaccessible = [model for model in local_models_requested if model in inaccessible_models]
        
        if requested_inaccessible:
            logger.warning(f"Some requested models may not be accessible: {requested_inaccessible}")
            logger.info("These models may be gated/restricted and require HF_ACCESS_TOKEN")
            
            if not force:
                accessible_models = [model for model in local_models_requested if model not in inaccessible_models]
                if accessible_models:
                    logger.info(f"Accessible models: {accessible_models}")
                    response = input("Continue with accessible models only? (y/N): ")
                    if response.lower() != 'y':
                        return False
                else:
                    logger.error("No accessible local models found")
                    return False
    except Exception as e:
        logger.warning(f"Could not check model accessibility: {e}")
    
    return True

def run_experiment_command(args):
    """Run experiments with execution order and skip existing functionality"""
    logger.info("Starting experiment execution with model loading...")
    
    # Load configurations
    try:
        prompts_config = Config.load_prompts_config()
        setups_config = Config.load_setups_config()
        models_config = Config.load_models_config()
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        return False
    
    # Initialize the argument resolver
    resolver = ArgumentResolver(prompts_config, setups_config, models_config)
    
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
    
    # Group combinations for optimal execution order
    grouped_combinations = group_combinations_for_efficiency(combinations)
    
    # Validate few-shot row parameter
    if args.few_shot_row is not None:
        if args.few_shot_row < 0:
            logger.error(f"Few-shot row index must be non-negative, got: {args.few_shot_row}")
            return False
        
        # Check if any few-shot experiments will be run
        has_few_shot = any(
            combo.mode == 'few-shot'
            for model_groups in grouped_combinations.values()
            for combos in model_groups.values()
            for combo in combos
        )
        if not has_few_shot:
            logger.error("Cannot specify --few-shot-row when no few-shot experiments will be run")
            return False
    
    # Validate model capability
    unique_models = list(grouped_combinations.keys())
    if not validate_local_model_capability(unique_models, models_config, args.force):
        return False
    
    # Check for existing files if --skip is enabled
    skipped_combinations = []
    remaining_combinations = {}
    
    if args.skip:
        logger.info("Checking for existing experiment files...")
        total_combinations = 0
        skipped_count = 0
        
        for model_name, model_setups in grouped_combinations.items():
            remaining_combinations[model_name] = {}
            
            for setup_name, setup_combinations in model_setups.items():
                remaining_combinations[model_name][setup_name] = []
                
                for combo in setup_combinations:
                    total_combinations += 1
                    
                    # Generate experiment name and check if file exists
                    experiment_name = Config.generate_experiment_name(
                        combo.setup, combo.model, combo.mode, combo.prompt, 
                        args.size, args.temperature, few_shot_row=args.few_shot_row
                    )
                    file_paths = Config.generate_file_paths(experiment_name)
                    inference_file = file_paths['inference']
                    
                    if os.path.exists(inference_file):
                        skipped_combinations.append((combo, experiment_name))
                        skipped_count += 1
                        logger.debug(f"Skipping existing experiment: {experiment_name}")
                    else:
                        remaining_combinations[model_name][setup_name].append(combo)
                
                # Remove empty setup groups
                if not remaining_combinations[model_name][setup_name]:
                    del remaining_combinations[model_name][setup_name]
            
            # Remove empty model groups
            if not remaining_combinations[model_name]:
                del remaining_combinations[model_name]
        
        logger.info(f"Skip analysis: {skipped_count} existing experiments found, {total_combinations - skipped_count} remaining to run")
        
        if skipped_count > 0:
            logger.info(f"Skipped experiments:")
            for i, (combo, exp_name) in enumerate(skipped_combinations[:5]):  # Show first 5
                logger.info(f"  - {exp_name}")
            if len(skipped_combinations) > 5:
                logger.info(f"  ... and {len(skipped_combinations) - 5} more")
        
        # Update grouped_combinations to only include remaining experiments
        grouped_combinations = remaining_combinations
    
    # Show execution plan
    log_execution_plan(grouped_combinations, models_config)
    
    # Calculate total experiments to run
    total_experiments = sum(
        len(combos) 
        for model_groups in grouped_combinations.values() 
        for combos in model_groups.values()
    )
    
    if total_experiments == 0:
        logger.info("No experiments to run (all already exist or no valid combinations)")
        if args.skip and skipped_combinations:
            logger.info(f"All {len(skipped_combinations)} experiments already exist and were skipped")
        return True  # Success - nothing to do
    
    # Ask for confirmation if many experiments
    if total_experiments > 10 and not args.force:
        skip_info = f" ({len(skipped_combinations)} skipped)" if args.skip else ""
        print(f"\nAbout to run {total_experiments} experiments{skip_info}. This may take a long time.")
        
        response = input("\nProceed with execution? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled by user")
            return False
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Execute experiments in order
    results = []
    success_count = 0
    failure_count = 0
    current_experiment = 0
    skipped_models = []
    
    for model_name, model_setups in grouped_combinations.items():
        model_config = models_config[model_name]
        model_experiments = sum(len(combos) for combos in model_setups.values())
        
        logger.info(f"Loading model: {model_name} ({model_experiments} experiments)")
        
        # Pre-load model if it's a local model
        model_loaded = False
        if model_config['type'] == 'local':
            try:
                if model_config.get('finetuned', False):
                    model_path = os.path.join(Config.FINETUNED_MODELS_DIR, 
                                            model_config['model_path'].split('/')[-1] + '_finetuned')
                    runner.model_manager.load_finetuned_model(model_path)
                else:
                    runner.model_manager.load_open_source_model(model_name, model_config['model_path'])
                model_loaded = True
                logger.info(f"Model {model_name} loaded successfully")
            except ValueError as e:
                # Handle authentication/accessibility errors
                if "authentication" in str(e).lower() or "not accessible" in str(e).lower():
                    logger.warning(f"Skipping model {model_name}: {e}")
                    skipped_models.append(model_name)
                    # Skip all experiments for this model
                    failure_count += model_experiments
                    continue
                else:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    failure_count += model_experiments
                    continue
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                # Skip all experiments for this model
                failure_count += model_experiments
                continue
        
        # Process all setups for this model
        for setup_name, setup_combinations in model_setups.items():
            logger.info(f"Processing setup: {setup_name} with model: {model_name} ({len(setup_combinations)} experiments)")
            
            # Pre-load and cache dataset for efficiency
            dataset_loaded = False
            try:
                # This will cache the dataset in the dataset manager
                runner.dataset_manager.load_dataset(setup_name)
                dataset_loaded = True
                logger.debug(f"Setup {setup_name} dataset loaded and cached")
            except Exception as e:
                logger.warning(f"Could not pre-load dataset for setup {setup_name}: {e}")
            
            # Run all experiments for this model+setup combination
            for combo in setup_combinations:
                current_experiment += 1
                
                try:
                    logger.info(f"Running experiment {current_experiment}/{total_experiments}: {combo}")
                    
                    experiment_config = {
                        'model': combo.model,
                        'setup': combo.setup,
                        'prompt': combo.prompt,
                        'mode': combo.mode,
                        'few_shot_row': args.few_shot_row,
                        'size': args.size,
                        'temperature': args.temperature
                    }
                    
                    # Since model is already loaded for local models, the experiment runner
                    # will use the existing loaded model instead of reloading
                    result = runner.run_experiment(experiment_config)
                    
                    if result:
                        results.append(result)
                        success_count += 1
                        logger.info(f"Completed ({current_experiment}/{total_experiments}): {result['experiment_name']}")
                    else:
                        failure_count += 1
                        logger.error(f"Failed ({current_experiment}/{total_experiments}): {combo}")
                
                except Exception as e:
                    failure_count += 1
                    logger.error(f"Error in experiment {current_experiment}/{total_experiments} ({combo}): {e}")
                    continue
        
        # Clean up model after processing all its setups
        if model_loaded:
            logger.info(f"Cleaning up model: {model_name}")
            runner.model_manager.cleanup_current_model()
    
    # Final summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT EXECUTION COMPLETE")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {failure_count}")
    if args.skip and skipped_combinations:
        logger.info(f"Skipped (already exist): {len(skipped_combinations)}")
    logger.info(f"Total: {total_experiments} experiments processed")
    
    if skipped_models:
        logger.info(f"Skipped models due to accessibility issues: {', '.join(skipped_models)}")
        logger.info("Consider adding HF_ACCESS_TOKEN to .env file for restricted models")
    
    if results:
        logger.info("Generated experiment files:")
        for result in results[:10]:  # Show first 10
            logger.info(f"  - {result['output_file']}")
        if len(results) > 10:
            logger.info(f"  ... and {len(results) - 10} more files")
    
    # Show efficiency summary for local models
    local_model_count = sum(1 for model in unique_models 
                           if models_config.get(model, {}).get('type') == 'local')
    if local_model_count > 0:
        logger.info(f"Efficiency: Loaded {local_model_count} local models instead of {total_experiments} (saved {total_experiments - local_model_count} loading operations)")
    
    logger.info("=" * 60)
    
    return success_count > 0 or (args.skip and len(skipped_combinations) > 0)

# =============================================================================
# PROMPT VALIDATION AND INSPECTION COMMANDS
# =============================================================================

def validate_prompt_command(args):
    """Validate prompt field paths and configurations"""
    logger.info("Starting prompt validation...")
    
    # Load configurations
    try:
        prompts_config = Config.load_prompts_config()
        setups_config = Config.load_setups_config()
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        return False
    
    # Initialize managers
    prompt_manager = PromptManager()
    dataset_manager = DatasetManager()
    
    if args.prompt:
        # Validate specific prompt(s)
        prompt_names = args.prompt.split(',')
        
        for prompt_name in prompt_names:
            prompt_name = prompt_name.strip()
            
            if prompt_name not in prompts_config:
                logger.error(f"Unknown prompt: {prompt_name}")
                continue
            
            try:
                # Get prompt requirements
                requirements = prompt_manager.get_prompt_field_requirements(prompt_name)
                setup_name = requirements['setup_name']
                
                print(f"\n=== VALIDATING PROMPT: {prompt_name} ===")
                print(f"Setup: {setup_name}")
                print(f"Mode: {requirements['mode']}")
                print(f"Template placeholders: {requirements['placeholder_count']}")
                print(f"Question field paths: {requirements['question_field_paths']}")
                print(f"Answer field path: {requirements['answer_field_path']}")
                
                # Load sample data to test field paths
                dataset = dataset_manager.load_dataset(setup_name)
                if dataset is None:
                    logger.error(f"Could not load dataset for setup: {setup_name}")
                    continue
                
                if len(dataset) == 0:
                    logger.error(f"Dataset is empty for setup: {setup_name}")
                    continue
                
                # Test field path resolution on sample data
                sample_row = dataset.iloc[0]
                validation_result = prompt_manager.validate_prompt_field_paths(
                    prompt_name, setup_name, sample_row
                )
                
                print(f"\nField Path Validation:")
                print(f"  Valid: {validation_result['valid']}")
                
                if validation_result['errors']:
                    print(f"  Errors:")
                    for error in validation_result['errors']:
                        print(f"    - {error}")
                
                if validation_result['warnings']:
                    print(f"  Warnings:")
                    for warning in validation_result['warnings']:
                        print(f"    - {warning}")
                
                print(f"  Sample Field Values:")
                for field_path, value in validation_result['field_values'].items():
                    print(f"    {field_path}: {repr(value)}")
                
            except Exception as e:
                logger.error(f"Error validating prompt {prompt_name}: {e}")
                continue
    
    else:
        # Validate all prompts
        print(f"\n=== VALIDATING ALL PROMPTS ({len(prompts_config)}) ===")
        
        valid_count = 0
        invalid_count = 0
        
        for prompt_name in prompts_config.keys():
            try:
                requirements = prompt_manager.get_prompt_field_requirements(prompt_name)
                setup_name = requirements['setup_name']
                
                # Quick validation without loading full dataset
                if setup_name not in setups_config:
                    print(f"❌ {prompt_name}: Invalid setup '{setup_name}'")
                    invalid_count += 1
                    continue
                
                # Check if dataset exists
                if not dataset_manager.is_dataset_downloaded(setup_name):
                    print(f"⚠️  {prompt_name}: Dataset not downloaded for setup '{setup_name}'")
                    continue
                
                print(f"✅ {prompt_name}: Valid (setup: {setup_name}, mode: {requirements['mode']})")
                valid_count += 1
                
            except Exception as e:
                print(f"❌ {prompt_name}: Error - {str(e)[:100]}...")
                invalid_count += 1
        
        print(f"\nValidation Summary:")
        print(f"  Valid: {valid_count}")
        print(f"  Invalid: {invalid_count}")
        print(f"  Total: {len(prompts_config)}")
    
    return True

def list_prompts_command(args):
    """List available prompts with details"""
    logger.info("Listing available prompts...")
    
    try:
        prompt_manager = PromptManager()
        
        if args.mode:
            prompts = prompt_manager.list_prompts(mode=args.mode)
            print(f"\n=== PROMPTS FOR MODE: {args.mode.upper()} ===")
        else:
            prompts = prompt_manager.list_prompts()
            print(f"\n=== ALL AVAILABLE PROMPTS ({len(prompts)}) ===")
        
        if not prompts:
            print("No prompts found.")
            return True
        
        for name, description in prompts.items():
            print(f"\n{name}:")
            print(f"  {description}")
            
            if args.details:
                try:
                    requirements = prompt_manager.get_prompt_field_requirements(name)
                    print(f"  Field paths: {requirements['question_field_paths']}")
                    print(f"  Answer path: {requirements['answer_field_path']}")
                    print(f"  Template placeholders: {requirements['placeholder_count']}")
                except Exception as e:
                    print(f"  Error getting details: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        return False

def inspect_prompt_command(args):
    """Inspect a specific prompt's configuration and requirements"""
    logger.info(f"Inspecting prompt: {args.prompt}")
    
    try:
        prompt_manager = PromptManager()
        
        if args.prompt not in prompt_manager.prompts_config:
            logger.error(f"Unknown prompt: {args.prompt}")
            logger.info(f"Available prompts: {list(prompt_manager.prompts_config.keys())}")
            return False
        
        # Get detailed requirements
        requirements = prompt_manager.get_prompt_field_requirements(args.prompt)
        
        print(f"\n=== PROMPT INSPECTION: {args.prompt} ===")
        print(f"Setup: {requirements['setup_name']}")
        print(f"Mode: {requirements['mode']}")
        print(f"Description: {prompt_manager.prompts_config[args.prompt].get('description', 'No description')}")
        
        print(f"\nTemplate:")
        print(f"  Placeholders: {requirements['placeholder_count']}")
        print(f"  Template: {requirements['template']}")
        
        if requirements['few_shot_template']:
            print(f"  Few-shot template: {requirements['few_shot_template']}")
        
        print(f"\nField Requirements:")
        print(f"  Question field paths: {requirements['question_field_paths']}")
        print(f"  Answer field path: {requirements['answer_field_path']}")
        
        # Test field path resolution if dataset is available
        dataset_manager = DatasetManager()
        setup_name = requirements['setup_name']
        
        if dataset_manager.is_dataset_downloaded(setup_name):
            print(f"\nTesting field path resolution...")
            
            dataset = dataset_manager.load_dataset(setup_name)
            if dataset is not None and not dataset.empty:
                sample_row = dataset.iloc[0]
                validation_result = prompt_manager.validate_prompt_field_paths(
                    args.prompt, setup_name, sample_row
                )
                
                print(f"  Validation: {'✅ Valid' if validation_result['valid'] else '❌ Invalid'}")
                
                if validation_result['errors']:
                    print(f"  Errors:")
                    for error in validation_result['errors']:
                        print(f"    - {error}")
                
                print(f"  Sample values:")
                for field_path, result in validation_result['field_path_results'].items():
                    if 'error' in result:
                        print(f"    {field_path}: ERROR - {result['error']}")
                    else:
                        status = "✅" if result['resolved'] else "❌"
                        print(f"    {field_path}: {status} {result.get('value_type', 'None')} - {result.get('value_preview', 'None')}")
        else:
            print(f"  Dataset not downloaded - cannot test field paths")
            print(f"  Run: python main.py download-datasets --setup {setup_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error inspecting prompt: {e}")
        return False

# =============================================================================
# OTHER COMMAND HANDLERS
# =============================================================================

def evaluate_command(args):
    """Evaluate experiment results"""
    logger.info("Starting evaluation...")
    
    # Initialize evaluation runner
    evaluator = EvaluationRunner()
    
    try:
        if args.experiment:
            # Evaluate specific experiment(s)
            experiments_to_evaluate = args.experiment.split(',')
            results = []
            
            for experiment_name in experiments_to_evaluate:
                result = evaluator.evaluate_experiment(experiment_name.strip())
                if result:
                    results.append(result)
                    logger.info(f"Evaluated: {experiment_name}")
                else:
                    logger.error(f"Failed to evaluate: {experiment_name}")
            
        else:
            # Evaluate all experiments
            results = evaluator.evaluate_all_experiments()
        
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
    
    # Initialize plotting runner
    plotter = PlottingRunner()
    
    try:
        if args.experiment:
            # Plot specific experiment(s)
            experiments_to_plot = args.experiment.split(',')
            
            if args.compare:
                # Generate comparison plots
                results = plotter.create_comparison_plots(experiments_to_plot)
                if results:
                    logger.info(f"Generated comparison plots: {results}")
                else:
                    logger.error("Failed to generate comparison plots")
            else:
                # Generate individual plots
                success_count = 0
                for experiment_name in experiments_to_plot:
                    result = plotter.create_individual_plot(experiment_name.strip())
                    if result:
                        success_count += 1
                        logger.info(f"Generated plot for: {experiment_name}")
                    else:
                        logger.error(f"Failed to generate plot for: {experiment_name}")
                
                return success_count > 0
        else:
            # Plot all experiments
            results = plotter.create_all_plots()
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
    """Show a populated prompt template with enhanced validation"""
    logger.info("Showing populated prompt template with validation...")
    
    # Load configurations
    try:
        prompts_config = Config.load_prompts_config()
        setups_config = Config.load_setups_config()
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
        # Get prompt configuration and validate
        requirements = prompt_manager.get_prompt_field_requirements(args.prompt)
        compatible_setup = requirements['setup_name']
        prompt_mode = requirements['mode']
        
        print(f"\n=== PROMPT VALIDATION ===")
        print(f"Prompt: {args.prompt}")
        print(f"Setup: {compatible_setup}")
        print(f"Mode: {prompt_mode}")
        
        # Load the compatible dataset
        logger.info(f"Loading dataset for setup: {compatible_setup}")
        dataset = dataset_manager.load_dataset(compatible_setup)
        if dataset is None:
            logger.error(f"Failed to load dataset for setup: {compatible_setup}")
            return False
        
        # Determine which row to use
        if args.row is not None:
            if args.row < 0 or args.row >= len(dataset):
                logger.error(f"Row index {args.row} out of bounds for dataset with {len(dataset)} rows")
                return False
            row = dataset.iloc[args.row]
            row_index = args.row
            logger.info(f"Using specified row {args.row}")
        else:
            # Use random row
            import random
            random.seed(Config.RANDOM_SEED)
            row_index = random.randint(0, len(dataset) - 1)
            row = dataset.iloc[row_index]
            logger.info(f"Using random row {row_index}")
        
        # Validate field paths with actual data
        validation_result = prompt_manager.validate_prompt_field_paths(
            args.prompt, compatible_setup, row
        )
        
        print(f"\nField Path Validation: {'✅ Valid' if validation_result['valid'] else '❌ Invalid'}")
        
        if validation_result['errors']:
            print(f"Errors:")
            for error in validation_result['errors']:
                print(f"  - {error}")
            print("\nCannot proceed with invalid field paths.")
            return False
        
        if validation_result['warnings']:
            print(f"Warnings:")
            for warning in validation_result['warnings']:
                print(f"  - {warning}")
        
        # Show field path resolution results
        print(f"\nField Path Resolution:")
        for field_path, result in validation_result['field_path_results'].items():
            status = "✅" if result.get('resolved', False) else "❌"
            value_preview = result.get('value_preview', 'None')
            print(f"  {field_path}: {status} {value_preview}")
        
        # Prepare the populated prompt
        populated_prompt = prompt_manager.prepare_prompt_for_row(
            prompt_name=args.prompt,
            row=row,
            setup_name=compatible_setup,
            mode=prompt_mode,
            dataset=dataset,
            few_shot_row=args.row if prompt_mode == 'few-shot' and args.row is not None else None
        )
        
        # Display results
        print("\n" + "=" * 80)
        print(f"POPULATED PROMPT TEMPLATE")
        print("=" * 80)
        print(f"Prompt: {args.prompt}")
        print(f"Mode: {prompt_mode}")
        print(f"Compatible Setup: {compatible_setup}")
        print(f"Row Used: {row_index}")
        print(f"Dataset Size: {len(dataset)} rows")
        print("=" * 80)
        print(f"\nPOPULATED TEMPLATE:")
        print("-" * 40)
        print(populated_prompt)
        print("-" * 40)
        
        # Show the original field values
        print(f"\nORIGINAL FIELD VALUES:")
        print("-" * 40)
        for field_path, value in validation_result['field_values'].items():
            print(f"{field_path}: {repr(value)}")
        
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
    all_setups = list(manager.setups_config.keys())
    
    # Check what's already downloaded
    already_downloaded = manager.list_downloaded_datasets()
    if already_downloaded:
        logger.info(f"Already downloaded: {', '.join(already_downloaded)}")
    
    # Parse setup arguments properly
    if not args.setup or args.setup.lower() == 'all':
        setups_to_download = all_setups
    else:
        if ',' in args.setup:
            setups_to_download = [item.strip() for item in args.setup.split(',')]
        else:
            setups_to_download = args.setup.split()
    
    success_count = 0
    failure_count = 0
    
    if not setups_to_download:
        logger.warning("No setups specified for download.")
        return False
        
    for setup_name in setups_to_download:
        if setup_name not in manager.setups_config:
            logger.error(f"Unknown setup: {setup_name}")
            failure_count += 1
            continue
        
        # Skip if already downloaded
        if setup_name in already_downloaded:
            logger.info(f"Setup '{setup_name}' already downloaded, skipping")
            continue
            
        try:
            if manager.download_dataset(setup_name):
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            logger.error(f"Error downloading dataset for setup {setup_name}: {e}")
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

# =============================================================================
# DATASET MANAGEMENT COMMANDS
# =============================================================================

def dataset_command(args):
    """Handle dataset management subcommands"""
    if args.dataset_command == 'list':
        return dataset_list_command(args)
    elif args.dataset_command == 'status':
        return dataset_status_command(args)
    elif args.dataset_command == 'validate':
        return dataset_validate_command(args)
    else:
        print("Unknown dataset subcommand. Use 'python main.py dataset -h' for help.")
        return False

def dataset_list_command(args):
    """List all configured datasets with detailed information"""
    logger.info("Listing all configured datasets...")
    
    try:
        manager = DatasetManager()
        available_setups = manager.get_available_setups()
        
        print(f"\n=== CONFIGURED DATASETS ({len(available_setups)}) ===")
        
        for setup_name, info in available_setups.items():
            print(f"\nDataset: {setup_name}")
            print(f"   Description: {info.get('description', 'No description')}")
            print(f"   File Type: {info.get('file_type', 'Unknown')}")
            print(f"   Downloaded: {'Yes' if info.get('is_downloaded') else 'No'}")
            
            if info.get('is_downloaded'):
                if info.get('is_loaded'):
                    print(f"   Rows: {info.get('num_rows', 'Unknown')}")
                    print(f"   Columns: {info.get('num_columns', 'Unknown')}")
                    print(f"   Question Fields: {info.get('question_fields', [])}")
                    print(f"   Answer Field: {info.get('answer_field', 'None')}")
                
                # Show pruning configuration if present
                prune_config = info.get('prune_config', {})
                if prune_config:
                    print(f"   Pruning Rules: {len(prune_config)} configured")
        
        return True
        
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return False

def dataset_status_command(args):
    """Show dataset download and validation status"""
    logger.info("Checking dataset status...")
    
    try:
        manager = DatasetManager()
        downloaded = manager.list_downloaded_datasets()
        all_setups = list(manager.setups_config.keys())
        
        print(f"\n=== DATASET STATUS ===")
        print(f"Total Configured: {len(all_setups)}")
        print(f"Downloaded: {len(downloaded)}")
        print(f"Missing: {len(all_setups) - len(downloaded)}")
        
        if downloaded:
            print(f"\nDownloaded Datasets:")
            for setup_name in downloaded:
                print(f"   ✓ {setup_name}")
        
        missing = set(all_setups) - set(downloaded)
        if missing:
            print(f"\nMissing Datasets:")
            for setup_name in missing:
                print(f"   ✗ {setup_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking dataset status: {e}")
        return False

def dataset_validate_command(args):
    """Validate dataset fields for all downloaded datasets"""
    logger.info("Validating dataset fields...")
    
    try:
        manager = DatasetManager()
        downloaded = manager.list_downloaded_datasets()
        
        if not downloaded:
            print("No datasets downloaded. Use 'python main.py download-datasets --setup all' to download.")
            return True
        
        print(f"\n=== DATASET FIELD VALIDATION ===")
        
        valid_count = 0
        invalid_count = 0
        
        for setup_name in downloaded:
            try:
                is_valid = manager.validate_dataset_fields(setup_name)
                status = "✓ Valid" if is_valid else "✗ Invalid"
                print(f"   {setup_name}: {status}")
                
                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    
            except Exception as e:
                print(f"   {setup_name}: ✗ Error - {e}")
                invalid_count += 1
        
        print(f"\nValidation Summary:")
        print(f"   Valid: {valid_count}")
        print(f"   Invalid: {invalid_count}")
        print(f"   Total Checked: {len(downloaded)}")
        
        return invalid_count == 0
        
    except Exception as e:
        logger.error(f"Error validating datasets: {e}")
        return False

# =============================================================================
# ENHANCED SYSTEM INFORMATION COMMANDS
# =============================================================================

def list_available_options():
    """List available models, setups, and prompts with enhanced dataset information"""
    try:
        prompts_config = Config.load_prompts_config()
        setups_config = Config.load_setups_config()
        models_config = Config.load_models_config()
        
        # Initialize managers for enhanced info
        dataset_manager = DatasetManager()
        prompt_manager = PromptManager()
        available_setups = dataset_manager.get_available_setups()
        
        print("\n=== AVAILABLE OPTIONS ===")
        
        print(f"\nModels ({len(models_config)}):")
        for model_name, config in models_config.items():
            model_type = config['type']
            finetuned = " (finetuned)" if config.get('finetuned', False) else ""
            print(f"  - {model_name}: {config['description']}{finetuned} [{model_type}]")
        
        print(f"\nSetups ({len(setups_config)}):")
        for setup_name, config in setups_config.items():
            setup_info = available_setups.get(setup_name, {})
            downloaded_status = "Downloaded" if setup_info.get('is_downloaded') else "Not Downloaded"
            file_type = setup_info.get('file_type', 'Unknown')
            print(f"  - {setup_name}: {config['description']} [{downloaded_status}, {file_type}]")
        
        print(f"\nPrompts ({len(prompts_config)}):")
        
        # Group prompts by mode for better organization
        zero_shot_prompts = prompt_manager.list_prompts(mode='zero-shot')
        few_shot_prompts = prompt_manager.list_prompts(mode='few-shot')
        
        if zero_shot_prompts:
            print("  Zero-shot:")
            for name, description in zero_shot_prompts.items():
                try:
                    requirements = prompt_manager.get_prompt_field_requirements(name)
                    field_count = len(requirements['question_field_paths'])
                    print(f"    - {name}: {prompts_config[name]['description']} [{field_count} fields]")
                except:
                    print(f"    - {name}: {prompts_config[name]['description']}")
        
        if few_shot_prompts:
            print("  Few-shot:")
            for name, description in few_shot_prompts.items():
                try:
                    requirements = prompt_manager.get_prompt_field_requirements(name)
                    field_count = len(requirements['question_field_paths'])
                    print(f"    - {name}: {prompts_config[name]['description']} [{field_count} fields]")
                except:
                    print(f"    - {name}: {prompts_config[name]['description']}")
        
        print(f"\nFor detailed prompt information, use:")
        print(f"  python main.py list-prompts --details")
        print(f"  python main.py inspect-prompt --prompt <prompt_name>")
        
    except Exception as e:
        print(f"Error loading configurations: {e}")

def check_system_command(args):
    """Check system status with enhanced dataset validation"""
    logger.info("Checking system status...")
    
    # Use the enhanced system status display
    gpu_status, memory_status = print_system_status()
    
    # Check configuration files
    config_status = Config.validate_configuration_files()
    
    print("Configuration Files:")
    for config_type, exists in config_status.items():
        status = "✓" if exists else "✗"
        print(f"   {config_type}.json: {status}")
    
    # Check directories
    print("\nDirectories:")
    try:
        created_dirs = Config.create_directories()
        print(f"   Created/verified {len(created_dirs)} directories")
    except Exception as e:
        print(f"   Error creating directories: {e}")
    
    # Check API keys including HF token
    print("\nAPI Keys:")
    print(f"   OpenAI: {'✓' if Config.OPENAI_API_KEY else '✗'}")
    print(f"   Google GenAI: {'✓' if Config.GENAI_API_KEY else '✗'}")
    print(f"   Anthropic: {'✓' if Config.ANTHROPIC_API_KEY else '✗'}")
    
    # Show detailed HF token status
    from utils import validate_hf_token
    hf_status = validate_hf_token()
    if hf_status['token_valid']:
        user_name = hf_status['user_info'].get('name', 'Unknown')
        print(f"   Hugging Face: ✓ (user: {user_name})")
    elif hf_status['token_available']:
        print(f"   Hugging Face: ✗ (invalid token)")
    else:
        print(f"   Hugging Face: ✗ (no token)")
    
    # Enhanced dataset status with field validation
    print("\nDataset Status:")
    try:
        dataset_manager = DatasetManager()
        available_setups = dataset_manager.get_available_setups()
        downloaded_datasets = dataset_manager.list_downloaded_datasets()
        
        total_setups = len(available_setups)
        downloaded_count = len(downloaded_datasets)
        
        print(f"   Total Configured: {total_setups}")
        print(f"   Downloaded: {downloaded_count}")
        print(f"   Missing: {total_setups - downloaded_count}")
        
        # Validate fields for downloaded datasets
        if downloaded_datasets:
            print("\nDataset Field Validation:")
            valid_count = 0
            for setup_name in downloaded_datasets:
                try:
                    is_valid = dataset_manager.validate_dataset_fields(setup_name)
                    status = "✓" if is_valid else "✗"
                    print(f"   {setup_name}: {status} {'Valid fields' if is_valid else 'Invalid fields'}")
                    if is_valid:
                        valid_count += 1
                except Exception as e:
                    print(f"   {setup_name}: ✗ Validation error - {str(e)[:50]}...")
            
            print(f"   Valid: {valid_count}/{len(downloaded_datasets)}")
        else:
            print("   No datasets downloaded for validation")
        
    except Exception as e:
        print(f"   Error checking dataset status: {e}")
    
    # Add prompt validation section
    print("\nPrompt Configuration:")
    try:
        prompt_manager = PromptManager()
        prompts_config = Config.load_prompts_config()
        valid_prompts = 0
        invalid_prompts = 0
        
        for prompt_name in prompts_config.keys():
            try:
                requirements = prompt_manager.get_prompt_field_requirements(prompt_name)
                if requirements.get('error'):
                    invalid_prompts += 1
                else:
                    valid_prompts += 1
            except:
                invalid_prompts += 1
        
        print(f"   Total Prompts: {len(prompts_config)}")
        print(f"   Valid: {valid_prompts}")
        print(f"   Invalid: {invalid_prompts}")
        
        if invalid_prompts > 0:
            print(f"   Run 'python main.py validate-prompts' for details")
        
    except Exception as e:
        print(f"   Error checking prompts: {e}")
    
    # Check for model accessibility issues
    print("\nModel Accessibility:")
    try:
        from models import ModelManager
        model_manager = ModelManager()
        inaccessible_models = model_manager.get_inaccessible_models()
        
        if inaccessible_models:
            print(f"   ⚠️  Some models may not be accessible: {len(inaccessible_models)}")
            print(f"   Potentially restricted: {', '.join(inaccessible_models[:3])}")
            if len(inaccessible_models) > 3:
                print(f"   ... and {len(inaccessible_models) - 3} more")
            print("   Consider adding HF_ACCESS_TOKEN to .env file")
        else:
            print("   ✓ All configured models appear accessible")
    except Exception as e:
        print(f"   ⚠️  Could not check model accessibility: {e}")
    
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
    
    if not hf_status['token_valid'] and Config.HF_ACCESS_TOKEN:
        print("   - Check HF_ACCESS_TOKEN validity")
    elif not hf_status['token_available']:
        print("   - Add HF_ACCESS_TOKEN to .env file for gated/restricted models")
    
    if downloaded_count < total_setups:
        print("   - Download missing datasets with: python main.py download-datasets --setup all")
    
    return all(config_status.values())

def list_commands_command():
    """List all available commands and their arguments with proper formatting"""
    print("\n=== AVAILABLE COMMANDS ===")
    
    print("\n1. run-experiment")
    print("   Description: Run inference experiments with intelligent argument resolution")
    print("   Arguments:")
    print("     --model MODELS           Model(s) to use (space-separated: 'gpt-4o-mini claude-3.5-sonnet' or 'all')")
    print("     --setup SETUPS           Setup(s) to use (space-separated: 'gmeg qald' or 'all')")
    print("     --prompt PROMPTS         Prompt(s) to use (space-separated: 'gmeg_v1 qald_v1' or 'all')")
    print("     --mode MODE              Prompting mode: zero-shot or few-shot")
    print("     --few-shot-row ROW       Specific row number for few-shot example (0-based, defaults to random)")
    print("     --size SIZE              Sample size (default: 50)")
    print("     --temperature TEMP       Temperature for generation (default: 0.1)")
    print("     --skip                   Skip experiments where response files already exist")
    print("     --force                  Force run even with validation errors")
    
    print("\n   Intelligent Argument Resolution:")
    print("     • Specify any combination of arguments - system handles compatibility automatically")
    print("     • Unspecified arguments default to all compatible options")
    print("     • Setup+prompt combinations are validated for compatibility")
    print("     • Mode filtering only uses prompts that support the specified mode")
    print("     • Multiple values supported: space-separated or comma-separated")
    
    print("\n2. evaluate")
    print("   Description: Evaluate experiment results using various metrics")
    print("   Arguments:")
    print("     --experiment NAMES       Specific experiment(s) to evaluate (comma-separated)")
    
    print("\n3. plot")
    print("   Description: Generate plots and visualizations from evaluation results")
    print("   Arguments:")
    print("     --experiment NAMES       Specific experiment(s) to plot (comma-separated)")
    print("     --compare               Generate comparison plots instead of individual")
    
    print("\n4. show-prompt")
    print("   Description: Display a populated prompt template with field path validation")
    print("   Arguments:")
    print("     --prompt PROMPT_NAME     Name of prompt template to show (required)")
    print("     --row ROW_NUMBER         Specific row number to use from dataset (0-based, optional)")
    print("                              If not specified, uses a random row")
    
    print("\n5. validate-prompts")
    print("   Description: Validate prompt configurations and field path resolution")
    print("   Arguments:")
    print("     --prompt PROMPTS         Specific prompt(s) to validate (comma-separated, optional)")
    
    print("\n6. list-prompts")
    print("   Description: List available prompts with detailed information")
    print("   Arguments:")
    print("     --mode MODE              Filter by prompting mode (zero-shot or few-shot)")
    print("     --details                Show detailed field path and template information")
    
    print("\n7. inspect-prompt")
    print("   Description: Inspect a specific prompt's configuration and test field paths")
    print("   Arguments:")
    print("     --prompt PROMPT_NAME     Name of prompt to inspect (required)")
    
    print("\n8. download-datasets")
    print("   Description: Download specified datasets from configured sources")
    print("   Arguments:")
    print("     --setup SETUPS           Setup(s) to download (space-separated or 'all')")
    
    print("\n9. dataset")
    print("   Description: Dataset management and information commands")
    print("   Subcommands:")
    print("     list                     List all configured datasets with details")
    print("     status                   Show download status summary")
    print("     validate                 Validate field structure of downloaded datasets")
    
    print("\n10. cleanup")
    print("   Description: Clean up system files and directories")
    print("   Arguments:")
    print("     --target TARGETS         What to clean: datasets, logs, results, cache, finetuned, all")
    print("     --dry-run               Show what would be cleaned without actually cleaning")
    
    print("\n11. list-options")
    print("   Description: List available models, setups, prompts with download status")
    print("   Arguments: None")
    
    print("\n12. list-commands / help / show-commands")
    print("   Description: Show this help message with all commands and their arguments")
    print("   Arguments: None")
    
    print("\n13. status")
    print("   Description: Check system status including datasets, configurations, and API keys")
    print("   Arguments: None")
    
    print("\n=== EXAMPLE USAGE ===")
    print("# Skip existing experiments when adding new ones")
    print("python main.py run-experiment --model 'gpt-4o-mini claude-3.5-sonnet' --skip")
    print("")
    print("# Resume interrupted experiment run")
    print("python main.py run-experiment --setup 'gmeg qald' --mode few-shot --skip")
    print("")
    print("# Validate all prompt configurations")
    print("python main.py validate-prompts")
    print("")
    print("# Validate specific prompts")
    print("python main.py validate-prompts --prompt 'gmeg_explaination,cose_explanation'")
    print("")
    print("# List all prompts with details")
    print("python main.py list-prompts --details")
    print("")
    print("# List only few-shot prompts")
    print("python main.py list-prompts --mode few-shot")
    print("")
    print("# Inspect a specific prompt configuration")
    print("python main.py inspect-prompt --prompt gmeg_explaination")
    print("")
    print("# Check dataset status and download missing ones")
    print("python main.py dataset status")
    print("python main.py download-datasets --setup all")
    print("")
    print("# Validate dataset fields after download")
    print("python main.py dataset validate")
    print("")
    print("# List all available datasets with detailed information")
    print("python main.py dataset list")
    print("")
    print("# Multiple models with space separation")
    print("python main.py run-experiment --model 'gpt-4o-mini claude-3.5-sonnet'")
    print("")
    print("# Multiple setups and automatic prompt selection")
    print("python main.py run-experiment --setup 'gmeg qald' --mode few-shot")
    print("")
    print("# Show populated prompt template with validation")
    print("python main.py show-prompt --prompt gmeg_explaination --row 42")
    print("")
    print("# Show populated prompt template with random data")
    print("python main.py show-prompt --prompt gmeg_explaination")
    print("")
    print("# Mixed arguments with compatibility validation")
    print("python main.py run-experiment --model 'gpt-4o-mini' --setup gmeg --mode zero-shot")
    print("")
    print("# Force incompatible combinations")
    print("python main.py run-experiment --prompt 'gmeg_explaination qald_v1_basic' --force")

def main():
    """Main CLI entry point"""
    # Initialize system
    try:
        initialize_system()
        global logger
        logger = setup_cli_logging()
        logger.info("XAI Explanation Evaluation System - Comprehensive CLI with JSON/JSONL Support")
    except Exception as e:
        print(f"System initialization failed: {e}")
        return 1
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description="XAI Explanation Evaluation System - Multi-Input Argument Handling with JSON/JSONL Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use 'python main.py list-commands' to see all available commands and examples."
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Experiment command
    exp_parser = subparsers.add_parser('run-experiment', help='Run inference experiments')
    exp_parser.add_argument('--model', type=str,
                                help='Model(s) to use (space-separated: "gpt-4o-mini claude-3.5-sonnet" or "all")')
    exp_parser.add_argument('--setup', type=str,
                                help='Setup(s) to use (space-separated: "gmeg qald" or "all")')
    exp_parser.add_argument('--prompt', type=str,
                                help='Prompt(s) to use (space-separated: "gmeg_v1 qald_v1" or "all")')
    exp_parser.add_argument('--mode', type=str, choices=['zero-shot', 'few-shot'],
                                help='Prompting mode: zero-shot or few-shot')
    exp_parser.add_argument('--few-shot-row', type=int,
                                help='Specific row number to use for few-shot example (0-based indexing, defaults to random)')
    exp_parser.add_argument('--size', type=int, default=Config.DEFAULT_SAMPLE_SIZE,
                                help=f'Sample size (default: {Config.DEFAULT_SAMPLE_SIZE})')
    exp_parser.add_argument('--temperature', type=float, default=Config.DEFAULT_TEMPERATURE,
                                help=f'Temperature for generation (default: {Config.DEFAULT_TEMPERATURE})')
    exp_parser.add_argument('--force', action='store_true',
                                help='Force run even with validation errors')
    exp_parser.add_argument('--skip', action='store_true',
                                help='Skip experiments where response files already exist')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate experiment results')
    eval_parser.add_argument('--experiment', type=str,
                           help='Specific experiment(s) to evaluate (comma-separated)')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Generate plots from evaluations')
    plot_parser.add_argument('--experiment', type=str,
                           help='Specific experiment(s) to plot (comma-separated)')
    plot_parser.add_argument('--compare', action='store_true',
                           help='Generate comparison plots instead of individual plots')
    
    # Show prompt command
    show_prompt_parser = subparsers.add_parser('show-prompt', help='Display populated prompt template')
    show_prompt_parser.add_argument('--prompt', type=str, required=True,
                                   help='Name of prompt template to show')
    show_prompt_parser.add_argument('--row', type=int,
                                   help='Specific row number to use from dataset (0-based, uses random if not specified)')
    
    # NEW: Prompt validation commands
    validate_prompt_parser = subparsers.add_parser('validate-prompts', 
                                                   help='Validate prompt configurations and field paths')
    validate_prompt_parser.add_argument('--prompt', type=str,
                                       help='Specific prompt(s) to validate (comma-separated)')
    
    list_prompts_parser = subparsers.add_parser('list-prompts',
                                               help='List available prompts with details') 
    list_prompts_parser.add_argument('--mode', type=str, choices=['zero-shot', 'few-shot'],
                                    help='Filter by prompting mode')
    list_prompts_parser.add_argument('--details', action='store_true',
                                    help='Show detailed field path information')
    
    inspect_prompt_parser = subparsers.add_parser('inspect-prompt',
                                                 help='Inspect a specific prompt configuration')
    inspect_prompt_parser.add_argument('--prompt', type=str, required=True,
                                      help='Name of prompt to inspect')
    
    # Download datasets command
    download_parser = subparsers.add_parser('download-datasets', help='Download specified datasets')
    download_parser.add_argument('--setup', type=str, default='all',
                                 help='Setup(s) to download (space-separated, or "all" for all setups)')
    
    # Dataset management commands
    dataset_parser = subparsers.add_parser('dataset', help='Dataset management and information')
    dataset_subparsers = dataset_parser.add_subparsers(dest='dataset_command', help='Dataset subcommands')
    
    dataset_subparsers.add_parser('list', help='List all configured datasets with detailed information')
    dataset_subparsers.add_parser('status', help='Show dataset download status summary')
    dataset_subparsers.add_parser('validate', help='Validate field structure of downloaded datasets')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up system files and directories')
    cleanup_parser.add_argument('--target', type=str, default='all',
                               choices=['datasets', 'logs', 'results', 'cache', 'finetuned', 'all'],
                               help='What to clean: datasets, logs, results, cache, finetuned, all')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                               help='Show what would be cleaned without actually cleaning')
    
    # Utility commands
    list_parser = subparsers.add_parser('list-options', help='List available models, setups, and prompts')
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
        if args.command == 'run-experiment':
            success = run_experiment_command(args)
        elif args.command == 'evaluate':
            success = evaluate_command(args)
        elif args.command == 'plot':
            success = plot_command(args)
        elif args.command == 'show-prompt':
            success = show_prompt_command(args)
        elif args.command == 'validate-prompts':
            success = validate_prompt_command(args)
        elif args.command == 'list-prompts':
            success = list_prompts_command(args)
        elif args.command == 'inspect-prompt':
            success = inspect_prompt_command(args)
        elif args.command == 'download-datasets':
            success = download_datasets_command(args)
        elif args.command == 'dataset':
            success = dataset_command(args)
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