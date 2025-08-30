#!/usr/bin/env python3
"""
XAI Explanation Evaluation System - Comprehensive CLI

This script provides a command-line interface for running experiments to evaluate
Large Language Models' alignment with user study results in XAI explanation evaluation.
Handles all possible argument permutations intelligently.
"""

import argparse
import sys
import os
import shutil
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass
from enum import Enum

# Load environment variables first
try:
    from dotenv import load_dotenv
    from pathlib import Path
    
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print("⚠️  No .env file found. Create one from .env.template with your API keys")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

from config import Config
from utils import setup_logging, initialize_system, print_system_status, validate_gpu_requirements_for_command
from experiment_runner import ExperimentRunner
from evaluator import EvaluationRunner
from plotter import PlottingRunner
from dataset_manager import DatasetManager

# =============================================================================
# COMPREHENSIVE ARGUMENT PERMUTATION SYSTEM
# =============================================================================

class ArgumentType(Enum):
    """Enumeration of argument types for systematic handling"""
    MODEL = "model"
    DATASET = "dataset" 
    PROMPT = "prompt"
    MODE = "mode"

@dataclass
class ArgumentSpec:
    """Specification of which arguments were provided by user"""
    model: bool = False
    dataset: bool = False
    prompt: bool = False
    mode: bool = False
    
    def __post_init__(self):
        """Calculate permutation signature for systematic handling"""
        self.signature = (self.model, self.dataset, self.prompt, self.mode)
        self.count = sum(self.signature)
        
    def get_permutation_name(self) -> str:
        """Get human-readable name for this permutation"""
        provided = []
        if self.model: provided.append("model")
        if self.dataset: provided.append("dataset") 
        if self.prompt: provided.append("prompt")
        if self.mode: provided.append("mode")
        
        if not provided:
            return "no_arguments"
        return "+".join(provided)

@dataclass
class ExperimentCombination:
    """A valid experiment combination after filtering"""
    model: str
    dataset: str
    prompt: str
    mode: str
    
    def __str__(self):
        return f"{self.model}+{self.dataset}+{self.prompt}+{self.mode}"

class ComprehensiveArgumentResolver:
    """
    Systematic resolver for all possible argument combinations.
    
    Handles all 16 possible permutations (including no arguments) with
    explicit logic for each case and comprehensive compatibility filtering.
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
    
    def resolve_arguments(self, args, force: bool = False) -> List[ExperimentCombination]:
        """
        Main entry point - resolves all argument combinations systematically.
        
        Args:
            args: Parsed command line arguments
            force: Whether to ignore compatibility errors
            
        Returns:
            List of valid experiment combinations
        """
        # Parse user inputs
        specified_models = self._parse_list_arg(args.model, self.all_models)
        specified_datasets = self._parse_list_arg(args.dataset, self.all_datasets) 
        specified_prompts = self._parse_list_arg(args.prompt, self.all_prompts)
        specified_mode = args.mode
        
        # Determine argument specification
        arg_spec = ArgumentSpec(
            model = bool(args.model),
            dataset = bool(args.dataset),
            prompt = bool(args.prompt), 
            mode = bool(args.mode)
        )
        
        self.logger.info(f"Resolving permutation: {arg_spec.get_permutation_name()} ({arg_spec.count} arguments)")
        
        # Route to appropriate resolution strategy
        return self._resolve_by_permutation(
            arg_spec, specified_models, specified_datasets, 
            specified_prompts, specified_mode, force
        )
    
    def _parse_list_arg(self, arg_value: Optional[str], all_options: List[str]) -> List[str]:
        """Parse space or comma separated argument values"""
        if not arg_value:
            return []
        if arg_value.lower() == 'all':
            return all_options
        if ',' in arg_value:
            return [item.strip() for item in arg_value.split(',')]
        return arg_value.split()
    
    def _resolve_by_permutation(self, arg_spec: ArgumentSpec, specified_models: List[str],
                               specified_datasets: List[str], specified_prompts: List[str], 
                               specified_mode: Optional[str], force: bool) -> List[ExperimentCombination]:
        """Route to specific permutation handler based on argument pattern"""
        
        # Handle all 16 possible permutations explicitly
        if arg_spec.signature == (False, False, False, False):
            return self._handle_no_arguments()
            
        elif arg_spec.signature == (True, False, False, False):
            return self._handle_model_only(specified_models)
            
        elif arg_spec.signature == (False, True, False, False):
            return self._handle_dataset_only(specified_datasets)
            
        elif arg_spec.signature == (False, False, True, False):
            return self._handle_prompt_only(specified_prompts)
            
        elif arg_spec.signature == (False, False, False, True):
            return self._handle_mode_only(specified_mode)
            
        elif arg_spec.signature == (True, True, False, False):
            return self._handle_model_dataset(specified_models, specified_datasets)
            
        elif arg_spec.signature == (True, False, True, False):
            return self._handle_model_prompt(specified_models, specified_prompts)
            
        elif arg_spec.signature == (True, False, False, True):
            return self._handle_model_mode(specified_models, specified_mode)
            
        elif arg_spec.signature == (False, True, True, False):
            return self._handle_dataset_prompt(specified_datasets, specified_prompts, force)
            
        elif arg_spec.signature == (False, True, False, True):
            return self._handle_dataset_mode(specified_datasets, specified_mode)
            
        elif arg_spec.signature == (False, False, True, True):
            return self._handle_prompt_mode(specified_prompts, specified_mode, force)
            
        elif arg_spec.signature == (True, True, True, False):
            return self._handle_model_dataset_prompt(specified_models, specified_datasets, specified_prompts, force)
            
        elif arg_spec.signature == (True, True, False, True):
            return self._handle_model_dataset_mode(specified_models, specified_datasets, specified_mode)
            
        elif arg_spec.signature == (True, False, True, True):
            return self._handle_model_prompt_mode(specified_models, specified_prompts, specified_mode, force)
            
        elif arg_spec.signature == (False, True, True, True):
            return self._handle_dataset_prompt_mode(specified_datasets, specified_prompts, specified_mode, force)
            
        elif arg_spec.signature == (True, True, True, True):
            return self._handle_all_arguments(specified_models, specified_datasets, specified_prompts, specified_mode, force)
        
        else:
            raise ValueError(f"Unhandled argument permutation: {arg_spec.signature}")
    
    # =============================================================================
    # SINGLE ARGUMENT PERMUTATION HANDLERS  
    # =============================================================================
    
    def _handle_no_arguments(self) -> List[ExperimentCombination]:
        """Handle case: python main.py run-baseline-exp"""
        self.logger.info("No arguments provided - running all compatible combinations")
        return self._build_all_compatible_combinations()
    
    def _handle_model_only(self, models: List[str]) -> List[ExperimentCombination]:
        """Handle case: --model gpt-4o-mini"""
        self.logger.info(f"Model only: running all compatible combinations with models {models}")
        combinations = []
        for model in models:
            combinations.extend(self._build_combinations_for_model(model))
        return combinations
    
    def _handle_dataset_only(self, datasets: List[str]) -> List[ExperimentCombination]:
        """Handle case: --dataset gmeg"""
        self.logger.info(f"Dataset only: running with compatible prompts for datasets {datasets}")
        combinations = []
        for dataset in datasets:
            compatible_prompts = self.dataset_to_prompts.get(dataset, [])
            self.logger.info(f"  Dataset {dataset}: {len(compatible_prompts)} compatible prompts")
            for model in self.all_models:
                for prompt in compatible_prompts:
                    mode = self.prompt_to_mode[prompt]
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        return combinations
    
    def _handle_prompt_only(self, prompts: List[str]) -> List[ExperimentCombination]:
        """Handle case: --prompt gmeg_v1_basic"""
        self.logger.info(f"Prompt only: inferring datasets and using all models for prompts {prompts}")
        combinations = []
        inferred_datasets = set()
        for prompt in prompts:
            dataset = self.prompt_to_dataset.get(prompt)
            if dataset:
                inferred_datasets.add(dataset)
                mode = self.prompt_to_mode[prompt]
                for model in self.all_models:
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        
        self.logger.info(f"  Inferred datasets: {list(inferred_datasets)}")
        return combinations
    
    def _handle_mode_only(self, mode: str) -> List[ExperimentCombination]:
        """Handle case: --mode few-shot"""
        self.logger.info(f"Mode only: running all {mode} compatible prompts")
        compatible_prompts = self.mode_to_prompts.get(mode, [])
        self.logger.info(f"  Found {len(compatible_prompts)} prompts supporting {mode} mode")
        
        combinations = []
        for prompt in compatible_prompts:
            dataset = self.prompt_to_dataset[prompt]
            for model in self.all_models:
                combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        return combinations
    
    # =============================================================================
    # TWO ARGUMENT PERMUTATION HANDLERS
    # =============================================================================
    
    def _handle_model_dataset(self, models: List[str], datasets: List[str]) -> List[ExperimentCombination]:
        """Handle case: --model gpt-4o-mini --dataset gmeg"""
        self.logger.info(f"Model+Dataset: finding compatible prompts")
        combinations = []
        for model in models:
            for dataset in datasets:
                compatible_prompts = self.dataset_to_prompts.get(dataset, [])
                self.logger.info(f"  {model}+{dataset}: {len(compatible_prompts)} compatible prompts")
                for prompt in compatible_prompts:
                    mode = self.prompt_to_mode[prompt]
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        return combinations
    
    def _handle_model_prompt(self, models: List[str], prompts: List[str]) -> List[ExperimentCombination]:
        """Handle case: --model gpt-4o-mini --prompt gmeg_v1_basic"""
        self.logger.info(f"Model+Prompt: inferring datasets")
        combinations = []
        inferred_datasets = set()
        for model in models:
            for prompt in prompts:
                dataset = self.prompt_to_dataset.get(prompt)
                if dataset:
                    inferred_datasets.add(dataset)
                    mode = self.prompt_to_mode[prompt]
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        
        self.logger.info(f"  Inferred datasets: {list(inferred_datasets)}")
        return combinations
    
    def _handle_model_mode(self, models: List[str], mode: str) -> List[ExperimentCombination]:
        """Handle case: --model gpt-4o-mini --mode few-shot"""
        self.logger.info(f"Model+Mode: finding compatible prompts for {mode}")
        compatible_prompts = self.mode_to_prompts.get(mode, [])
        self.logger.info(f"  Found {len(compatible_prompts)} prompts supporting {mode}")
        
        combinations = []
        for model in models:
            for prompt in compatible_prompts:
                dataset = self.prompt_to_dataset[prompt]
                combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        return combinations
    
    def _handle_dataset_prompt(self, datasets: List[str], prompts: List[str], force: bool) -> List[ExperimentCombination]:
        """Handle case: --dataset gmeg --prompt gmeg_v1_basic"""
        self.logger.info(f"Dataset+Prompt: validating compatibility")
        combinations = []
        
        for dataset in datasets:
            for prompt in prompts:
                expected_dataset = self.prompt_to_dataset.get(prompt)
                if expected_dataset != dataset and not force:
                    self.logger.warning(f"  Skipping incompatible: {prompt} (for {expected_dataset}) + {dataset}")
                    continue
                
                mode = self.prompt_to_mode[prompt]
                for model in self.all_models:
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        
        return combinations
    
    def _handle_dataset_mode(self, datasets: List[str], mode: str) -> List[ExperimentCombination]:
        """Handle case: --dataset gmeg --mode few-shot"""
        self.logger.info(f"Dataset+Mode: finding prompts compatible with both")
        combinations = []
        
        for dataset in datasets:
            dataset_prompts = set(self.dataset_to_prompts.get(dataset, []))
            mode_prompts = set(self.mode_to_prompts.get(mode, []))
            compatible_prompts = list(dataset_prompts & mode_prompts)
            
            self.logger.info(f"  {dataset}+{mode}: {len(compatible_prompts)} compatible prompts")
            for model in self.all_models:
                for prompt in compatible_prompts:
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        
        return combinations
    
    def _handle_prompt_mode(self, prompts: List[str], mode: str, force: bool) -> List[ExperimentCombination]:
        """Handle case: --prompt gmeg_v1_basic --mode zero-shot"""
        self.logger.info(f"Prompt+Mode: validating mode compatibility")
        combinations = []
        
        for prompt in prompts:
            prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')
            if prompt_mode != mode and not force:
                self.logger.warning(f"  Skipping incompatible: {prompt} (is {prompt_mode}) with {mode}")
                continue
                
            dataset = self.prompt_to_dataset[prompt]
            for model in self.all_models:
                combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        
        return combinations
    
    # =============================================================================
    # THREE ARGUMENT PERMUTATION HANDLERS
    # =============================================================================
    
    def _handle_model_dataset_prompt(self, models: List[str], datasets: List[str], prompts: List[str], force: bool) -> List[ExperimentCombination]:
        """Handle case: --model gpt-4o-mini --dataset gmeg --prompt gmeg_v1_basic"""
        self.logger.info(f"Model+Dataset+Prompt: validating all compatibility")
        combinations = []
        
        for model in models:
            for dataset in datasets:
                for prompt in prompts:
                    expected_dataset = self.prompt_to_dataset.get(prompt)
                    if expected_dataset != dataset and not force:
                        self.logger.warning(f"  Skipping incompatible: {prompt}+{dataset}")
                        continue
                    
                    mode = self.prompt_to_mode[prompt]
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        
        return combinations
    
    def _handle_model_dataset_mode(self, models: List[str], datasets: List[str], mode: str) -> List[ExperimentCombination]:
        """Handle case: --model gpt-4o-mini --dataset gmeg --mode few-shot"""
        self.logger.info(f"Model+Dataset+Mode: finding compatible prompts")
        combinations = []
        
        for model in models:
            for dataset in datasets:
                dataset_prompts = set(self.dataset_to_prompts.get(dataset, []))
                mode_prompts = set(self.mode_to_prompts.get(mode, []))
                compatible_prompts = list(dataset_prompts & mode_prompts)
                
                self.logger.info(f"  {model}+{dataset}+{mode}: {len(compatible_prompts)} compatible prompts")
                for prompt in compatible_prompts:
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        
        return combinations
    
    def _handle_model_prompt_mode(self, models: List[str], prompts: List[str], mode: str, force: bool) -> List[ExperimentCombination]:
        """Handle case: --model gpt-4o-mini --prompt gmeg_v1_basic --mode zero-shot"""
        self.logger.info(f"Model+Prompt+Mode: validating mode compatibility and inferring datasets")
        combinations = []
        
        for model in models:
            for prompt in prompts:
                prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')
                if prompt_mode != mode and not force:
                    self.logger.warning(f"  Skipping incompatible: {prompt} (is {prompt_mode}) with {mode}")
                    continue
                
                dataset = self.prompt_to_dataset[prompt]
                combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        
        return combinations
    
    def _handle_dataset_prompt_mode(self, datasets: List[str], prompts: List[str], mode: str, force: bool) -> List[ExperimentCombination]:
        """Handle case: --dataset gmeg --prompt gmeg_v1_basic --mode zero-shot"""
        self.logger.info(f"Dataset+Prompt+Mode: validating all compatibility")
        combinations = []
        
        for dataset in datasets:
            for prompt in prompts:
                # Check dataset compatibility
                expected_dataset = self.prompt_to_dataset.get(prompt)
                if expected_dataset != dataset and not force:
                    self.logger.warning(f"  Skipping incompatible: {prompt}+{dataset}")
                    continue
                
                # Check mode compatibility  
                prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')
                if prompt_mode != mode and not force:
                    self.logger.warning(f"  Skipping incompatible: {prompt} (is {prompt_mode}) with {mode}")
                    continue
                
                for model in self.all_models:
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        
        return combinations
    
    # =============================================================================
    # ALL ARGUMENTS HANDLER
    # =============================================================================
    
    def _handle_all_arguments(self, models: List[str], datasets: List[str], prompts: List[str], mode: str, force: bool) -> List[ExperimentCombination]:
        """Handle case: all arguments specified"""
        self.logger.info(f"All arguments specified: validating complete compatibility")
        combinations = []
        
        for model in models:
            for dataset in datasets:
                for prompt in prompts:
                    # Check dataset compatibility
                    expected_dataset = self.prompt_to_dataset.get(prompt)
                    if expected_dataset != dataset and not force:
                        self.logger.warning(f"  Skipping incompatible: {prompt}+{dataset}")
                        continue
                    
                    # Check mode compatibility
                    prompt_mode = self.prompt_to_mode.get(prompt, 'zero-shot')  
                    if prompt_mode != mode and not force:
                        self.logger.warning(f"  Skipping incompatible: {prompt} (is {prompt_mode}) with {mode}")
                        continue
                    
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        
        return combinations
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _build_all_compatible_combinations(self) -> List[ExperimentCombination]:
        """Build all possible valid combinations"""
        combinations = []
        for model in self.all_models:
            for dataset in self.all_datasets:
                for prompt in self.dataset_to_prompts.get(dataset, []):
                    mode = self.prompt_to_mode[prompt]
                    combinations.append(ExperimentCombination(model, dataset, prompt, mode))
        return combinations
    
    def _build_combinations_for_model(self, model: str) -> List[ExperimentCombination]:
        """Build all valid combinations for a specific model"""
        combinations = []
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
            logger.error("❌ Cannot run local models: PyTorch not available")
            logger.info("💡 Available options:")
            logger.info("   - Install PyTorch with CUDA support for GPU acceleration")
            logger.info("   - Use API-based models only (GPT, Gemini, Claude)")
            return False
        else:
            logger.warning("⚠️  Local models requested but no GPU available")
            logger.warning(f"   Requested local models: {', '.join(local_models_requested)}")
            logger.warning("   This will be VERY slow and may require >16GB RAM")
            
            if not force:
                response = input("Continue with CPU-only local model inference? (y/N): ")
                if response.lower() != 'y':
                    logger.info("💡 Consider using API models for better performance:")
                    if api_models_requested:
                        logger.info(f"   Available API models in your selection: {', '.join(api_models_requested)}")
                    else:
                        api_models = [m for m, c in models_config.items() if c['type'] == 'api']
                        logger.info(f"   Available API models: {', '.join(api_models[:3])}...")
                    return False
    
    elif not gpu_status['can_run_local_models']:
        logger.warning(f"⚠️  GPU memory may be insufficient for local models ({gpu_status['total_memory']:.1f} GB available)")
        logger.warning("   Consider using smaller models or API-based models for better reliability")
    
    return True

def run_baseline_experiment_command(args):
    """Run baseline experiments with comprehensive argument permutation handling"""
    logger.info("Starting baseline experiment execution with comprehensive permutation handling...")
    
    # Load configurations
    try:
        prompts_config = Config.load_prompts_config()
        datasets_config = Config.load_datasets_config()
        models_config = Config.load_models_config()
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        return False
    
    # Initialize the comprehensive argument resolver
    resolver = ComprehensiveArgumentResolver(prompts_config, datasets_config, models_config)
    
    # Resolve all argument combinations
    try:
        combinations = resolver.resolve_arguments(args, force=args.force)
    except Exception as e:
        logger.error(f"Error resolving arguments: {e}")
        return False
    
    if not combinations:
        logger.error("No valid experiment combinations found")
        logger.info("Check your arguments and use --force to ignore compatibility warnings")
        return False
    
    # Group combinations by mode for organized execution
    combinations_by_mode = {}
    for combo in combinations:
        mode = combo.mode
        if mode not in combinations_by_mode:
            combinations_by_mode[mode] = []
        combinations_by_mode[mode].append(combo)
    
    # Log execution plan
    logger.info(f"Resolved to {len(combinations)} total experiment combinations:")
    for mode, mode_combos in combinations_by_mode.items():
        logger.info(f"  {mode}: {len(mode_combos)} experiments")
        
        # Show sample combinations for verification
        sample_count = min(3, len(mode_combos))
        for i in range(sample_count):
            combo = mode_combos[i]
            logger.info(f"    - {combo}")
        if len(mode_combos) > sample_count:
            logger.info(f"    ... and {len(mode_combos) - sample_count} more")
    
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
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Execute all combinations
    results = []
    success_count = 0
    failure_count = 0
    
    for mode, mode_combinations in combinations_by_mode.items():
        logger.info(f"Executing {len(mode_combinations)} {mode} experiments...")
        
        for combo in mode_combinations:
            try:
                logger.info(f"Running {mode} experiment: {combo}")
                
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
                    logger.info(f"✅ Completed: {result['experiment_name']}")
                else:
                    failure_count += 1
                    logger.error(f"❌ Failed: {combo}")
            
            except Exception as e:
                failure_count += 1
                logger.error(f"❌ Error in experiment {combo}: {e}")
                continue
    
    # Final summary
    logger.info(f"Experiment execution completed:")
    logger.info(f"  ✅ Success: {success_count}")
    logger.info(f"  ❌ Failed: {failure_count}")
    logger.info(f"  📊 Total: {len(combinations)} combinations processed")
    
    if results:
        logger.info("Generated experiment files:")
        for result in results[:5]:  # Show first 5
            logger.info(f"  - {result['output_file']}")
        if len(results) > 5:
            logger.info(f"  ... and {len(results) - 5} more files")
    
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
                    logger.info(f"✅ Evaluated: {experiment_name}")
                else:
                    logger.error(f"❌ Failed to evaluate: {experiment_name}")
            
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
                    logger.info(f"✅ Generated comparison plots: {results}")
                else:
                    logger.error("❌ Failed to generate comparison plots")
            else:
                # Generate individual plots
                success_count = 0
                for experiment_name in experiments_to_plot:
                    result = plotter.create_individual_plot(experiment_name.strip(), args.experiment_type)
                    if result:
                        success_count += 1
                        logger.info(f"✅ Generated plot for: {experiment_name}")
                    else:
                        logger.error(f"❌ Failed to generate plot for: {experiment_name}")
                
                return success_count > 0
        else:
            # Plot all experiments
            results = plotter.create_all_plots(args.experiment_type)
            if results:
                logger.info(f"✅ Generated plots for {len(results)} experiments")
                return True
            else:
                logger.error("❌ Failed to generate plots")
                return False
                
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
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
            logger.error(f"❌ Unknown dataset: {dataset_name}")
            failure_count += 1
            continue
            
        try:
            if manager.download_dataset(dataset_name):
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            logger.error(f"❌ Error downloading dataset {dataset_name}: {e}")
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
                        logger.info(f"✅ Cleaned directory: {target_dir}")
                    else:
                        os.remove(target_dir)
                        logger.info(f"✅ Removed file: {target_dir}")
                    
                    cleaned_items.append(target)
            else:
                logger.info(f"Target does not exist: {target_dir}")
                
        except Exception as e:
            error_msg = f"Error cleaning {target}: {e}"
            logger.error(f"❌ {error_msg}")
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
    print("     --model MODELS           Model(s) to use (space-separated or 'all')")
    print("     --dataset DATASETS       Dataset(s) to use (space-separated or 'all')")
    print("     --prompt PROMPTS         Prompt(s) to use (space-separated or 'all')")
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
    
    print("\n4. download-datasets")
    print("   Description: Download specified datasets from configured sources")
    print("   Arguments:")
    print("     --dataset DATASETS       Dataset(s) to download (space-separated or 'all')")
    
    print("\n5. cleanup")
    print("   Description: Clean up system files and directories")
    print("   Arguments:")
    print("     --target TARGETS         What to clean: datasets, logs, results, cache, finetuned, all")
    print("     --dry-run               Show what would be cleaned without actually cleaning")
    
    print("\n6. list-options")
    print("   Description: List available models, datasets, prompts, and experiment types")
    print("   Arguments: None")
    
    print("\n7. list-commands")
    print("   Description: Show this help message with all commands and their arguments")
    print("   Arguments: None")
    
    print("\n8. status")
    print("   Description: Check system status including configuration files and API keys")
    print("   Arguments: None")
    
    print("\n=== EXAMPLE USAGE ===")
    print("# Any single argument - system finds compatible combinations")
    print("python main.py run-baseline-exp --dataset gmeg")
    print("python main.py run-baseline-exp --mode few-shot")
    print("python main.py run-baseline-exp --model gpt-4o-mini")
    print("")
    print("# Multiple arguments - system validates compatibility")
    print("python main.py run-baseline-exp --model gpt-4o-mini --mode few-shot")
    print("python main.py run-baseline-exp --dataset gmeg --mode zero-shot")
    print("")
    print("# Complex combinations")
    print("python main.py run-baseline-exp --model \"gpt-4o-mini claude-3.5-sonnet\" --dataset gmeg")
    print("")
    print("# All arguments - traditional explicit specification")
    print("python main.py run-baseline-exp --model gpt-4o-mini --dataset gmeg --prompt gmeg_v1_basic --mode zero-shot")

def check_system_command(args):
    """Check system status"""
    logger.info("Checking system status...")
    
    # Use the enhanced system status display
    gpu_status, memory_status = print_system_status()
    
    # Check configuration files
    config_status = Config.validate_configuration_files()
    
    print("📄 Configuration Files:")
    for config_type, exists in config_status.items():
        status = "✅" if exists else "❌"
        print(f"   {config_type}.json: {status}")
    
    # Check directories
    print("\n📂 Directories:")
    try:
        created_dirs = Config.create_directories()
        print(f"   Created/verified {len(created_dirs)} directories")
    except Exception as e:
        print(f"   Error creating directories: {e}")
    
    # Check API keys
    print("\n🔑 API Keys:")
    print(f"   OpenAI: {'✅' if Config.OPENAI_API_KEY else '❌'}")
    print(f"   Google GenAI: {'✅' if Config.GENAI_API_KEY else '❌'}")
    print(f"   Anthropic: {'✅' if Config.ANTHROPIC_API_KEY else '❌'}")
    
    # Recommendations based on system status
    print("\n💡 Recommendations:")
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
        description="XAI Explanation Evaluation System - Comprehensive Argument Handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use 'python main.py list-commands' to see all available commands and examples."
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Baseline experiment command
    baseline_parser = subparsers.add_parser('run-baseline-exp', help='Run baseline inference experiments')
    baseline_parser.add_argument('--model', type=str,
                                help='Model(s) to use (space-separated, or "all" for all models)')
    baseline_parser.add_argument('--dataset', type=str,
                                help='Dataset(s) to use (space-separated, or "all" for all datasets)')
    baseline_parser.add_argument('--prompt', type=str,
                                help='Prompt(s) to use (space-separated, or "all" for all prompts)')
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