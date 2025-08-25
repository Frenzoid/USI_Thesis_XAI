#!/usr/bin/env python3
"""
XAI Explanation Evaluation System - Refactored CLI

This script provides a command-line interface for running experiments to evaluate
Large Language Models' alignment with user study results in XAI explanation evaluation.
"""

import argparse
import sys
import os
import shutil
from typing import List, Optional

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

def setup_cli_logging():
    """Setup logging for CLI"""
    logger = setup_logging("main", "INFO")
    return logger

def validate_experiment_type(experiment_type: str):
    """Validate experiment type"""
    if not Config.validate_experiment_type(experiment_type):
        raise ValueError(f"Unsupported experiment type: {experiment_type}. Supported: {Config.EXPERIMENT_TYPES}")

def parse_list_argument(arg_value: Optional[str], all_options: List[str]) -> List[str]:
    """Parse command line arguments that can be 'all' or a space-separated list"""
    if not arg_value or arg_value.lower() == 'all':
        return all_options
    # Handle both space-separated and comma-separated values
    if ',' in arg_value:
        return [item.strip() for item in arg_value.split(',')]
    else:
        return arg_value.split()

def run_experiment_command(args):
    """Run experiments with specified parameters"""
    logger.info("Starting experiment execution...")
    
    # Check if we can run experiments based on system capabilities
    from utils import check_gpu_availability
    gpu_status = check_gpu_availability()
    
    # Validate model types being requested
    try:
        models_config = Config.load_models_config()
    except Exception as e:
        logger.error(f"Failed to load models configuration: {e}")
        return False
    
    # Parse model arguments
    all_models = list(models_config.keys())
    models_to_run = parse_list_argument(args.model, all_models)
    
    # Check if any local models are being requested
    local_models_requested = []
    api_models_requested = []
    
    for model in models_to_run:
        if model in models_config:
            if models_config[model]['type'] == 'local':
                local_models_requested.append(model)
            else:
                api_models_requested.append(model)
    
    # Warn about local models on CPU or insufficient GPU
    if local_models_requested:
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
                
                if not args.force:
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
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # ... rest of the existing function remains the same ...
    
    # Validate experiment type
    validate_experiment_type(args.experiment_type)
    
    # Load configurations
    try:
        prompts_config = Config.load_prompts_config()
        datasets_config = Config.load_datasets_config()
        models_config = Config.load_models_config()
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        return False
    
    # Parse arguments - handle case where arguments might be None
    all_models = list(models_config.keys())
    all_datasets = list(datasets_config.keys())
    all_prompts = list(prompts_config.keys())
    
    models_to_run = parse_list_argument(args.model, all_models)
    datasets_to_run = parse_list_argument(args.dataset, all_datasets)
    prompts_to_run = parse_list_argument(args.prompt, all_prompts)
    
    # Validate configurations for compatibility
    failed_validations = []
    for dataset in datasets_to_run:
        for prompt in prompts_to_run:
            if prompts_config[prompt]['compatible_dataset'] != dataset:
                failed_validations.append(f"Prompt '{prompt}' not compatible with dataset '{dataset}'")
    
    if failed_validations and not args.force:
        logger.error("Configuration validation failures:")
        for failure in failed_validations:
            logger.error(f"  - {failure}")
        logger.error("Use --force to ignore validation errors")
        return False
    
    # Execute experiments
    total_experiments = len(models_to_run) * len(datasets_to_run) * len(prompts_to_run)
    logger.info(f"Running {total_experiments} experiment configurations")
    
    results = []
    success_count = 0
    failure_count = 0
    
    for model in models_to_run:
        for dataset in datasets_to_run:
            for prompt in prompts_to_run:
                # Skip invalid combinations unless forced
                if not args.force and prompts_config[prompt]['compatible_dataset'] != dataset:
                    logger.warning(f"Skipping incompatible combination: {prompt} + {dataset}")
                    continue
                
                try:
                    logger.info(f"Running experiment: {model} + {dataset} + {prompt}")
                    
                    experiment_config = {
                        'experiment_type': args.experiment_type,
                        'model': model,
                        'dataset': dataset,
                        'prompt': prompt,
                        'size': args.size,
                        'temperature': args.temperature
                    }
                    
                    if args.experiment_type == "baseline":
                        result = runner.run_baseline_experiment(experiment_config)
                    else:
                        raise ValueError(f"Experiment type '{args.experiment_type}' not implemented yet")
                    
                    if result:
                        results.append(result)
                        success_count += 1
                        logger.info(f"✅ Completed experiment: {result['experiment_name']}")
                    else:
                        failure_count += 1
                        logger.error(f"❌ Failed experiment: {model} + {dataset} + {prompt}")
                
                except Exception as e:
                    failure_count += 1
                    logger.error(f"❌ Error in experiment {model} + {dataset} + {prompt}: {e}")
                    continue
    
    # Summary
    logger.info(f"Experiment batch completed: {success_count} successful, {failure_count} failed")
    
    if results:
        logger.info("Generated experiment files:")
        for result in results:
            logger.info(f"  - {result['output_file']}")
    
    return success_count > 0

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
    datasets_to_download = parse_list_argument(args.dataset, all_datasets)
    
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
            print(f"  - {prompt_name}: {config['description']} [for {compatible}]")
        
        print(f"\nExperiment Types ({len(Config.EXPERIMENT_TYPES)}):")
        for exp_type in Config.EXPERIMENT_TYPES:
            print(f"  - {exp_type}")
        
    except Exception as e:
        print(f"Error loading configurations: {e}")

def list_commands_command():
    """List all available commands and their arguments with proper formatting"""
    print("\n=== AVAILABLE COMMANDS ===")
    
    print("\n1. run-experiment")
    print("   Description: Run inference experiments with specified models, datasets, and prompts")
    print("   Arguments:")
    print("     --experiment-type TYPE    Experiment type (default: baseline)")
    print("     --model MODELS           Model(s) to use (space-separated or 'all')")
    print("     --dataset DATASETS       Dataset(s) to use (space-separated or 'all')")
    print("     --prompt PROMPTS         Prompt(s) to use (space-separated or 'all')")
    print("     --size SIZE              Sample size (default: 50)")
    print("     --temperature TEMP       Temperature for generation (default: 0.1)")
    print("     --force                  Force run even with validation errors")
    
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
    print("# Run single experiment")
    print("python main.py run-experiment --model llama3.2-1b --dataset gmeg --prompt gmeg_v1_basic")
    print("")
    print("# Evaluate all baseline experiments") 
    print("python main.py evaluate --experiment-type baseline")
    print("")
    print("# Generate comparison plots")
    print("python main.py plot --experiment exp1,exp2,exp3 --compare")
    print("")
    print("# Clean up all temporary files")
    print("python main.py cleanup --target all")

def check_system_command(args):
    """Check system status"""
    logger.info("Checking system status...")
    
    # Use the enhanced system status display
    gpu_status, memory_status = print_system_status()
    
    # Check configuration files
    config_status = Config.validate_configuration_files()
    
    print("📁 Configuration Files:")
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
        logger.info("XAI Explanation Evaluation System - CLI")
    except Exception as e:
        print(f"System initialization failed: {e}")
        return 1
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description="XAI Explanation Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use 'python main.py list-commands' to see all available commands and examples."
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run experiment command
    run_parser = subparsers.add_parser('run-experiment', help='Run inference experiments')
    run_parser.add_argument('--experiment-type', type=str, default='baseline',
                          help='Type of experiment to run (default: baseline)')
    run_parser.add_argument('--model', type=str,
                          help='Model(s) to use (space-separated, or "all" for all models)')
    run_parser.add_argument('--dataset', type=str,
                          help='Dataset(s) to use (space-separated, or "all" for all datasets)')
    run_parser.add_argument('--prompt', type=str,
                          help='Prompt(s) to use (space-separated, or "all" for all prompts)')
    run_parser.add_argument('--size', type=int, default=Config.DEFAULT_SAMPLE_SIZE,
                          help=f'Sample size (default: {Config.DEFAULT_SAMPLE_SIZE})')
    run_parser.add_argument('--temperature', type=float, default=Config.DEFAULT_TEMPERATURE,
                          help=f'Temperature for generation (default: {Config.DEFAULT_TEMPERATURE})')
    run_parser.add_argument('--force', action='store_true',
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
        if args.command == 'run-experiment':
            success = run_experiment_command(args)
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
            logger.info("✅ Command completed successfully")
            return 0
        else:
            logger.error("❌ Command failed")
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