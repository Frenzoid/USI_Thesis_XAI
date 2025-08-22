#!/usr/bin/env python3
"""
XAI Explanation Evaluation System - Command Line Interface

This script provides a command-line interface for running experiments to evaluate
Large Language Models' alignment with user study results in XAI explanation evaluation.
"""

import argparse
import sys
import json
import os
from typing import List, Dict

# Load environment variables first, before any other imports
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

# Now import the rest
from config import Config
from utils import setup_logging, download_dataset, initialize_system, load_json_config
from experiments import ExperimentRunner

def setup_cli_logging():
    """Setup logging for CLI"""
    logger = setup_logging("main", "INFO")
    return logger

def download_datasets_command(args):
    """Download datasets specified in datasets.json"""
    logger.info("Starting dataset download...")
    
    datasets_config = args.datasets_config or Config.DATASETS_JSON
    
    if not os.path.exists(datasets_config):
        logger.error(f"Datasets configuration file not found: {datasets_config}")
        return False
    
    try:
        download_dataset(datasets_config, Config.DATA_DIR)
        logger.info("Dataset download completed successfully")
        return True
    except Exception as e:
        logger.error(f"Dataset download failed: {e}")
        return False

def run_experiments_command(args):
    """Run experiments from configuration file"""
    logger.info("Starting experiment execution...")
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Initialize system components
    runner.initialize_system()
    
    # Load experiment configurations
    if args.config:
        experiments = runner.load_experiments_from_config(args.config)
    else:
        logger.error("No experiment configuration file provided")
        return False
    
    if not experiments:
        logger.error("No valid experiment configurations loaded")
        return False
    
    # Filter experiments if specific models/prompts requested
    if args.models:
        experiments = [exp for exp in experiments if exp.get('model_name') in args.models]
        logger.info(f"Filtered to {len(experiments)} experiments matching models: {args.models}")
    
    if args.prompts:
        experiments = [exp for exp in experiments if exp.get('prompt_key') in args.prompts]
        logger.info(f"Filtered to {len(experiments)} experiments matching prompts: {args.prompts}")
    
    # Validate configurations
    valid_experiments = []
    for i, config in enumerate(experiments):
        if runner.validate_experiment_config(config):
            valid_experiments.append(config)
        else:
            logger.warning(f"Skipping invalid experiment config {i+1}")
    
    if not valid_experiments:
        logger.error("No valid experiment configurations found")
        return False
    
    logger.info(f"Running {len(valid_experiments)} valid experiments")
    
    # Set force rerun if specified
    if args.force:
        for config in valid_experiments:
            config['force_rerun'] = True
    
    # Run experiments
    try:
        results = runner.run_experiment_batch(
            valid_experiments, 
            auto_visualize=not args.no_viz
        )
        
        if results:
            logger.info(f"Successfully completed {len(results)} experiments")
            
            # Print summary
            summary = runner.get_results_summary()
            logger.info("=== EXPERIMENT SUMMARY ===")
            logger.info(f"Total experiments: {summary['total_experiments']}")
            logger.info(f"Successful experiments: {summary['successful_experiments']}")
            logger.info(f"Total samples evaluated: {summary['total_samples']}")
            logger.info(f"Models tested: {summary['models_tested']}")
            logger.info(f"Prompts tested: {summary['prompts_tested']}")
            
            if 'best_performance' in summary:
                best = summary['best_performance']
                logger.info(f"Best performance: {best['experiment']} (F1: {best['f1_score']:.4f})")
            
            return True
        else:
            logger.error("No experiments completed successfully")
            return False
            
    except Exception as e:
        logger.error(f"Error running experiments: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def generate_visualizations_command(args):
    """Generate visualizations from existing results"""
    logger.info("Generating visualizations from existing results...")
    
    if not os.path.exists(Config.RESULTS_DIR):
        logger.error(f"Results directory not found: {Config.RESULTS_DIR}")
        return False
    
    # Load all result files
    result_files = [f for f in os.listdir(Config.RESULTS_DIR) if f.endswith('.json')]
    
    if not result_files:
        logger.error("No result files found")
        return False
    
    logger.info(f"Found {len(result_files)} result files")
    
    # Load results
    from utils import load_results
    results = []
    for file in result_files:
        filepath = os.path.join(Config.RESULTS_DIR, file)
        result = load_results(filepath)
        if result and 'aggregated_scores' in result:
            results.append(result)
    
    if not results:
        logger.error("No valid results loaded")
        return False
    
    logger.info(f"Loaded {len(results)} valid results")
    
    # Generate visualizations
    try:
        runner = ExperimentRunner()
        runner.auto_generate_visualizations(results)
        logger.info("Visualizations generated successfully")
        return True
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return False

def list_models_command(args):
    """List available models"""
    logger.info("Listing available models...")
    
    from models import ModelManager
    model_manager = ModelManager()
    model_manager.setup_api_clients()
    
    available_models = model_manager.get_available_models()
    
    print("\n=== AVAILABLE MODELS ===")
    
    if available_models['open_source']:
        print("\nOpen Source Models:")
        for model in available_models['open_source']:
            print(f"  - {model}")
    else:
        print("\nOpen Source Models: None available (check Unsloth installation)")
    
    if available_models['api']:
        print("\nAPI Models:")
        for model in available_models['api']:
            print(f"  - {model}")
    else:
        print("\nAPI Models: None available (check API keys)")
    
    return True

def list_prompts_command(args):
    """List available prompts"""
    logger.info("Listing available prompts...")
    
    from prompts import PromptManager
    prompt_manager = PromptManager()
    
    prompts = prompt_manager.list_prompts()
    
    print("\n=== AVAILABLE PROMPTS ===")
    for key, description in prompts.items():
        print(f"\n{key}:")
        print(f"  {description}")
    
    return True

def list_datasets_command(args):
    """List available datasets"""
    logger.info("Listing available datasets...")
    
    from prompts import PromptManager
    from datasets_manager import DatasetManager
    
    prompt_manager = PromptManager()
    dataset_manager = DatasetManager(prompt_manager)
    
    available_datasets = dataset_manager.get_available_datasets()
    
    print("\n=== AVAILABLE DATASETS ===")
    
    for dataset_name, info in available_datasets.items():
        config = info['config']
        status = "✅ Downloaded" if info['exists'] else "❌ Not Downloaded"
        loaded = "📂 Loaded" if info['loaded'] else ""
        
        print(f"\n📊 {config['name']} ({dataset_name})")
        print(f"   Status: {status} {loaded}")
        print(f"   Type: {config['type']}")
        print(f"   Task: {config.get('task_type', 'N/A')}")
        print(f"   Description: {config['description'][:100]}...")
        print(f"   Download: {info['download_link']}")
        
        # Show field mapping
        field_mapping = config.get('field_mapping', {})
        if field_mapping:
            print(f"   Question fields: {field_mapping.get('question_fields', [])}")
            print(f"   Answer field: {field_mapping.get('answer_field', 'N/A')}")
        
        # Show compatible prompts
        compatible_prompts = config.get('compatible_prompts', [])
        if compatible_prompts:
            print(f"   Compatible prompts: {compatible_prompts[:3]}{'...' if len(compatible_prompts) > 3 else ''}")
    
    print(f"\n📈 Total: {len(available_datasets)} datasets configured")
    downloaded_count = sum(1 for info in available_datasets.values() if info['exists'])
    print(f"📥 Downloaded: {downloaded_count}/{len(available_datasets)}")
    
    return True

def check_compatibility_command(args):
    """Check dataset and prompt compatibility"""
    logger.info("Checking dataset and prompt compatibility...")
    
    from prompts import PromptManager
    from datasets_manager import DatasetManager
    
    prompt_manager = PromptManager()
    dataset_manager = DatasetManager(prompt_manager)
    
    print("\n=== DATASET & PROMPT COMPATIBILITY ===")
    
    # If specific dataset and prompt specified
    if args.dataset and args.prompt:
        dataset_name = args.dataset.lower().replace('-', '_')
        prompt_key = args.prompt
        
        compatible = dataset_manager.validate_prompt_compatibility(dataset_name, prompt_key)
        status = "✅ Compatible" if compatible else "❌ Not Compatible"
        
        print(f"\n{status}: {prompt_key} with {dataset_name}")
        
        if not compatible:
            compatible_prompts = dataset_manager.get_compatible_prompts(dataset_name)
            print(f"💡 Compatible prompts for {dataset_name}: {compatible_prompts}")
        
        return compatible
    
    # Show full compatibility matrix
    available_datasets = dataset_manager.get_available_datasets()
    available_prompts = list(prompt_manager.prompts.keys())
    
    print("\n📋 Compatibility Matrix:")
    print("=" * 50)
    
    for dataset_name, info in available_datasets.items():
        config = info['config']
        print(f"\n📊 {config['name']} ({dataset_name}):")
        
        compatible_prompts = dataset_manager.get_compatible_prompts(dataset_name)
        
        print("   ✅ Compatible prompts:")
        for prompt in compatible_prompts:
            print(f"      - {prompt}")
        
        incompatible_prompts = [p for p in available_prompts if p not in compatible_prompts]
        if incompatible_prompts:
            print("   ❌ Incompatible prompts:")
            for prompt in incompatible_prompts[:5]:  # Show first 5
                print(f"      - {prompt}")
            if len(incompatible_prompts) > 5:
                print(f"      ... and {len(incompatible_prompts) - 5} more")
    
    return True
    """Check system status and requirements"""
    logger.info("Checking system status...")
    
    from utils import get_system_info, validate_api_keys
    from models import ModelManager
    
    # Get system info
    system_info = get_system_info()
    
    print("\n=== SYSTEM STATUS ===")
    print(f"Python: {system_info['python_version'].split()[0]}")
    print(f"PyTorch: {system_info['torch_version']}")
    print(f"CUDA Available: {system_info['cuda_available']}")
    
    if system_info['cuda_available'] and 'gpu_info' in system_info:
        gpu_info = system_info['gpu_info']
        print(f"GPU: {gpu_info['name']} ({gpu_info['total_memory_gb']:.1f} GB)")
    
    # Check API keys
    api_status = validate_api_keys()
    print(f"\nAPI Keys:")
    print(f"  OpenAI: {'✓' if api_status['openai'] else '✗'}")
    print(f"  Google GenAI: {'✓' if api_status['genai'] else '✗'}")
    print(f"  Anthropic: {'✓' if api_status['anthropic'] else '✗'}")
    
    # Check model availability
    model_manager = ModelManager()
    model_manager.setup_api_clients()
    available_models = model_manager.get_available_models()
    
    print(f"\nModel Availability:")
    print(f"  Open Source: {len(available_models['open_source'])} available")
    print(f"  API Models: {len(available_models['api'])} available")
    
    # Check directories
    print(f"\nDirectories:")
    for name, path in system_info['config_directories'].items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {name}: {path} {exists}")
    
    return True

def create_example_configs_command(args):
    """Create example configuration files"""
    logger.info("Creating example configuration files...")
    
    # Example datasets.json
    datasets_config = [
        {
            "name": "GMEG-EXP",
            "link": "https://github.com/your-repo/gmeg-exp/archive/main.zip",
            "storage_folder": "DS_GMEG_EXP"
        }
    ]
    
    datasets_file = "datasets_example.json"
    with open(datasets_file, 'w') as f:
        json.dump(datasets_config, f, indent=2)
    print(f"Created example datasets config: {datasets_file}")
    
    # Example experiments.json
    experiments_config = {
        "experiments": [
            {
                "model_name": "llama3.2-1b",
                "model_type": "open_source",
                "prompt_key": "gmeg_v1_basic",
                "sample_size": 10,
                "dataset_type": "gmeg",
                "dataset_name": "gmeg",
                "experiment_name": "baseline_llama_basic"
            },
            {
                "model_name": "llama3.2-1b",
                "model_type": "open_source",
                "prompt_key": "gmeg_v2_enhanced",
                "sample_size": 10,
                "dataset_type": "gmeg",
                "dataset_name": "gmeg",
                "experiment_name": "baseline_llama_enhanced"
            },
            {
                "model_name": "gpt-4o-mini",
                "model_type": "api",
                "prompt_key": "gmeg_v1_basic",
                "sample_size": 10,
                "dataset_type": "gmeg",
                "dataset_name": "gmeg",
                "experiment_name": "baseline_gpt_basic"
            },
            {
                "model_name": "gemini-1.5-flash",
                "model_type": "api",
                "prompt_key": "gmeg_v1_basic",
                "sample_size": 10,
                "dataset_type": "gmeg",
                "dataset_name": "gmeg",
                "experiment_name": "baseline_gemini_basic"
            }
        ]
    }
    
    experiments_file = "experiments_example.json"
    with open(experiments_file, 'w') as f:
        json.dump(experiments_config, f, indent=2)
    print(f"Created example experiments config: {experiments_file}")
    
    print("\nExample configuration files created!")
    print("Edit these files with your specific datasets and experiment configurations.")
    
    return True

def main():
    """Main CLI entry point"""
    # Initialize system and logging
    init_result = initialize_system()
    global logger
    logger = setup_cli_logging()
    
    logger.info("XAI Explanation Evaluation System - CLI")
    logger.info("=" * 50)
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description="XAI Explanation Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download datasets
  python main.py download --datasets-config datasets.json
  
  # List available resources
  python main.py list-datasets
  python main.py list-models
  python main.py list-prompts
  
  # Check compatibility
  python main.py check-compatibility --dataset gmeg_exp --prompt gmeg_v1_basic
  python main.py check-compatibility  # Show full compatibility matrix
  
  # Run all experiments
  python main.py run --config experiments.json
  
  # Run specific models only
  python main.py run --config experiments.json --models llama3.2-1b gpt-4o-mini
  
  # Run specific datasets only  
  python main.py run --config experiments.json --models llama3.2-1b --prompts general_basic
  
  # Force rerun existing experiments
  python main.py run --config experiments.json --force
  
  # Generate visualizations only
  python main.py visualize
  
  # Check system status
  python main.py status
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download datasets')
    download_parser.add_argument('--datasets-config', type=str, 
                               help='Path to datasets configuration JSON file')
    
    # Run experiments command
    run_parser = subparsers.add_parser('run', help='Run experiments')
    run_parser.add_argument('--config', type=str, required=True,
                          help='Path to experiments configuration JSON file')
    run_parser.add_argument('--models', nargs='+', type=str,
                          help='Run only experiments with these models')
    run_parser.add_argument('--prompts', nargs='+', type=str,
                          help='Run only experiments with these prompts')
    run_parser.add_argument('--force', action='store_true',
                          help='Force rerun of existing experiments')
    run_parser.add_argument('--no-viz', action='store_true',
                          help='Skip automatic visualization generation')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations from existing results')
    
    # List commands
    models_parser = subparsers.add_parser('list-models', help='List available models')
    prompts_parser = subparsers.add_parser('list-prompts', help='List available prompts')
    datasets_parser = subparsers.add_parser('list-datasets', help='List available datasets')
    
    # Compatibility check command
    compat_parser = subparsers.add_parser('check-compatibility', help='Check dataset and prompt compatibility')
    compat_parser.add_argument('--dataset', type=str, help='Dataset name to check')
    compat_parser.add_argument('--prompt', type=str, help='Prompt key to check')
    
    # System status command
    status_parser = subparsers.add_parser('status', help='Check system status and requirements')
    
    # Create example configs command
    examples_parser = subparsers.add_parser('create-examples', help='Create example configuration files')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        if args.command == 'download':
            success = download_datasets_command(args)
        elif args.command == 'run':
            success = run_experiments_command(args)
        elif args.command == 'visualize':
            success = generate_visualizations_command(args)
        elif args.command == 'list-models':
            success = list_models_command(args)
        elif args.command == 'list-prompts':
            success = list_prompts_command(args)
        elif args.command == 'list-datasets':
            success = list_datasets_command(args)
        elif args.command == 'check-compatibility':
            success = check_compatibility_command(args)
        elif args.command == 'status':
            success = check_system_command(args)
        elif args.command == 'create-examples':
            success = create_example_configs_command(args)
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