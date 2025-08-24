import os
import sys
import json
import time
import gc
import logging
import subprocess
import shutil
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Any
import torch
import random
import numpy as np

from config import Config

# Set up logging
def setup_logging(name: str, level: str = None):
    """Set up logging for a component"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level or Config.LOG_LEVEL))
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Ensure logs directory exists
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(Config.get_log_file(name))
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(Config.LOG_FORMAT)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

# Initialize main logger
logger = setup_logging("utils")

def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    logger.debug("Clearing GPU memory...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    logger.debug("GPU memory cleared")

def show_gpu_stats():
    """Display current GPU memory usage"""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        memory_used = torch.cuda.max_memory_reserved() / 1024**3
        memory_total = gpu_stats.total_memory / 1024**3
        
        stats_msg = f"GPU: {gpu_stats.name}, Memory: {memory_used:.2f} GB / {memory_total:.2f} GB"
        logger.info(stats_msg)
        return stats_msg
    else:
        logger.warning("CUDA not available")
        return "CUDA not available"

def set_random_seeds(seed: int = None):
    """Set random seeds for reproducibility"""
    seed = seed or Config.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seeds to {seed}")

def get_system_info():
    """Get system information for debugging"""
    info = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_directory': os.getcwd(),
        'config_directories': {
            'configs': Config.CONFIGS_DIR,
            'data': Config.DATA_DIR,
            'outputs': Config.OUTPUTS_DIR,
            'responses': Config.RESPONSES_DIR,
            'evaluations': Config.EVALUATIONS_DIR,
            'plots': Config.PLOTS_DIR,
            'models_cache': Config.MODELS_CACHE_DIR,
            'finetuned_models': Config.FINETUNED_MODELS_DIR
        }
    }
    
    if torch.cuda.is_available():
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            info['gpu_info'] = {
                'name': gpu_props.name,
                'total_memory_gb': gpu_props.total_memory / 1024**3,
                'major': gpu_props.major,
                'minor': gpu_props.minor
            }
        except Exception as e:
            info['gpu_info_error'] = str(e)
    
    logger.info(f"System info collected: Python {info['python_version'].split()[0]}, "
                f"PyTorch {info['torch_version']}, CUDA: {info['cuda_available']}")
    
    return info

def validate_api_keys():
    """Validate that required API keys are present"""
    api_status = {}
    
    if Config.OPENAI_API_KEY:
        api_status['openai'] = True
        logger.info("OpenAI API key found")
    else:
        api_status['openai'] = False
        logger.warning("OpenAI API key not found - OpenAI models will be unavailable")
    
    if Config.GENAI_API_KEY:
        api_status['genai'] = True
        logger.info("Google GenAI API key found")
    else:
        api_status['genai'] = False
        logger.warning("Google GenAI API key not found - Gemini models will be unavailable")
    
    if Config.ANTHROPIC_API_KEY:
        api_status['anthropic'] = True
        logger.info("Anthropic API key found")
    else:
        api_status['anthropic'] = False
        logger.warning("Anthropic API key not found - Claude models will be unavailable")
    
    return api_status

def validate_configuration_files():
    """Validate that configuration files exist and are valid"""
    logger.info("Validating configuration files...")
    
    validation_results = {
        'files_exist': Config.validate_configuration_files(),
        'configs_valid': {}
    }
    
    # Test loading each configuration
    try:
        prompts_config = Config.load_prompts_config()
        validation_results['configs_valid']['prompts'] = True
        logger.info(f"Prompts config valid: {len(prompts_config)} prompts loaded")
    except Exception as e:
        validation_results['configs_valid']['prompts'] = False
        logger.error(f"Prompts config invalid: {e}")
    
    try:
        datasets_config = Config.load_datasets_config()
        validation_results['configs_valid']['datasets'] = True
        logger.info(f"Datasets config valid: {len(datasets_config)} datasets loaded")
    except Exception as e:
        validation_results['configs_valid']['datasets'] = False
        logger.error(f"Datasets config invalid: {e}")
    
    try:
        models_config = Config.load_models_config()
        validation_results['configs_valid']['models'] = True
        logger.info(f"Models config valid: {len(models_config)} models loaded")
    except Exception as e:
        validation_results['configs_valid']['models'] = False
        logger.error(f"Models config invalid: {e}")
    
    return validation_results

@contextmanager
def experiment_context(experiment_name: str):
    """Context manager for experiments with cleanup and error handling"""
    logger.info(f"Starting experiment context: {experiment_name}")
    logger.info("=" * 50)
    
    start_time = time.time()
    success = False
    
    try:
        yield
        success = True
    except Exception as e:
        logger.error(f"Experiment '{experiment_name}' failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        end_time = time.time()
        duration = end_time - start_time
        
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{status}: Experiment '{experiment_name}' completed in {duration:.2f} seconds")
        
        # Always clean up GPU memory
        try:
            clear_gpu_memory()
        except Exception as cleanup_error:
            logger.warning(f"GPU cleanup error: {cleanup_error}")

def download_dataset(dataset_json: str, dataset_path: str):
    """Download datasets from JSON configuration (legacy function for compatibility)"""
    logger.info(f"Starting dataset download from {dataset_json}")
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        logger.info(f"Created dataset directory: {dataset_path}")

    if not os.path.exists(dataset_json):
        logger.error(f"Dataset configuration file {dataset_json} not found")
        return

    try:
        with open(dataset_json, "r") as f:
            datasets = json.load(f)
        logger.info(f"Loaded dataset configuration with {len(datasets)} datasets")
    except Exception as e:
        logger.error(f"Error reading dataset configuration: {e}")
        return

    for dataset in datasets:
        try:
            storage_folder = os.path.join(dataset_path, dataset["storage_folder"])
            os.makedirs(storage_folder, exist_ok=True)
            
            filename = os.path.join(storage_folder, dataset["name"])
            
            if not os.path.exists(filename):
                logger.info(f"Downloading {dataset['name']}...")
                try:
                    subprocess.run(["wget", "-q", "-O", filename, dataset["link"]], 
                                 check=True, timeout=300)
                    
                    if filename.endswith('.zip'):
                        logger.info(f"Extracting {filename}...")
                        subprocess.run(["unzip", "-q", filename, "-d", os.path.dirname(filename)], 
                                     check=True, timeout=120)
                    
                    logger.info(f"Successfully downloaded: {dataset['name']}")
                    
                except subprocess.TimeoutExpired:
                    logger.error(f"Download timeout for {dataset['name']}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to download {dataset['name']}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error downloading {dataset['name']}: {e}")
            else:
                logger.info(f"Already exists: {dataset['name']}")
                
        except KeyError as e:
            logger.error(f"Missing key in dataset configuration: {e}")
        except Exception as e:
            logger.error(f"Error processing dataset configuration: {e}")

def safe_filename(name: str) -> str:
    """Create a safe filename from any string"""
    # Remove or replace problematic characters
    safe_chars = "".join(c for c in name if c.isalnum() or c in ('-', '_', '.'))
    # Ensure it's not empty and not too long
    if not safe_chars:
        safe_chars = "unnamed"
    return safe_chars[:100]  # Limit length

def load_json_config(filepath: str) -> Dict[str, Any]:
    """Load JSON configuration file"""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {filepath}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {filepath}: {e}")
        return {}

def create_directory_structure():
    """Create all necessary directories for the system"""
    logger.info("Creating directory structure...")
    
    try:
        created_dirs = Config.create_directories()
        logger.info(f"Created/verified {len(created_dirs)} directories")
        
        # Create configuration files if they don't exist
        config_files_status = Config.validate_configuration_files()
        
        missing_configs = [name for name, exists in config_files_status.items() if not exists]
        if missing_configs:
            logger.warning(f"Missing configuration files: {missing_configs}")
            logger.info("Please ensure the following files exist:")
            for config_name in missing_configs:
                if config_name == 'prompts':
                    logger.info(f"  - {Config.PROMPTS_JSON}")
                elif config_name == 'datasets':
                    logger.info(f"  - {Config.DATASETS_JSON}")
                elif config_name == 'models':
                    logger.info(f"  - {Config.MODELS_JSON}")
        
        return created_dirs
        
    except Exception as e:
        logger.error(f"Error creating directory structure: {e}")
        raise

def check_system_requirements():
    """Check system requirements and dependencies"""
    logger.info("Checking system requirements...")

    # Figure out torch / cuda availability without shadowing the global name
    torch_available = True
    cuda_available = False
    try:
        # if torch is installed, this works; if not, the except block handles it
        cuda_available = torch.cuda.is_available()
    except Exception:
        torch_available = False
        cuda_available = False
        logger.error("PyTorch not available")

    requirements = {
        'python_version_ok': sys.version_info >= (3, 8),
        'torch_available': torch_available,
        'cuda_available': cuda_available,
        'directories_created': False,
        'config_files_valid': False,
        'api_keys_present': False
    }

    # Check directories
    try:
        create_directory_structure()
        requirements['directories_created'] = True
    except Exception as e:
        logger.error(f"Directory creation failed: {e}")

    # Check configuration files
    try:
        validation_results = validate_configuration_files()
        all_configs_valid = all(validation_results['configs_valid'].values())
        requirements['config_files_valid'] = all_configs_valid
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")

    # Check API keys
    api_status = validate_api_keys()
    requirements['api_keys_present'] = any(api_status.values())

    return requirements

def initialize_system():
    """Initialize the system - create directories, set seeds, validate configs"""
    logger.info("Initializing XAI explanation evaluation system...")
    
    # Set random seeds
    set_random_seeds()
    
    # Check system requirements
    requirements = check_system_requirements()
    
    # Get system info
    system_info = get_system_info()
    
    # Show GPU stats if available
    if torch.cuda.is_available():
        show_gpu_stats()
    
    # Log initialization results
    if requirements['python_version_ok']:
        logger.info(f"✅ Python version: {sys.version.split()[0]}")
    else:
        logger.error(f"❌ Python version {sys.version.split()[0]} < 3.8 required")
    
    if requirements['torch_available']:
        logger.info("✅ PyTorch available")
    else:
        logger.error("❌ PyTorch not available")
    
    if requirements['cuda_available']:
        logger.info("✅ CUDA available")
    else:
        logger.info("ℹ️  CUDA not available (will use CPU)")
    
    if requirements['directories_created']:
        logger.info("✅ Directory structure created")
    else:
        logger.error("❌ Directory structure creation failed")
    
    if requirements['config_files_valid']:
        logger.info("✅ Configuration files valid")
    else:
        logger.error("❌ Configuration files invalid or missing")
    
    if requirements['api_keys_present']:
        logger.info("✅ API keys configured")
    else:
        logger.warning("⚠️  No API keys configured (API models will be unavailable)")
    
    logger.info("System initialization complete")
    
    return {
        'requirements': requirements,
        'system_info': system_info,
        'api_status': validate_api_keys()
    }

"""
# Run initialization when imported (only if not already run)
if __name__ != "__main__":
    try:
        _initialization_result = initialize_system()
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        # Don't raise the exception to allow partial functionality
"""