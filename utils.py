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

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(name: str, level: str = None):
    """
    Set up logging for a component with both file and console output.
    
    Args:
        name: Component name for logger and log file
        level: Log level (defaults to Config.LOG_LEVEL)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level or Config.LOG_LEVEL))
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Ensure logs directory exists
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        # File handler - saves all logs to file
        fh = logging.FileHandler(Config.get_log_file(name))
        fh.setLevel(logging.DEBUG)
        
        # Console handler - shows important messages on screen
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter for consistent log format
        formatter = logging.Formatter(Config.LOG_FORMAT)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

# Initialize main logger
logger = setup_logging("utils")

# =============================================================================
# GPU/CPU DETECTION AND RESTRICTIONS
# =============================================================================

def check_gpu_availability():
    """
    Check GPU availability and return detailed status.
    
    Returns:
        dict: GPU availability status and details
    """
    gpu_status = {
        'torch_available': False,
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'total_memory': 0,
        'can_run_local_models': False
    }
    
    try:
        import torch
        gpu_status['torch_available'] = True
        
        if torch.cuda.is_available():
            gpu_status['cuda_available'] = True
            gpu_status['gpu_count'] = torch.cuda.device_count()
            
            for i in range(gpu_status['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                gpu_status['gpu_names'].append(props.name)
                gpu_status['total_memory'] += props.total_memory
            
            # Convert to GB for readability
            gpu_status['total_memory'] = gpu_status['total_memory'] / (1024**3)
            
            # Check if we have enough memory for local models (at least 6GB recommended)
            gpu_status['can_run_local_models'] = gpu_status['total_memory'] >= 6.0
            
    except ImportError:
        logger.warning("PyTorch not available - local models cannot be used")
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
    
    return gpu_status

def get_memory_status():
    """
    Get current memory status (RAM and GPU).
    
    Returns:
        dict: Memory usage information
    """
    memory_status = {
        'ram_total_gb': 0,
        'ram_available_gb': 0,
        'ram_percent_used': 0,
        'gpu_memory_used_gb': 0,
        'gpu_memory_total_gb': 0,
        'gpu_memory_percent_used': 0
    }
    
    try:
        import psutil
        
        # RAM information
        ram = psutil.virtual_memory()
        memory_status['ram_total_gb'] = ram.total / (1024**3)
        memory_status['ram_available_gb'] = ram.available / (1024**3)
        memory_status['ram_percent_used'] = ram.percent
        
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # GPU memory information
            memory_status['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_status['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if memory_status['gpu_memory_total_gb'] > 0:
                memory_status['gpu_memory_percent_used'] = (
                    memory_status['gpu_memory_used_gb'] / memory_status['gpu_memory_total_gb'] * 100
                )
    
    except Exception as e:
        logger.debug(f"Could not get GPU memory status: {e}")
    
    return memory_status

def print_system_status():
    """Print comprehensive system status including GPU and memory."""
    gpu_status = check_gpu_availability()
    memory_status = get_memory_status()
    
    print("\n=== SYSTEM STATUS ===")
    
    # GPU Status
    print(f"\n🖥️  Compute Status:")
    if gpu_status['cuda_available']:
        print(f"   ✅ GPU Available: {gpu_status['gpu_count']} device(s)")
        for i, name in enumerate(gpu_status['gpu_names']):
            print(f"      GPU {i}: {name}")
        print(f"   📊 Total GPU Memory: {gpu_status['total_memory']:.1f} GB")
        print(f"   🚀 Can run local models: {'Yes' if gpu_status['can_run_local_models'] else 'No (need ≥6GB)'}")
    else:
        if gpu_status['torch_available']:
            print(f"   ⚠️  CPU Only (PyTorch available, no CUDA)")
        else:
            print(f"   ❌ No PyTorch - local models unavailable")
    
    # Memory Status
    print(f"\n💾 Memory Status:")
    if memory_status['ram_total_gb'] > 0:
        print(f"   RAM: {memory_status['ram_available_gb']:.1f} GB available of {memory_status['ram_total_gb']:.1f} GB ({memory_status['ram_percent_used']:.1f}% used)")
    
    if memory_status['gpu_memory_total_gb'] > 0:
        print(f"   GPU: {memory_status['gpu_memory_used_gb']:.1f} GB used of {memory_status['gpu_memory_total_gb']:.1f} GB ({memory_status['gpu_memory_percent_used']:.1f}% used)")
    
    print()
    return gpu_status, memory_status

def validate_gpu_requirements_for_command(command: str, gpu_status: dict = None) -> bool:
    """
    Validate if a command can be run based on GPU requirements.
    
    Args:
        command: Command being executed
        gpu_status: GPU status dictionary (will check if None)
        
    Returns:
        bool: True if command can be run, False otherwise
    """
    if gpu_status is None:
        gpu_status = check_gpu_availability()
    
    # Commands that require local model capability (GPU or sufficient CPU)
    gpu_dependent_commands = {
        'run-experiment': 'Running experiments with local models requires GPU/sufficient resources'
    }
    
    # Commands that are always allowed regardless of GPU
    always_allowed = {
        'list-options', 'list-commands', 'status', 'cleanup', 
        'download-datasets', 'evaluate', 'plot'
    }
    
    if command in always_allowed:
        return True
    
    if command in gpu_dependent_commands:
        # For run-experiment, we need to be more specific about what's allowed
        return True  # We'll check this more specifically when model is being loaded
    
    return True  # Default: allow command

def clear_gpu_memory():
    """
    Clear GPU memory and run garbage collection.
    
    Important for freeing memory when switching between large models
    or when running multiple experiments sequentially.
    """
    logger.debug("Clearing GPU memory...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    logger.debug("GPU memory cleared")

def show_gpu_stats():
    """
    Display current GPU memory usage and device information.
    
    Returns:
        str: GPU status message
    """
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

# =============================================================================
# RANDOM SEED AND REPRODUCIBILITY
# =============================================================================

def set_random_seeds(seed: int = None):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (defaults to Config.RANDOM_SEED)
    """
    seed = seed or Config.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seeds to {seed}")

# =============================================================================
# SYSTEM INFORMATION AND DIAGNOSTICS
# =============================================================================

def get_system_info():
    """
    Collect comprehensive system information for debugging and logging.
    
    Returns:
        dict: System information including Python, PyTorch, CUDA, and directory info
    """
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
            'models_cache': Config.CACHED_MODELS_DIR,
            'finetuned_models': Config.FINETUNED_MODELS_DIR
        }
    }
    
    # Get GPU details if available
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

# =============================================================================
# CONFIGURATION AND API KEY VALIDATION
# =============================================================================

def validate_api_keys():
    """
    Check which API keys are available and log their status.
    
    Returns:
        dict: API key availability status for each provider
    """
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
    """
    Validate that configuration files exist and can be loaded properly.
    
    Returns:
        dict: Configuration validation results
    """
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

# =============================================================================
# EXPERIMENT CONTEXT MANAGEMENT
# =============================================================================

@contextmanager
def experiment_context(experiment_name: str):
    """
    Context manager for experiments with automatic cleanup and error handling.
    
    Provides consistent experiment lifecycle management with timing and cleanup.
    
    Args:
        experiment_name: Name of experiment for logging
        
    Yields:
        None: Context for experiment execution
    """
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

# =============================================================================
# LEGACY DATASET DOWNLOAD FUNCTION
# =============================================================================

def download_dataset(dataset_json: str, dataset_path: str):
    """
    Download datasets from JSON configuration (legacy function for compatibility).
    
    This function maintains compatibility with the existing download infrastructure
    used by the experiment runner.
    
    Args:
        dataset_json: Path to JSON file with dataset configurations
        dataset_path: Base path for dataset storage
    """
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
                    # Download with wget (requires wget to be installed)
                    subprocess.run(["wget", "-q", "-O", filename, dataset["link"]], 
                                 check=True, timeout=300)
                    
                    # Extract if it's a zip file
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

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_filename(name: str) -> str:
    """
    Create a filesystem-safe filename from any string.
    
    Args:
        name: Input string to make safe
        
    Returns:
        str: Safe filename with problematic characters removed
    """
    # Remove or replace problematic characters
    safe_chars = "".join(c for c in name if c.isalnum() or c in ('-', '_', '.'))
    # Ensure it's not empty and not too long
    if not safe_chars:
        safe_chars = "unnamed"
    return safe_chars[:100]  # Limit length

def load_json_config(filepath: str) -> Dict[str, Any]:
    """
    Load JSON configuration file with error handling.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        dict: Loaded configuration or empty dict if failed
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {filepath}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {filepath}: {e}")
        return {}

# =============================================================================
# DIRECTORY AND SYSTEM SETUP
# =============================================================================

def create_directory_structure():
    """
    Create all necessary directories for the system.
    
    Returns:
        list: List of created/verified directories
        
    Raises:
        Exception: If directory creation fails
    """
    logger.info("Creating directory structure...")
    
    try:
        created_dirs = Config.create_directories()
        logger.info(f"Created/verified {len(created_dirs)} directories")
        
        # Check configuration files and warn about missing ones
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
    """
    Check system requirements and dependencies.
    
    Returns:
        dict: Requirements status for different components
    """
    logger.info("Checking system requirements...")

    # Check torch/cuda availability with proper error handling
    torch_available = True
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_available = False
        cuda_available = False
        logger.error("PyTorch not available")
    except Exception as e:
        torch_available = True  # torch imported but cuda check failed
        cuda_available = False
        logger.warning(f"PyTorch available but CUDA check failed: {e}")

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
    """
    Initialize the system - create directories, set seeds, validate configs.
    
    Returns:
        dict: Initialization results and system status
    """
    logger.info("Initializing XAI explanation evaluation system...")
    
    # Set random seeds for reproducibility
    set_random_seeds()
    
    # Check system requirements
    requirements = check_system_requirements()
    
    # Get system info
    system_info = get_system_info()
    
    # Show GPU stats if available
    if torch.cuda.is_available():
        show_gpu_stats()
    
    # Log initialization results with status indicators
    status_indicators = {
        True: "✅",
        False: "❌"
    }
    
    logger.info(f"{status_indicators[requirements['python_version_ok']]} Python version: {sys.version.split()[0]}")
    if not requirements['python_version_ok']:
        logger.error("Python version must be 3.8 or higher")
    
    logger.info(f"{status_indicators[requirements['torch_available']]} PyTorch available")
    
    if requirements['cuda_available']:
        logger.info("✅ CUDA available")
    else:
        logger.info("ℹ️  CUDA not available (will use CPU)")
    
    logger.info(f"{status_indicators[requirements['directories_created']]} Directory structure")
    logger.info(f"{status_indicators[requirements['config_files_valid']]} Configuration files")
    
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