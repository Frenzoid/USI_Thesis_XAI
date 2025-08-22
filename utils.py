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
    logger.info("Clearing GPU memory...")
    
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

def save_results(results: Dict, experiment_name: str, timestamp: str = None) -> str:
    """
    Save experiment results with timestamp
    
    Args:
        results: Dictionary containing experiment results
        experiment_name: Name for the experiment
        timestamp: Timestamp string (if None, will generate current timestamp)
    
    Returns:
        str: Path to saved file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure results directory exists
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Clean experiment name for filename
    clean_name = "".join(c for c in experiment_name if c.isalnum() or c in ('-', '_')).rstrip()
    filename = f"{clean_name}_{timestamp}.json"
    filepath = os.path.join(Config.RESULTS_DIR, filename)
    
    # Create a copy of results to avoid modifying original
    results_to_save = dict(results)
    
    # Add save metadata
    results_to_save['save_metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'saved_timestamp': timestamp,
        'filename': filename,
        'filepath': filepath
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving results to {filepath}: {e}")
        # Try alternative filename if there's an issue
        alt_filename = f"experiment_result_{timestamp}.json"
        alt_filepath = os.path.join(Config.RESULTS_DIR, alt_filename)
        
        try:
            with open(alt_filepath, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
            logger.info(f"Results saved to alternative path: {alt_filepath}")
            return alt_filepath
        except Exception as e2:
            logger.critical(f"Critical error: Cannot save results anywhere: {e2}")
            return None

def load_results(filepath: str) -> Dict:
    """Load saved experiment results"""
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded results from: {filepath}")
        return results
    except Exception as e:
        logger.error(f"Error loading results from {filepath}: {e}")
        return {}

def safe_filename(name: str) -> str:
    """Create a safe filename from any string"""
    # Remove or replace problematic characters
    safe_chars = "".join(c for c in name if c.isalnum() or c in ('-', '_', '.'))
    # Ensure it's not empty and not too long
    if not safe_chars:
        safe_chars = "unnamed"
    return safe_chars[:100]  # Limit length

@contextmanager
def experiment_context(experiment_name: str):
    """Context manager for experiments with cleanup and error handling"""
    logger.info(f"Starting experiment: {experiment_name}")
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
        raise  # Re-raise the exception
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
    """Download datasets from JSON configuration"""
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
                                 check=True, timeout=300)  # 5 minute timeout
                    
                    if filename.endswith('.zip'):
                        logger.info(f"Extracting {filename}...")
                        subprocess.run(["unzip", "-q", filename, "-d", os.path.dirname(filename)], 
                                     check=True, timeout=120)  # 2 minute timeout
                    
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
            'data': Config.DATA_DIR,
            'results': Config.RESULTS_DIR,
            'plots': Config.PLOTS_DIR,
            'models': Config.MODELS_DIR
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

# Initialize system on import
def initialize_system():
    """Initialize the system - create directories, set seeds, etc."""
    logger.info("Initializing XAI explanation evaluation system...")
    
    # Create directories
    created_dirs = Config.create_directories()
    logger.info(f"Created/verified {len(created_dirs)} directories")
    
    # Set random seeds
    set_random_seeds()
    
    # Show system info
    system_info = get_system_info()
    show_gpu_stats()
    
    # Validate API keys
    api_status = validate_api_keys()
    
    logger.info("System initialization complete")
    
    return {
        'directories': created_dirs,
        'system_info': system_info,
        'api_status': api_status
    }

# Run initialization
if __name__ != "__main__":
    initialize_system()
