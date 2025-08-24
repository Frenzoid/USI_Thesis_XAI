import os
import json
from datetime import datetime
from typing import Dict, Any

class Config:
    """Configuration management for XAI explanation evaluation system"""
    
    # Base paths
    BASE_DIR = os.getcwd()
    
    # Configuration directories
    CONFIGS_DIR = "./configs"
    PROMPTS_JSON = os.path.join(CONFIGS_DIR, "prompts.json")
    DATASETS_JSON = os.path.join(CONFIGS_DIR, "datasets.json")
    MODELS_JSON = os.path.join(CONFIGS_DIR, "models.json")
    
    # Data and output directories
    DATA_DIR = "./datasets"
    OUTPUTS_DIR = "./outputs"
    
    # Output subdirectories by experiment type
    RESPONSES_DIR = os.path.join(OUTPUTS_DIR, "responses")
    EVALUATIONS_DIR = os.path.join(OUTPUTS_DIR, "evaluations")
    PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
    
    # Experiment type subdirectories
    BASELINE_RESPONSES_DIR = os.path.join(RESPONSES_DIR, "baseline")
    BASELINE_EVALUATIONS_DIR = os.path.join(EVALUATIONS_DIR, "baseline")
    BASELINE_PLOTS_DIR = os.path.join(PLOTS_DIR, "baseline")
    
    # Cache and temporary directories
    MODELS_CACHE_DIR = "./models_cache"
    FINETUNED_MODELS_DIR = "./finetuned_models"
    LOGS_DIR = "./logs"
    
    # Model configurations
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    
    # LLM parameters (defaults - can be overridden)
    MAX_NEW_TOKENS = 256
    MAX_SEQ_LENGTH = 4096
    DEFAULT_TEMPERATURE = 0.1
    
    # API Keys (loaded from environment variables)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")
    
    # Experiment parameters
    RANDOM_SEED = 42
    DEFAULT_SAMPLE_SIZE = 50
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Supported experiment types
    EXPERIMENT_TYPES = ["baseline"]  # Will expand later with "masked", "impersonation"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.CONFIGS_DIR,
            cls.DATA_DIR,
            cls.OUTPUTS_DIR,
            cls.RESPONSES_DIR,
            cls.EVALUATIONS_DIR,
            cls.PLOTS_DIR,
            cls.BASELINE_RESPONSES_DIR,
            cls.BASELINE_EVALUATIONS_DIR,
            cls.BASELINE_PLOTS_DIR,
            cls.MODELS_CACHE_DIR,
            cls.FINETUNED_MODELS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        return directories
    
    @classmethod
    def load_prompts_config(cls) -> Dict[str, Any]:
        """Load prompts configuration from JSON file"""
        try:
            with open(cls.PROMPTS_JSON, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts configuration file not found: {cls.PROMPTS_JSON}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompts configuration: {e}")
    
    @classmethod
    def load_datasets_config(cls) -> Dict[str, Any]:
        """Load datasets configuration from JSON file"""
        try:
            with open(cls.DATASETS_JSON, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Datasets configuration file not found: {cls.DATASETS_JSON}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in datasets configuration: {e}")
    
    @classmethod
    def load_models_config(cls) -> Dict[str, Any]:
        """Load models configuration from JSON file"""
        try:
            with open(cls.MODELS_JSON, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Models configuration file not found: {cls.MODELS_JSON}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in models configuration: {e}")
    
    @classmethod
    def validate_experiment_type(cls, experiment_type: str) -> bool:
        """Validate experiment type is supported"""
        return experiment_type in cls.EXPERIMENT_TYPES
    
    @classmethod
    def get_output_dirs_for_experiment_type(cls, experiment_type: str) -> Dict[str, str]:
        """Get output directories for a specific experiment type"""
        if not cls.validate_experiment_type(experiment_type):
            raise ValueError(f"Unsupported experiment type: {experiment_type}. Supported: {cls.EXPERIMENT_TYPES}")
        
        return {
            'responses': os.path.join(cls.RESPONSES_DIR, experiment_type),
            'evaluations': os.path.join(cls.EVALUATIONS_DIR, experiment_type),
            'plots': os.path.join(cls.PLOTS_DIR, experiment_type)
        }
    
    @classmethod
    def generate_experiment_name(cls, experiment_type: str, dataset: str, model: str, 
                                prompt: str, size: int, temperature: float) -> str:
        """Generate experiment name based on configuration"""
        # Clean temperature to avoid floating point precision issues
        temp_str = f"{temperature:.1f}".replace(".", "p")
        return f"{experiment_type}_{dataset}_{model}_{prompt}_{size}_{temp_str}"
    
    @classmethod
    def generate_file_paths(cls, experiment_type: str, experiment_name: str) -> Dict[str, str]:
        """Generate file paths for experiment outputs"""
        dirs = cls.get_output_dirs_for_experiment_type(experiment_type)
        
        return {
            'inference': os.path.join(dirs['responses'], f"inference_{experiment_name}.json"),
            'evaluation': os.path.join(dirs['evaluations'], f"evaluation_{experiment_name}.json"),
            'plot': os.path.join(dirs['plots'], f"plot_{experiment_name}.html")
        }
    
    @classmethod
    def get_log_file(cls, name="main"):
        """Get log file path for a specific component"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return os.path.join(cls.LOGS_DIR, f"{name}_{timestamp}.log")
    
    @classmethod
    def validate_configuration_files(cls) -> Dict[str, bool]:
        """Validate that all required configuration files exist"""
        files = {
            'prompts': os.path.exists(cls.PROMPTS_JSON),
            'datasets': os.path.exists(cls.DATASETS_JSON),
            'models': os.path.exists(cls.MODELS_JSON)
        }
        return files