import os
import json
from datetime import datetime
from typing import Dict, Any

class Config:
    """
    Configuration management for XAI explanation evaluation system.
    
    This class centralizes all configuration constants, paths, and settings
    used throughout the system. It provides methods to load JSON configurations,
    create directory structures, and generate file paths for experiments.
    """
    
    # =============================================================================
    # BASE PATHS AND DIRECTORIES
    # =============================================================================
    
    BASE_DIR = os.getcwd()
    
    # Configuration directories - where JSON config files are stored
    CONFIGS_DIR = "./configs"
    PROMPTS_JSON = os.path.join(CONFIGS_DIR, "prompts.json")
    DATASETS_JSON = os.path.join(CONFIGS_DIR, "datasets.json")
    MODELS_JSON = os.path.join(CONFIGS_DIR, "models.json")
    
    # Data and output directories - main storage locations
    DATA_DIR = "./datasets"          # Downloaded datasets
    OUTPUTS_DIR = "./outputs"        # All generated outputs
    
    # Output subdirectories organized by type
    RESPONSES_DIR = os.path.join(OUTPUTS_DIR, "responses")      # Model inference results
    EVALUATIONS_DIR = os.path.join(OUTPUTS_DIR, "evaluations") # Evaluation metrics
    PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")            # Visualizations
    
    # Experiment type specific subdirectories
    # Each experiment type gets its own organized structure
    BASELINE_RESPONSES_DIR = os.path.join(RESPONSES_DIR, "baseline")
    BASELINE_EVALUATIONS_DIR = os.path.join(EVALUATIONS_DIR, "baseline")
    BASELINE_PLOTS_DIR = os.path.join(PLOTS_DIR, "baseline")
    
    # Cache and temporary directories - homogenous naming
    CACHED_MODELS_DIR = "./cached_models"      # Downloaded/cached model files (renamed for consistency)
    FINETUNED_MODELS_DIR = "./finetuned_models" # Custom finetuned models
    LOGS_DIR = "./logs"                        # System logs
    
    # =============================================================================
    # MODEL CONFIGURATION DEFAULTS
    # =============================================================================
    
    # Default embedding model for semantic similarity calculations
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    
    # LLM generation parameters (can be overridden per experiment)
    MAX_NEW_TOKENS = 256           # Maximum tokens to generate per response
    MAX_SEQ_LENGTH = 4096          # Maximum sequence length for local models
    DEFAULT_TEMPERATURE = 0.1       # Default sampling temperature
    
    # =============================================================================
    # API CREDENTIALS
    # =============================================================================
    
    # API Keys loaded from environment variables
    # Users should set these in their .env file
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")
    
    # =============================================================================
    # EXPERIMENT PARAMETERS
    # =============================================================================
    
    # Reproducibility and defaults
    RANDOM_SEED = 42               # Fixed seed for reproducible experiments
    DEFAULT_SAMPLE_SIZE = 50       # Default number of samples per experiment
    
    # Supported experiment types
    # Currently only baseline is implemented, but structure allows for extension
    EXPERIMENT_TYPES = ["baseline"]  # Future: "masked", "impersonation"
    
    # =============================================================================
    # LOGGING CONFIGURATION
    # =============================================================================
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # =============================================================================
    # DIRECTORY MANAGEMENT METHODS
    # =============================================================================
    
    @classmethod
    def create_directories(cls):
        """
        Create all necessary directories for the system.
        
        This ensures the complete directory structure exists before
        any operations that might try to write files.
        
        Returns:
            list: List of created/verified directory paths
        """
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
            cls.CACHED_MODELS_DIR,
            cls.FINETUNED_MODELS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        return directories
    
    # =============================================================================
    # CONFIGURATION LOADING METHODS
    # =============================================================================
    
    @classmethod
    def load_prompts_config(cls) -> Dict[str, Any]:
        """
        Load prompts configuration from JSON file.
        
        Returns:
            dict: Prompt templates and metadata
            
        Raises:
            FileNotFoundError: If prompts.json doesn't exist
            ValueError: If JSON is malformed
        """
        try:
            with open(cls.PROMPTS_JSON, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts configuration file not found: {cls.PROMPTS_JSON}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompts configuration: {e}")
    
    @classmethod
    def load_datasets_config(cls) -> Dict[str, Any]:
        """
        Load datasets configuration from JSON file.
        
        Returns:
            dict: Dataset sources, paths, and metadata
            
        Raises:
            FileNotFoundError: If datasets.json doesn't exist
            ValueError: If JSON is malformed
        """
        try:
            with open(cls.DATASETS_JSON, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Datasets configuration file not found: {cls.DATASETS_JSON}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in datasets configuration: {e}")
    
    @classmethod
    def load_models_config(cls) -> Dict[str, Any]:
        """
        Load models configuration from JSON file.
        
        Returns:
            dict: Model definitions, parameters, and metadata
            
        Raises:
            FileNotFoundError: If models.json doesn't exist
            ValueError: If JSON is malformed
        """
        try:
            with open(cls.MODELS_JSON, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Models configuration file not found: {cls.MODELS_JSON}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in models configuration: {e}")
    
    # =============================================================================
    # EXPERIMENT TYPE VALIDATION
    # =============================================================================
    
    @classmethod
    def validate_experiment_type(cls, experiment_type: str) -> bool:
        """
        Validate that an experiment type is supported.
        
        Args:
            experiment_type: Type to validate
            
        Returns:
            bool: True if supported, False otherwise
        """
        return experiment_type in cls.EXPERIMENT_TYPES
    
    @classmethod
    def get_output_dirs_for_experiment_type(cls, experiment_type: str) -> Dict[str, str]:
        """
        Get output directory paths for a specific experiment type.
        
        Args:
            experiment_type: Type of experiment
            
        Returns:
            dict: Directory paths for responses, evaluations, and plots
            
        Raises:
            ValueError: If experiment type is not supported
        """
        if not cls.validate_experiment_type(experiment_type):
            raise ValueError(f"Unsupported experiment type: {experiment_type}. Supported: {cls.EXPERIMENT_TYPES}")
        
        return {
            'responses': os.path.join(cls.RESPONSES_DIR, experiment_type),
            'evaluations': os.path.join(cls.EVALUATIONS_DIR, experiment_type),
            'plots': os.path.join(cls.PLOTS_DIR, experiment_type)
        }
    
    # =============================================================================
    # EXPERIMENT NAMING AND FILE PATH GENERATION
    # =============================================================================
    
    @classmethod
    def generate_experiment_name(cls, experiment_type: str, dataset: str, model: str, 
                                mode: str, prompt: str, size: int, temperature: float) -> str:
        """
        Generate a standardized experiment name.
        
        Creates descriptive names that uniquely identify experiments:
        Format: {experiment_type}_{dataset}_{model}_{mode}_{prompt}_{size}_{temperature}
        
        Args:
            experiment_type: Type of experiment (e.g., 'baseline')
            dataset: Dataset name (e.g., 'gmeg')
            model: Model name (e.g., 'gpt-4o-mini')
            mode: Prompting mode ('zero-shot' or 'few-shot')
            prompt: Prompt name (e.g., 'gmeg_v1_basic')
            size: Sample size (e.g., 50)
            temperature: Generation temperature (e.g., 0.1)
            
        Returns:
            str: Standardized experiment name
        """
        # Format temperature to avoid floating point precision issues
        # 0.1 becomes "0p1", 1.0 becomes "1p0"
        temp_str = f"{temperature:.1f}".replace(".", "p")
        return f"{experiment_type}_{dataset}_{model}_{mode}_{prompt}_{size}_{temp_str}"
    
    @classmethod
    def generate_file_paths(cls, experiment_type: str, experiment_name: str) -> Dict[str, str]:
        """
        Generate complete file paths for all experiment outputs.
        
        Creates organized file paths for inference results, evaluations, and plots.
        
        Args:
            experiment_type: Type of experiment
            experiment_name: Generated experiment name (with or without mode)
            
        Returns:
            dict: File paths for inference, evaluation, and plot outputs
        """
        dirs = cls.get_output_dirs_for_experiment_type(experiment_type)
        
        return {
            'inference': os.path.join(dirs['responses'], f"inference_{experiment_name}.json"),
            'evaluation': os.path.join(dirs['evaluations'], f"evaluation_{experiment_name}.json"),
            'plot': os.path.join(dirs['plots'], f"plot_{experiment_name}.html")
        }
    
    @classmethod
    def extract_experiment_type_from_name(cls, experiment_name: str) -> str:
        """
        Extract experiment type from experiment name.
        
        Args:
            experiment_name: Full experiment name
            
        Returns:
            str: Experiment type (e.g., 'baseline')
        """
        if not experiment_name:
            raise ValueError("Experiment name cannot be empty")
        
        name_parts = experiment_name.split('_')
        if not name_parts:
            raise ValueError("Invalid experiment name format")
        
        # First part should be the experiment type
        potential_type = name_parts[0]
        
        if potential_type in cls.EXPERIMENT_TYPES:
            return potential_type
        
        raise ValueError(f"Experiment type '{potential_type}' not recognized. Supported types: {cls.EXPERIMENT_TYPES}")
        
    
    @classmethod 
    def extract_mode_from_name(cls, experiment_name: str) -> str:
        """
        Extract mode from experiment name by substring search.
        
        Args:
            experiment_name: Full experiment name
            
        Returns:
            str: Mode ('zero-shot', 'few-shot', or 'unknown')
        """
        if not experiment_name:
            raise ValueError("Experiment name cannot be empty")
            
        if "zero-shot" in experiment_name:
            return "zero-shot"
        if "few-shot" in experiment_name:
            return "few-shot"
        
        raise ValueError("Mode not found in experiment name. Expected 'zero-shot' or 'few-shot'.")
    
    # =============================================================================
    # LOGGING UTILITIES
    # =============================================================================
    
    @classmethod
    def get_log_file(cls, name="main"):
        """
        Generate log file path for a specific component.
        
        Creates daily log files with component names for easy debugging.
        
        Args:
            name: Component name for the log file
            
        Returns:
            str: Path to log file
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        return os.path.join(cls.LOGS_DIR, f"{name}_{timestamp}.log")
    
    # =============================================================================
    # CONFIGURATION VALIDATION
    # =============================================================================
    
    @classmethod
    def validate_configuration_files(cls) -> Dict[str, bool]:
        """
        Validate that all required configuration files exist.
        
        Returns:
            dict: Existence status for each configuration file
        """
        files = {
            'prompts': os.path.exists(cls.PROMPTS_JSON),
            'datasets': os.path.exists(cls.DATASETS_JSON),
            'models': os.path.exists(cls.MODELS_JSON)
        }
        return files