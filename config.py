import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

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
    SETUPS_JSON = os.path.join(CONFIGS_DIR, "setups.json")
    MODELS_JSON = os.path.join(CONFIGS_DIR, "models.json")
    
    # Data and output directories - main storage locations
    DATA_DIR = "./datasets"          # Downloaded datasets
    OUTPUTS_DIR = "./outputs"        # All generated outputs
    
    # Output subdirectories organized by type
    RESPONSES_DIR = os.path.join(OUTPUTS_DIR, "responses")      # Model inference results
    EVALUATIONS_DIR = os.path.join(OUTPUTS_DIR, "evaluations") # Evaluation metrics
    PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")            # Visualizations
    
    # Cache and temporary directories
    CACHED_MODELS_DIR = "./cached_models"      # Downloaded/cached model files
    FINETUNED_MODELS_DIR = "./finetuned_models" # Custom finetuned models
    LOGS_DIR = "./logs"                        # System logs
    
    # =============================================================================
    # MODEL CONFIGURATION DEFAULTS
    # =============================================================================
    
    # Default embedding model for semantic similarity calculations
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    
    # LLM generation parameters (can be overridden per experiment and per model)
    MAX_NEW_TOKENS = 256           # Maximum tokens to generate per response
    MAX_SEQ_LENGTH = 4096          # Maximum sequence length for local models
    DEFAULT_TEMPERATURE = 0.1       # Default sampling temperature
    
    # =============================================================================
    # LOCAL MODEL LOADING DEFAULTS
    # =============================================================================
    
    # Unsloth/FastLanguageModel defaults
    DEFAULT_LOAD_IN_4BIT = False
    DEFAULT_TRUST_REMOTE_CODE = False
    DEFAULT_LOCAL_FILES_ONLY = False
    DEFAULT_USE_UNSLOTH = False  # Whether to prefer Unsloth over standard transformers
    
    # Standard Transformers defaults
    DEFAULT_TORCH_DTYPE = "auto"
    DEFAULT_DEVICE_MAP = "auto"
    DEFAULT_USE_CHAT_TEMPLATE = True
    
    # =============================================================================
    # REASONING MODEL DEFAULTS
    # =============================================================================
    
    # Reasoning/thinking model specific parameters
    DEFAULT_IS_REASONING_MODEL = False
    DEFAULT_THINKING_BUDGET = 512
    DEFAULT_TOKENIZE_CHAT_TEMPLATE = False  # Most models use False, reasoning models use True
    DEFAULT_DECODE_FULL_OUTPUT = False      # Reasoning models typically decode full output
    
    # =============================================================================
    # GENERATION PARAMETER DEFAULTS
    # =============================================================================
    
    # Core generation parameters
    DEFAULT_TOP_P = 0.9
    DEFAULT_TOP_K = 50
    DEFAULT_REPETITION_PENALTY = 1.0
    DEFAULT_DO_SAMPLE = True
    DEFAULT_NUM_BEAMS = 1
    DEFAULT_EARLY_STOPPING = False
    
    # Advanced generation parameters
    DEFAULT_LENGTH_PENALTY = 1.0
    DEFAULT_NO_REPEAT_NGRAM_SIZE = 0
    DEFAULT_ENCODER_NO_REPEAT_NGRAM_SIZE = 0
    DEFAULT_BAD_WORDS_IDS = None
    DEFAULT_FORCE_WORDS_IDS = None
    DEFAULT_RENORMALIZE_LOGITS = False
    DEFAULT_CONSTRAINTS = None
    DEFAULT_FORCED_BOS_TOKEN_ID = None
    DEFAULT_FORCED_EOS_TOKEN_ID = None
    DEFAULT_REMOVE_INVALID_VALUES = False
    DEFAULT_EXPONENTIAL_DECAY_LENGTH_PENALTY = None
    DEFAULT_SUPPRESS_TOKENS = None
    DEFAULT_BEGIN_SUPPRESS_TOKENS = None
    DEFAULT_FORCED_DECODER_IDS = None
    
    # Chat template defaults
    DEFAULT_CHAT_TEMPLATE_ROLE = "user"
    DEFAULT_ADD_GENERATION_PROMPT = True
    
    # =============================================================================
    # INFERENCE PIPELINE DEFAULTS
    # =============================================================================
    
    # Token handling
    DEFAULT_SKIP_SPECIAL_TOKENS = True
    DEFAULT_CLEAN_UP_TOKENIZATION_SPACES = True
    DEFAULT_TRUNCATION = True
    DEFAULT_MAX_LENGTH = 2048
    DEFAULT_PADDING = True
    DEFAULT_RETURN_TENSORS = "pt"
    
    # Output processing
    DEFAULT_STRIP_PROMPT_FROM_OUTPUT = True
    DEFAULT_RETURN_FULL_TEXT = False
    
    # =============================================================================
    # API CREDENTIALS
    # =============================================================================
    
    # API Keys loaded from environment variables
    # Users should set these in their .env file
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")
    
    # Hugging Face Access Token for restricted repositories
    HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN", "")
    
    # =============================================================================
    # EXPERIMENT PARAMETERS
    # =============================================================================
    
    # Reproducibility and defaults
    RANDOM_SEED = 42               # Fixed seed for reproducible experiments
    DEFAULT_SAMPLE_SIZE = 50       # Default number of samples per experiment
    
    # =============================================================================
    # LOGGING CONFIGURATION
    # =============================================================================
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # =============================================================================
    # HELPER METHODS FOR DEFAULTS
    # =============================================================================
    
    @classmethod
    def get_default_generation_config(cls) -> Dict[str, Any]:
        """
        Get default generation configuration dictionary.
        
        Returns:
            dict: Default generation parameters
        """
        return {
            'max_new_tokens': cls.MAX_NEW_TOKENS,
            'temperature': cls.DEFAULT_TEMPERATURE,
            'top_p': cls.DEFAULT_TOP_P,
            'top_k': cls.DEFAULT_TOP_K,
            'repetition_penalty': cls.DEFAULT_REPETITION_PENALTY,
            'do_sample': cls.DEFAULT_DO_SAMPLE,
            'num_beams': cls.DEFAULT_NUM_BEAMS,
            'early_stopping': cls.DEFAULT_EARLY_STOPPING,
            'length_penalty': cls.DEFAULT_LENGTH_PENALTY,
            'no_repeat_ngram_size': cls.DEFAULT_NO_REPEAT_NGRAM_SIZE,
            'pad_token_id': None,  # Will be set from tokenizer
            'eos_token_id': None,  # Will be set from tokenizer
            'skip_special_tokens': cls.DEFAULT_SKIP_SPECIAL_TOKENS,
            'clean_up_tokenization_spaces': cls.DEFAULT_CLEAN_UP_TOKENIZATION_SPACES
        }
    
    @classmethod
    def get_default_model_loading_config(cls) -> Dict[str, Any]:
        """
        Get default model loading configuration.
        
        Returns:
            dict: Default model loading parameters
        """
        return {
            'load_in_4bit': cls.DEFAULT_LOAD_IN_4BIT,
            'trust_remote_code': cls.DEFAULT_TRUST_REMOTE_CODE,
            'local_files_only': cls.DEFAULT_LOCAL_FILES_ONLY,
            'torch_dtype': cls.DEFAULT_TORCH_DTYPE,
            'device_map': cls.DEFAULT_DEVICE_MAP,
            'use_unsloth': cls.DEFAULT_USE_UNSLOTH,
            'use_chat_template': cls.DEFAULT_USE_CHAT_TEMPLATE,
            'is_reasoning_model': cls.DEFAULT_IS_REASONING_MODEL
        }
    
    @classmethod
    def get_default_reasoning_config(cls) -> Dict[str, Any]:
        """
        Get default reasoning model configuration.
        
        Returns:
            dict: Default reasoning model parameters
        """
        return {
            'thinking_budget': cls.DEFAULT_THINKING_BUDGET,
            'tokenize_chat_template': cls.DEFAULT_TOKENIZE_CHAT_TEMPLATE,
            'decode_full_output': cls.DEFAULT_DECODE_FULL_OUTPUT
        }
    
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
        
        Expected structure:
        {
          "prompt_name": {
            "compatible_setup": "setup_name",
            "mode": "zero-shot",
            "template": "...",
            "description": "..."
          }
        }
        
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
    def load_setups_config(cls) -> Dict[str, Any]:
        """
        Load setups configuration from JSON file.
        
        Expected structure:
        {
          "setup_name": {
            "description": "...",
            "dataset": {
              "download_link": "...",
              "download_path": "...",
              "csv_file": "..." OR "parquet_file": "..."
            },
            "prompt_fields": {
              "question_fields": [X, json:context.contexts[*], ...],
              "answer_field": "..."
            },
            "prune_row": { ... },  // Optional
            "custom_metrics": { ... }  // Optional
          }
        }
        
        Returns:
            dict: Setup configurations including dataset sources, paths, custom metrics, and row pruning rules
            
        Raises:
            FileNotFoundError: If setups.json doesn't exist
            ValueError: If JSON is malformed
        """
        try:
            with open(cls.SETUPS_JSON, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Setups configuration file not found: {cls.SETUPS_JSON}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in setups configuration: {e}")
    
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
    # EXPERIMENT NAMING AND FILE PATH GENERATION
    # =============================================================================
    
    @classmethod
    def generate_experiment_name(cls, setup: str, model: str, 
                                mode: str, prompt: str, size: int, temperature: float,
                                few_shot_row: Optional[int] = None) -> str:
        """
        Generate a standardized experiment name.
        
        Creates descriptive names that uniquely identify experiments:
        Format for zero-shot: {setup}_{model}_{mode}_{prompt}_{size}_{temperature}
        Format for few-shot: {setup}_{model}_{mode}-{few_shot_row}_{prompt}_{size}_{temperature}
        
        Args:
            setup: Setup name (e.g., 'gmeg')
            model: Model name (e.g., 'gpt-4o-mini')
            mode: Prompting mode ('zero-shot' or 'few-shot')
            prompt: Prompt name (e.g., 'gmeg_v1_basic')
            size: Sample size (e.g., 50)
            temperature: Generation temperature (e.g., 0.1)
            few_shot_row: Row number used for few-shot example (only for few-shot mode)
            
        Returns:
            str: Standardized experiment name
        """

        # Format temperature to avoid floating point precision issues
        # 0.1 becomes "0p1", 1.0 becomes "1p0"
        temp_str = f"{temperature:.3f}".replace(".", "p")
        
        # For few-shot experiments, include the row number used for the example
        if mode == 'few-shot' and few_shot_row is not None:
            mode_with_row = f"{mode}-{few_shot_row}"
        else:
            mode_with_row = mode

        return f"{setup}__{model}__{mode_with_row}__{prompt}__{size}__{temp_str}"
    
    @classmethod
    def generate_file_paths(cls, experiment_name: str) -> Dict[str, str]:
        """
        Generate complete file paths for all experiment outputs.
        
        Creates organized file paths for inference results, evaluations, and plots.
        
        Args:
            experiment_name: Generated experiment name
            
        Returns:
            dict: File paths for inference, evaluation, and plot outputs
        """
        return {
            'inference': os.path.join(cls.RESPONSES_DIR, f"inference_{experiment_name}.json"),
            'evaluation': os.path.join(cls.EVALUATIONS_DIR, f"evaluation_{experiment_name}.json"),
            'plot': os.path.join(cls.PLOTS_DIR, f"plot_{experiment_name}.html")
        }
    
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
            'setups': os.path.exists(cls.SETUPS_JSON),
            'models': os.path.exists(cls.MODELS_JSON)
        }
        return files