import os
from datetime import datetime

class Config:
    """Configuration management for XAI explanation evaluation system"""
    
    # Paths
    BASE_DIR = os.getcwd()
    DATA_DIR = "./datasets"
    RESULTS_DIR = "./results"
    MODELS_DIR = "./models"
    PLOTS_DIR = "./plots"
    CHECKPOINTS_DIR = "./checkpoints"
    LOGS_DIR = "./logs"
    
    # Dataset configuration
    DATASETS_JSON = "datasets.json"
    
    # Model configurations
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    
    # LLM parameters
    MAX_NEW_TOKENS = 256
    MAX_SEQ_LENGTH = 4096
    TEMPERATURE = 0.1
    
    # API Keys (should be set as environment variables in production)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")
    
    # Experiment parameters
    RANDOM_SEED = 42
    SAMPLE_SIZE = 100  # Default sample size for experiments
    
    # Evaluation parameters
    EVAL_BATCH_SIZE = 10
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR, cls.RESULTS_DIR, cls.MODELS_DIR, 
            cls.PLOTS_DIR, cls.CHECKPOINTS_DIR, cls.LOGS_DIR,
            os.path.join(cls.PLOTS_DIR, "individual_experiments"),
            os.path.join(cls.PLOTS_DIR, "comparative_analysis"),
            os.path.join(cls.PLOTS_DIR, "comprehensive_reports"),
            os.path.join(cls.PLOTS_DIR, "model_comparisons"),
            os.path.join(cls.PLOTS_DIR, "prompt_comparisons")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        return directories
    
    @classmethod
    def get_log_file(cls, name="main"):
        """Get log file path for a specific component"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return os.path.join(cls.LOGS_DIR, f"{name}_{timestamp}.log")
