import torch
import os
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
try:
    import google.generativeai as genai
    # For newer Google GenAI SDK
    genai_available = True
except ImportError:
    try:
        # Fallback for older SDK structure
        from google import genai
        from google.genai import types
        genai_available = True
    except ImportError:
        genai_available = False
        logger.warning("Google GenAI SDK not available")

from config import Config
from utils import setup_logging, clear_gpu_memory, show_gpu_stats, check_gpu_availability, get_memory_status

logger = setup_logging("models")

class ModelManager:
    """
    Centralized model management system with JSON configuration support.
    
    Handles three types of models:
    1. Local models via Unsloth (open-source LLMs)
    2. API models (OpenAI, Google, Anthropic)
    3. Embedding models for similarity calculations
    """
    
    def __init__(self):
        """Initialize model manager with empty state and check GPU availability"""
        # Local model state
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.embedding_model = None
        
        # API clients
        self.openai_client = None
        self.genai_client = None
        self.anthropic_client = None
        
        # System status
        self.gpu_status = check_gpu_availability()
        
        # Load model configurations from JSON
        self.models_config = Config.load_models_config()
        
        # Check if Unsloth is available for local model support
        self.unsloth_available = self._check_unsloth_availability()
        
        # Log system capabilities
        self._log_system_capabilities()
        
        logger.info("ModelManager initialized")
    
    def _log_system_capabilities(self):
        """Log what the system can and cannot do based on available hardware"""
        if self.gpu_status['cuda_available']:
            logger.info(f"🚀 GPU detected: {self.gpu_status['gpu_count']} device(s) with {self.gpu_status['total_memory']:.1f} GB total memory")
            if self.gpu_status['can_run_local_models']:
                logger.info("✅ System can run local models")
            else:
                logger.warning("⚠️  GPU memory may be insufficient for larger local models")
        else:
            if self.gpu_status['torch_available']:
                logger.warning("⚠️  CPU-only mode detected - local models will be slow and may require significant RAM")
            else:
                logger.warning("❌ No PyTorch detected - local models unavailable")
        
        # Count available models by type
        local_models = sum(1 for m in self.models_config.values() if m['type'] == 'local')
        api_models = sum(1 for m in self.models_config.values() if m['type'] == 'api')
        
        logger.info(f"📊 Models configured: {local_models} local, {api_models} API-based")
    
    def _check_unsloth_availability(self):
        """
        Check if Unsloth library is available for efficient local model loading.
        
        Unsloth provides optimized loading and inference for open-source models.
        Without it, local model experiments cannot be run.
        
        Returns:
            bool: True if Unsloth is available, False otherwise
        """
        try:
            from unsloth import FastLanguageModel, is_bfloat16_supported
            from trl import SFTTrainer
            from transformers import TrainingArguments
            logger.info("Unsloth available for open-source model training")
            return True
        except ImportError:
            logger.warning("Unsloth not available. Local model experiments will be disabled.")
            return False
    
    def cleanup_current_model(self):
        """
        Clean up currently loaded model to free GPU/CPU memory.
        
        Important for systems with limited memory when switching between models.
        """
        logger.info("Cleaning up current model...")
        
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
        
        if self.current_tokenizer is not None:
            del self.current_tokenizer
            self.current_tokenizer = None
        
        self.current_model_name = None
        clear_gpu_memory()
        logger.info("Model cleanup completed")
    
    def load_embedding_model(self):
        """
        Load embedding model for semantic similarity calculations.
        
        Uses sentence-transformers for high-quality embeddings.
        Only loads once per session for efficiency.
        
        Returns:
            HuggingFaceEmbeddings: Loaded embedding model
        """
        if self.embedding_model is None:
            logger.info(f"🔄 Loading embedding model: {Config.EMBEDDING_MODEL}")
            
            # Show memory status before loading
            memory_before = get_memory_status()
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                cache_folder=Config.CACHED_MODELS_DIR  # Use consistent cache directory
            )
            
            logger.info("✅ Embedding model loaded successfully")
            
            # Show memory status after loading
            memory_after = get_memory_status()
            if memory_before['ram_total_gb'] > 0 and memory_after['ram_total_gb'] > 0:
                ram_used = memory_before['ram_available_gb'] - memory_after['ram_available_gb']
                logger.info(f"   📈 Embedding model used ~{ram_used:.1f} GB RAM")
            
        else:
            logger.info("✅ Embedding model already loaded")
        
        return self.embedding_model
    
    def load_open_source_model(self, model_name: str, model_path: str):
        """
        Load an open source model using Unsloth for efficient inference.
        
        Unsloth provides 4-bit quantization and other optimizations for
        running large models on consumer hardware.
        
        Args:
            model_name: Name from models.json configuration
            model_path: Hugging Face model path or local path
            
        Raises:
            ImportError: If Unsloth is not available
            ValueError: If model configuration is invalid
            RuntimeError: If GPU requirements are not met
        """
        # Check GPU requirements first
        if not self.gpu_status['cuda_available'] and not self.gpu_status['torch_available']:
            raise RuntimeError("Cannot load local models: PyTorch not available")
        
        if not self.gpu_status['cuda_available']:
            logger.warning("⚠️  Loading local model on CPU - this will be very slow and may require >16GB RAM")
            response = input("Continue loading local model on CPU? (y/N): ")
            if response.lower() != 'y':
                raise RuntimeError("User cancelled CPU-only model loading")
        
        if not self.unsloth_available:
            raise ImportError("Unsloth not available. Cannot load open source models.")
        
        if model_name not in self.models_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.models_config[model_name]
        
        if model_config['type'] != 'local':
            raise ValueError(f"Model {model_name} is not a local model")
        
        # Show memory status before loading
        logger.info("📊 Memory status before model loading:")
        memory_before = get_memory_status()
        if memory_before['ram_total_gb'] > 0:
            logger.info(f"   RAM: {memory_before['ram_available_gb']:.1f} GB available")
        if memory_before['gpu_memory_total_gb'] > 0:
            logger.info(f"   GPU: {memory_before['gpu_memory_used_gb']:.1f} GB / {memory_before['gpu_memory_total_gb']:.1f} GB used")
        
        # Clean up any existing model first
        self.cleanup_current_model()
        
        logger.info(f"🔄 Loading open-source model: {model_path}")
        
        try:
            from unsloth import FastLanguageModel
            
            # Load with optimizations for inference
            self.current_model, self.current_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=Config.MAX_SEQ_LENGTH,
                load_in_4bit=True,  # 4-bit quantization for memory efficiency
                cache_dir=Config.CACHED_MODELS_DIR  # Updated to use new consistent naming
            )
            
            # Set for inference mode (disables training-specific features)
            FastLanguageModel.for_inference(self.current_model)
            
            self.current_model_name = model_name
            logger.info(f"✅ Successfully loaded model: {model_name}")
            
            # Show memory status after loading
            logger.info("📊 Memory status after model loading:")
            memory_after = get_memory_status()
            show_gpu_stats()
            
            if memory_after['ram_total_gb'] > 0:
                ram_used = memory_before['ram_available_gb'] - memory_after['ram_available_gb']
                logger.info(f"   📈 Model used ~{ram_used:.1f} GB RAM")
            
            if memory_after['gpu_memory_total_gb'] > 0:
                gpu_used = memory_after['gpu_memory_used_gb'] - memory_before['gpu_memory_used_gb']
                logger.info(f"   📈 Model used ~{gpu_used:.1f} GB GPU memory")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise
    
    def load_finetuned_model(self, model_path: str):
        """
        Load a finetuned model from local filesystem.
        
        Finetuned models are stored in the finetuned_models directory
        and loaded with the same optimizations as base models.
        
        Args:
            model_path: Path to finetuned model directory
            
        Raises:
            ImportError: If Unsloth is not available
        """
        if not self.unsloth_available:
            raise ImportError("Unsloth not available. Cannot load finetuned models.")
        
        # Clean up any existing model first
        self.cleanup_current_model()
        
        logger.info(f"Loading finetuned model from: {model_path}")
        
        try:
            from unsloth import FastLanguageModel
            
            # Load finetuned model with same optimizations
            self.current_model, self.current_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                cache_dir=Config.CACHED_MODELS_DIR,  # Updated to use new consistent naming
                local_files_only=True,  # Only load from local files
                max_seq_length=Config.MAX_SEQ_LENGTH,
                load_in_4bit=True
            )
            
            # Set for inference mode
            FastLanguageModel.for_inference(self.current_model)
            
            self.current_model_name = f"finetuned_{os.path.basename(model_path)}"
            logger.info(f"Successfully loaded finetuned model from: {model_path}")
            show_gpu_stats()
            
        except Exception as e:
            logger.error(f"Failed to load finetuned model from {model_path}: {e}")
            raise
    
    def setup_api_clients(self):
        """
        Initialize API clients for external model providers.
        
        Only initializes clients for which API keys are available.
        Gracefully handles missing keys or initialization failures.
        """
        logger.info("Setting up API clients...")
        
        # OpenAI client setup
        if Config.OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Google GenAI client setup
        if Config.GENAI_API_KEY and genai_available:
            try:
                # Configure the API key for the newer SDK
                genai.configure(api_key=Config.GENAI_API_KEY)
                self.genai_client = genai  # Use module directly
                logger.info("Google GenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize GenAI client: {e}")
        
        # Anthropic client setup
        if Config.ANTHROPIC_API_KEY:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic library not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
    
    def query_open_source(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """
        Query currently loaded open source model for text generation.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            
        Returns:
            str: Generated text (with prompt removed)
            
        Raises:
            ValueError: If no model is loaded
        """
        if self.current_model is None or self.current_tokenizer is None:
            raise ValueError("No open source model loaded")
        
        max_tokens = max_tokens or Config.MAX_NEW_TOKENS
        temperature = temperature or Config.DEFAULT_TEMPERATURE
        
        logger.debug(f"Querying open-source model with prompt length: {len(prompt)}")
        
        try:
            # Tokenize input
            inputs = self.current_tokenizer(prompt, return_tensors="pt").to(self.current_model.device)
            
            # Generate with no gradients for efficiency
            with torch.no_grad():
                output = self.current_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,  # Use sampling only if temperature > 0
                    pad_token_id=self.current_tokenizer.eos_token_id
                )
            
            # Decode and clean response
            decoded = self.current_tokenizer.decode(output[0], skip_special_tokens=True)
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):].strip()
            
            logger.debug(f"Generated response length: {len(decoded)}")
            return decoded
            
        except Exception as e:
            logger.error(f"Error querying open-source model: {e}")
            raise
    
    def query_openai(self, prompt: str, model: str = 'gpt-4o-mini', max_tokens: int = None, temperature: float = None) -> str:
        """
        Query OpenAI API for text generation.
        
        Args:
            prompt: Input text prompt
            model: OpenAI model name (e.g., 'gpt-4o-mini')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            str: Generated text
            
        Raises:
            ValueError: If client is not initialized
        """
        if self.openai_client is None:
            raise ValueError("OpenAI client not initialized. Check API key.")
        
        max_tokens = max_tokens or Config.MAX_NEW_TOKENS
        temperature = temperature or Config.DEFAULT_TEMPERATURE
        
        logger.debug(f"Querying OpenAI model {model} with prompt length: {len(prompt)}")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            result = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI response length: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Error querying OpenAI model {model}: {e}")
            raise
    
    def query_genai(self, prompt: str, model: str = 'gemini-1.5-flash', max_tokens: int = None, temperature: float = None) -> str:
        """
        Query Google GenAI API for text generation.
        
        Args:
            prompt: Input text prompt
            model: Gemini model name (e.g., 'gemini-1.5-flash')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            str: Generated text
            
        Raises:
            ValueError: If client is not initialized
        """
        if self.genai_client is None:
            raise ValueError("GenAI client not initialized. Check API key.")
        
        temperature = temperature or Config.DEFAULT_TEMPERATURE
        max_tokens = max_tokens or Config.MAX_NEW_TOKENS
        
        logger.debug(f"Querying GenAI model {model} with prompt length: {len(prompt)}")
        
        try:
            # Use the newer Google GenAI SDK format
            model_instance = self.genai_client.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            result = response.text.strip()
            logger.debug(f"GenAI response length: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Error querying GenAI model {model}: {e}")
            raise
    
    def query_anthropic(self, prompt: str, model: str = 'claude-3-5-sonnet-20241022', max_tokens: int = None, temperature: float = None) -> str:
        """
        Query Anthropic Claude API for text generation.
        
        Args:
            prompt: Input text prompt
            model: Claude model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            str: Generated text
            
        Raises:
            ValueError: If client is not initialized
        """
        if self.anthropic_client is None:
            raise ValueError("Anthropic client not initialized. Check API key.")
        
        max_tokens = max_tokens or Config.MAX_NEW_TOKENS
        temperature = temperature or Config.DEFAULT_TEMPERATURE
        
        logger.debug(f"Querying Anthropic model {model} with prompt length: {len(prompt)}")
        
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.content[0].text.strip()
            logger.debug(f"Anthropic response length: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Error querying Anthropic model {model}: {e}")
            raise
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available for use.
        
        For local models: checks Unsloth availability and finetuned model existence
        For API models: checks if the appropriate client is initialized
        
        Args:
            model_name: Name of model to check
            
        Returns:
            bool: True if model can be used, False otherwise
        """
        if model_name not in self.models_config:
            return False
        
        model_config = self.models_config[model_name]
        
        if model_config['type'] == 'local':
            # Check Unsloth availability first
            if not self.unsloth_available:
                return False
            
            # For finetuned models, check if the finetuned version exists
            if model_config.get('finetuned', False):
                model_path = os.path.join(Config.FINETUNED_MODELS_DIR, 
                                          model_config['model_path'].split('/')[-1] + '_finetuned')
                return os.path.exists(model_path)
            
            return True
            
        elif model_config['type'] == 'api':
            # Check if appropriate API client is available
            provider = model_config['provider']
            
            if provider == 'openai':
                return self.openai_client is not None
            elif provider == 'google':
                return self.genai_client is not None
            elif provider == 'anthropic':
                return self.anthropic_client is not None
            
        return False