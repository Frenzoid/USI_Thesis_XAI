import torch
import os
from typing import Optional, Dict, Any, List, Union, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import google.generativeai as genai

from config import Config
from utils import setup_logging, clear_gpu_memory, show_gpu_stats, check_gpu_availability, get_memory_status

logger = setup_logging("models")

# =============================================================================
# UTILITY CLASSES FOR BETTER ORGANIZATION
# =============================================================================

class MemoryMonitor:
    """Utility class for consistent memory usage monitoring"""
    
    @staticmethod
    def log_memory_usage(operation_name: str, memory_before: dict):
        """Log memory usage for an operation"""
        memory_after = get_memory_status()
        
        if memory_after['ram_total_gb'] > 0 and memory_before['ram_total_gb'] > 0:
            ram_used = memory_before['ram_available_gb'] - memory_after['ram_available_gb']
            logger.info(f"{operation_name} used ~{ram_used:.1f} GB RAM")
        
        if memory_after['gpu_memory_total_gb'] > 0 and memory_before['gpu_memory_total_gb'] > 0:
            gpu_used = memory_after['gpu_memory_used_gb'] - memory_before['gpu_memory_used_gb']
            logger.info(f"{operation_name} used ~{gpu_used:.1f} GB GPU memory")
    
    @staticmethod
    def get_memory_before_operation() -> dict:
        """Get memory status before an operation"""
        memory_before = get_memory_status()
        if memory_before['ram_total_gb'] > 0:
            logger.info(f"   RAM: {memory_before['ram_available_gb']:.1f} GB available")
        if memory_before['gpu_memory_total_gb'] > 0:
            logger.info(f"   GPU: {memory_before['gpu_memory_used_gb']:.1f} GB / {memory_before['gpu_memory_total_gb']:.1f} GB used")
        return memory_before


class ConfigurationManager:
    """Handles model configuration validation and normalization"""
    
    @staticmethod
    def validate_and_normalize_config(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize model configuration with comprehensive defaults.
        
        Args:
            model_name: Name of the model
            model_config: Raw model configuration from JSON
            
        Returns:
            dict: Validated and normalized configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not model_config:
            raise ValueError(f"Empty configuration for model: {model_name}")
        
        # Validate required fields
        ConfigurationManager._validate_required_fields(model_name, model_config)
        
        # Create normalized configuration with defaults
        normalized = model_config.copy()
        
        # Set system-level defaults first
        defaults = Config.get_default_model_loading_config()
        for key, value in defaults.items():
            normalized.setdefault(key, value)
        
        # Set model-specific defaults
        ConfigurationManager._set_model_specific_defaults(normalized)
        
        # Normalize sub-configurations
        ConfigurationManager._normalize_reasoning_config(normalized)
        ConfigurationManager._normalize_generation_config(normalized)
        
        return normalized
    
    @staticmethod
    def _validate_required_fields(model_name: str, model_config: Dict[str, Any]):
        """Validate required fields for model configuration"""
        if 'type' not in model_config:
            raise ValueError(f"Model {model_name} missing required 'type' field")
        
        if model_config['type'] == 'local' and 'model_path' not in model_config:
            raise ValueError(f"Local model {model_name} missing required 'model_path' field")
        
        if model_config['type'] == 'api':
            required_api_fields = ['provider', 'model_name']
            for field in required_api_fields:
                if field not in model_config:
                    raise ValueError(f"API model {model_name} missing required '{field}' field")
    
    @staticmethod
    def _set_model_specific_defaults(normalized: Dict[str, Any]):
        """Set model-specific default values"""
        normalized.setdefault('max_tokens', Config.MAX_NEW_TOKENS)
        normalized.setdefault('finetuned', False)
        normalized.setdefault('description', f"Model configuration")
    
    @staticmethod
    def _normalize_reasoning_config(normalized: Dict[str, Any]):
        """Normalize reasoning configuration"""
        if normalized.get('is_reasoning_model', False):
            reasoning_defaults = Config.get_default_reasoning_config()
            reasoning_config = normalized.get('reasoning_config', {})
            normalized['reasoning_config'] = {**reasoning_defaults, **reasoning_config}
    
    @staticmethod
    def _normalize_generation_config(normalized: Dict[str, Any]):
        """Normalize generation configuration"""
        gen_defaults = Config.get_default_generation_config()
        gen_config = normalized.get('generation_config', {})
        normalized['generation_config'] = {**gen_defaults, **gen_config}


class ModelLoader:
    """Handles different model loading strategies"""
    
    @staticmethod
    def load_with_unsloth(model_path: str, config: Dict[str, Any]):
        """Load model using Unsloth FastLanguageModel"""
        from unsloth import FastLanguageModel
        
        # Prepare Unsloth-specific arguments
        load_args = {
            'model_name': model_path,
            'max_seq_length': Config.MAX_SEQ_LENGTH,
            'load_in_4bit': config.get('load_in_4bit', False),
            'trust_remote_code': config.get('trust_remote_code', False),
            'cache_dir': Config.CACHED_MODELS_DIR
        }
        
        # Handle dtype parameter
        ModelLoader._handle_dtype_parameter(load_args, config)
        
        if config.get('local_files_only', False):
            load_args['local_files_only'] = True
        
        logger.debug(f"Unsloth loading args: {load_args}")
        
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(**load_args)
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        except Exception as e:
            if 'torch_dtype' in str(e) or 'dtype' in str(e):
                return ModelLoader._retry_unsloth_without_dtype(load_args)
            else:
                raise
    
    @staticmethod
    def _handle_dtype_parameter(load_args: dict, config: Dict[str, Any]):
        """Handle dtype parameter for Unsloth (prefers 'dtype' over 'torch_dtype')"""
        torch_dtype = config.get('torch_dtype', 'auto')
        if torch_dtype != 'auto':
            load_args['dtype'] = torch_dtype
        
        # Remove any torch_dtype to avoid conflicts
        if 'torch_dtype' in load_args:
            del load_args['torch_dtype']
    
    @staticmethod
    def _retry_unsloth_without_dtype(load_args: dict):
        """Retry Unsloth loading without dtype specification"""
        from unsloth import FastLanguageModel
        
        logger.warning("Dtype issue detected. Retrying without dtype specification...")
        clean_args = {k: v for k, v in load_args.items() if k not in ['dtype', 'torch_dtype']}
        logger.debug(f"Retrying with clean args: {clean_args}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(**clean_args)
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    
    @staticmethod
    def load_with_transformers(model_path: str, config: Dict[str, Any]):
        """Load model using standard transformers"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        tokenizer_args = ModelLoader._prepare_tokenizer_args(config)
        logger.debug(f"Tokenizer loading args: {tokenizer_args}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
        
        # Load model
        model_args = ModelLoader._prepare_model_args(config)
        logger.debug(f"Model loading args: {model_args}")
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)
        
        return model, tokenizer
    
    @staticmethod
    def _prepare_tokenizer_args(config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare tokenizer loading arguments"""
        args = {
            'trust_remote_code': config.get('trust_remote_code', False),
            'cache_dir': Config.CACHED_MODELS_DIR
        }
        
        if config.get('local_files_only', False):
            args['local_files_only'] = True
        
        return args
    
    @staticmethod
    def _prepare_model_args(config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare model loading arguments"""
        args = {
            'trust_remote_code': config.get('trust_remote_code', False),
            'torch_dtype': config.get('torch_dtype', 'auto'),
            'device_map': config.get('device_map', 'auto'),
            'cache_dir': Config.CACHED_MODELS_DIR
        }
        
        if config.get('local_files_only', False):
            args['local_files_only'] = True
        
        # Handle quantization
        if config.get('load_in_4bit', False):
            ModelLoader._add_quantization_config(args)
        
        return args
    
    @staticmethod
    def _add_quantization_config(model_args: Dict[str, Any]):
        """Add quantization configuration to model args"""
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_args['quantization_config'] = quantization_config
            logger.info("Using 4-bit quantization with BitsAndBytes")
        except ImportError:
            logger.warning("BitsAndBytes not available, loading without quantization")


class APIQueryManager:
    """Handles API model queries with unified interface"""
    
    def __init__(self, openai_client=None, genai_client=None, anthropic_client=None):
        self.openai_client = openai_client
        self.genai_client = genai_client
        self.anthropic_client = anthropic_client
    
    def query_api_model(self, prompt: str, provider: str, model: str, 
                       max_tokens: int = None, temperature: float = None) -> str:
        """
        Generic API query method that routes to appropriate provider.
        
        Args:
            prompt: Input text prompt
            provider: API provider ('openai', 'google', 'anthropic')
            model: Model name for the provider
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            str: Generated text
        """
        max_tokens = max_tokens or Config.MAX_NEW_TOKENS
        temperature = temperature or Config.DEFAULT_TEMPERATURE
        
        logger.debug(f"Querying {provider} model {model} with prompt length: {len(prompt)}")
        
        if provider == 'openai':
            return self._query_openai_impl(prompt, model, max_tokens, temperature)
        elif provider == 'google':
            return self._query_genai_impl(prompt, model, max_tokens, temperature)
        elif provider == 'anthropic':
            return self._query_anthropic_impl(prompt, model, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown API provider: {provider}")
    
    def _query_openai_impl(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """OpenAI implementation"""
        if self.openai_client is None:
            raise ValueError("OpenAI client not initialized. Check API key.")
        
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
    
    def _query_genai_impl(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """Google GenAI implementation"""
        if self.genai_client is None:
            raise ValueError("GenAI client not initialized. Check API key.")
        
        try:
            model_instance = self.genai_client.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config=self.genai_client.types.GenerationConfig(
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
    
    def _query_anthropic_impl(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """Anthropic implementation"""
        if self.anthropic_client is None:
            raise ValueError("Anthropic client not initialized. Check API key.")
        
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text.strip()
            logger.debug(f"Anthropic response length: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Error querying Anthropic model {model}: {e}")
            raise


# =============================================================================
# MAIN MODEL MANAGER CLASS (REFACTORED)
# =============================================================================

class ModelManager:
    """
    Centralized model management system with comprehensive support for:
    1. Unsloth models (4-bit quantized and full precision)
    2. Standard transformer models (4-bit quantized and full precision)
    3. Reasoning models with special inference patterns
    4. API models (OpenAI, Google, Anthropic)
    5. Embedding models for similarity calculations
    
    The system automatically detects model type and applies appropriate loading and inference methods.
    """
    
    def __init__(self):
        """Initialize model manager with minimal state - load components only when needed"""
        # Local model state
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.current_model_config = None
        self.embedding_model = None
        
        # API clients
        self.openai_client = None
        self.genai_client = None
        self.anthropic_client = None
        self.api_manager = None
        
        # Lazy initialization flags
        self._gpu_status_checked = False
        self._unsloth_availability_checked = False
        self._gpu_status = None
        self._unsloth_available = None
        
        # Load model configurations from JSON
        self.models_config = Config.load_models_config()
        
        logger.info("ModelManager initialized (components will load on-demand)")
    
    # =============================================================================
    # SYSTEM STATUS AND CAPABILITIES
    # =============================================================================
    
    def _get_gpu_status(self):
        """Lazy initialization of GPU status checking"""
        if not self._gpu_status_checked:
            self._gpu_status = check_gpu_availability()
            self._gpu_status_checked = True
            self._log_system_capabilities()
        return self._gpu_status
    
    def _log_system_capabilities(self):
        """Log what the system can and cannot do based on available hardware"""
        gpu_status = self._gpu_status
        if gpu_status['cuda_available']:
            logger.info(f"GPU detected: {gpu_status['gpu_count']} device(s) with {gpu_status['total_memory']:.1f} GB total memory")
            if gpu_status['can_run_local_models']:
                logger.info("System can run local models")
            else:
                logger.warning("GPU memory may be insufficient for larger local models")
        else:
            if gpu_status['torch_available']:
                logger.info("CPU-only mode detected - local models will be slow")
            else:
                logger.info("No PyTorch detected - local models unavailable")
        
        # Count available models by type
        local_models = sum(1 for m in self.models_config.values() if m.get('type') == 'local')
        api_models = sum(1 for m in self.models_config.values() if m.get('type') == 'api')
        
        logger.info(f"Models configured: {local_models} local, {api_models} API-based")
    
    def _check_unsloth_availability(self):
        """Lazy check for Unsloth library availability"""
        if not self._unsloth_availability_checked:
            try:
                from unsloth import FastLanguageModel, is_bfloat16_supported
                from trl import SFTTrainer
                from transformers import TrainingArguments
                logger.info("Unsloth available for optimized model training and inference")
                self._unsloth_available = True
            except ImportError:
                logger.warning("Unsloth not available. Will use standard transformers for local models.")
                self._unsloth_available = False
            
            self._unsloth_availability_checked = True
        
        return self._unsloth_available
    
    # =============================================================================
    # MODEL TYPE CHECKING HELPERS
    # =============================================================================
    
    def _is_local_model(self, model_config: dict) -> bool:
        """Check if model configuration represents a local model"""
        return model_config.get('type') == 'local'
    
    def _is_reasoning_model(self, model_config: dict) -> bool:
        """Check if model configuration represents a reasoning model"""
        return model_config.get('is_reasoning_model', False)
    
    def _is_api_model(self, model_config: dict) -> bool:
        """Check if model configuration represents an API model"""
        return model_config.get('type') == 'api'
    
    # =============================================================================
    # MODEL LOADING - CORE INFRASTRUCTURE
    # =============================================================================
    
    def cleanup_current_model(self):
        """Clean up currently loaded model to free GPU/CPU memory"""
        logger.info("Cleaning up current model...")
        
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
        
        if self.current_tokenizer is not None:
            del self.current_tokenizer
            self.current_tokenizer = None
        
        self.current_model_name = None
        self.current_model_config = None
        clear_gpu_memory()
        logger.info("Model cleanup completed")
    
    def _load_local_model(self, model_name: str, model_path: str, model_config: dict):
        """
        Unified function to load local models with comprehensive configuration support.
        
        Args:
            model_name: Name from models.json configuration
            model_path: Hugging Face model path or local path
            model_config: Raw model configuration dictionary
        """
        # Validate and normalize configuration
        config = ConfigurationManager.validate_and_normalize_config(model_name, model_config)
        
        # Check GPU requirements
        self._validate_gpu_requirements()
        
        # Log memory status before loading
        logger.info("Memory status before model loading:")
        memory_before = MemoryMonitor.get_memory_before_operation()
        
        # Clean up any existing model first
        self.cleanup_current_model()
        
        # Log configuration being used
        self._log_loading_configuration(model_path, config)
        
        try:
            # Choose loading method based on configuration
            model, tokenizer = self._select_and_load_model(model_path, config)
            
            # Store references
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_name = model_name
            self.current_model_config = config
            
            logger.info(f"Successfully loaded model: {model_name}")
            
            # Show memory status after loading
            logger.info("Memory status after model loading:")
            MemoryMonitor.log_memory_usage("Model loading", memory_before)
            show_gpu_stats()
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.cleanup_current_model()
            raise
    
    def _validate_gpu_requirements(self):
        """Validate GPU requirements for local model loading"""
        gpu_status = self._get_gpu_status()
        if not gpu_status['cuda_available'] and not gpu_status['torch_available']:
            raise RuntimeError("Cannot load local models: PyTorch not available")
        
        if not gpu_status['cuda_available']:
            logger.warning("Loading local model on CPU - this will be very slow and may require >16GB RAM")
            response = input("Continue loading local model on CPU? (y/N): ")
            if response.lower() != 'y':
                raise RuntimeError("User cancelled CPU-only model loading")
    
    def _log_loading_configuration(self, model_path: str, config: Dict[str, Any]):
        """Log the configuration being used for model loading"""
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Configuration summary:")
        logger.info(f"   Type: {'Reasoning' if self._is_reasoning_model(config) else 'Standard'}")
        logger.info(f"   Loader: {'Unsloth' if config.get('use_unsloth') else 'Transformers'}")
        logger.info(f"   Quantization: {'4-bit' if config.get('load_in_4bit') else 'Full precision'}")
        logger.info(f"   Trust remote code: {config.get('trust_remote_code')}")
        logger.info(f"   Use chat template: {config.get('use_chat_template')}")
    
    def _select_and_load_model(self, model_path: str, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Select appropriate loading method and load the model"""
        use_unsloth = config.get('use_unsloth', False)
        
        if use_unsloth and self._check_unsloth_availability():
            logger.info("Loading with Unsloth (optimized)")
            return ModelLoader.load_with_unsloth(model_path, config)
        else:
            if use_unsloth:
                logger.warning("Unsloth requested but not available, falling back to standard transformers")
            logger.info("Loading with standard transformers")
            return ModelLoader.load_with_transformers(model_path, config)
    
    # =============================================================================
    # PUBLIC MODEL LOADING INTERFACE (PRESERVED FOR COMPATIBILITY)
    # =============================================================================
    
    def load_open_source_model(self, model_name: str, model_path: str):
        """Load an open source model using appropriate method"""
        if model_name not in self.models_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.models_config[model_name]
        
        if not self._is_local_model(model_config):
            raise ValueError(f"Model {model_name} is not a local model")
        
        self._load_local_model(model_name, model_path, model_config)
    
    def load_finetuned_model(self, model_path: str):
        """Load a finetuned model from local filesystem"""
        logger.info(f"Loading finetuned model from: {model_path}")
        
        # Find corresponding model configuration
        model_config, model_name = self._find_finetuned_config(model_path)
        
        # Use unified loading function
        self._load_local_model(model_name, model_path, model_config)
    
    def _find_finetuned_config(self, model_path: str) -> Tuple[Dict[str, Any], str]:
        """Find configuration for finetuned model"""
        # Try to find corresponding model configuration with finetuned=true
        for config_name, config in self.models_config.items():
            if (self._is_local_model(config) and 
                config.get('finetuned', False) and
                (config_name in model_path or 
                 os.path.basename(model_path).replace('_finetuned', '') in config.get('model_path', ''))):
                return config.copy(), config_name
        
        # Look for base model configuration to inherit from
        base_model_name = os.path.basename(model_path).replace('_finetuned', '')
        for config_name, config in self.models_config.items():
            if (self._is_local_model(config) and 
                not config.get('finetuned', False) and
                base_model_name in config.get('model_path', '')):
                inherited_config = config.copy()
                inherited_config['local_files_only'] = True
                model_name = f"finetuned_{config_name}"
                logger.info(f"Using base model configuration from: {config_name}")
                return inherited_config, model_name
        
        # Ultimate fallback: default configuration
        default_config = {
            'type': 'local',
            'model_path': model_path,
            'local_files_only': True,
            'finetuned': True
        }
        model_name = f"finetuned_{os.path.basename(model_path)}"
        logger.info("Using default configuration for finetuned model")
        return default_config, model_name
    
    # =============================================================================
    # PROMPT PREPARATION AND GENERATION UTILITIES
    # =============================================================================
    
    def _prepare_chat_prompt(self, prompt: str, use_chat_template: bool = None) -> Union[str, torch.Tensor]:
        """Prepare prompt using chat template if configured"""
        if use_chat_template is None:
            use_chat_template = self.current_model_config.get('use_chat_template', True)
        
        if not use_chat_template or not hasattr(self.current_tokenizer, 'apply_chat_template'):
            return prompt
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Check if this is a reasoning model that needs tokenized input
            if self._is_reasoning_model(self.current_model_config):
                return self._prepare_reasoning_model_input(messages)
            else:
                # For standard models, return formatted text
                formatted_prompt = self.current_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                logger.debug("Applied chat template to prompt")
                return formatted_prompt
            
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}, using raw prompt")
            return prompt
    
    def _prepare_reasoning_model_input(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Prepare tokenized input for reasoning models"""
        reasoning_config = self.current_model_config.get('reasoning_config', {})
        thinking_budget = reasoning_config.get('thinking_budget', 512)
        
        logger.debug(f"Applying chat template with tokenization for reasoning model (thinking_budget={thinking_budget})")
        
        tokenized_chat = self.current_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            thinking_budget=thinking_budget
        )
        
        return tokenized_chat
    
    def _prepare_generation_kwargs(self, max_tokens: int = None, temperature: float = None, **kwargs) -> Dict[str, Any]:
        """Prepare generation arguments from configuration and parameters"""
        # Start with model's generation config
        gen_config = self.current_model_config.get('generation_config', {}).copy()
        
        # Override with experiment parameters
        if max_tokens is not None:
            gen_config['max_new_tokens'] = max_tokens
        if temperature is not None:
            gen_config['temperature'] = temperature
        
        # Override with any additional kwargs
        gen_config.update(kwargs)
        
        # Remove parameters that belong to tokenizer.decode()
        decode_only_params = {'skip_special_tokens', 'clean_up_tokenization_spaces'}
        generation_params = {k: v for k, v in gen_config.items() if k not in decode_only_params}
        
        # Set tokenizer-specific tokens if available
        self._set_tokenizer_tokens(generation_params)
        
        # Remove None values
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        
        return generation_params
    
    def _set_tokenizer_tokens(self, generation_params: Dict[str, Any]):
        """Set tokenizer-specific tokens for generation"""
        if hasattr(self.current_tokenizer, 'pad_token_id') and self.current_tokenizer.pad_token_id is not None:
            generation_params.setdefault('pad_token_id', self.current_tokenizer.pad_token_id)
        elif hasattr(self.current_tokenizer, 'eos_token_id'):
            generation_params.setdefault('pad_token_id', self.current_tokenizer.eos_token_id)
        
        if hasattr(self.current_tokenizer, 'eos_token_id'):
            generation_params.setdefault('eos_token_id', self.current_tokenizer.eos_token_id)
    
    def _prepare_decoding_kwargs(self) -> Dict[str, Any]:
        """Prepare decoding arguments from configuration"""
        gen_config = self.current_model_config.get('generation_config', {})
        
        decoding_params = {
            'skip_special_tokens': gen_config.get('skip_special_tokens', True),
            'clean_up_tokenization_spaces': gen_config.get('clean_up_tokenization_spaces', True)
        }
        
        return decoding_params
    
    # =============================================================================
    # MODEL INFERENCE - LOCAL MODELS (PRESERVED INTERFACE)
    # =============================================================================
    
    def query_open_source(self, prompt: str, max_tokens: int = None, temperature: float = None, **kwargs) -> str:
        """Query currently loaded open source model for text generation"""
        if self.current_model is None or self.current_tokenizer is None:
            raise ValueError("No open source model loaded")
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        logger.debug(f"Querying open-source model with prompt length: {len(prompt)}")
        
        try:
            # Check if this is a reasoning model
            if self._is_reasoning_model(self.current_model_config):
                return self._query_reasoning_model(prompt, max_tokens, temperature, **kwargs)
            else:
                return self._query_standard_model(prompt, max_tokens, temperature, **kwargs)
                
        except Exception as e:
            logger.error(f"Error querying open-source model: {e}")
            logger.error(f"Model: {self.current_model_name}")
            logger.error(f"Prompt length: {len(prompt) if prompt else 'None'}")
            raise
    
    def _query_reasoning_model(self, prompt: str, max_tokens: int = None, temperature: float = None, **kwargs) -> str:
        """Handle reasoning model inference"""
        formatted_prompt = self._prepare_chat_prompt(prompt)
        
        if not isinstance(formatted_prompt, torch.Tensor):
            raise ValueError("Reasoning model should return tokenized input")
        
        inputs = formatted_prompt.to(self.current_model.device)
        gen_kwargs = self._prepare_generation_kwargs(max_tokens, temperature, **kwargs)
        decode_kwargs = self._prepare_decoding_kwargs()
        
        with torch.no_grad():
            outputs = self.current_model.generate(inputs, **gen_kwargs)
        
        if outputs is None or len(outputs) == 0:
            raise ValueError("Model generation returned empty output")
        
        # For reasoning models, check if we should decode the full output
        reasoning_config = self.current_model_config.get('reasoning_config', {})
        decode_full = reasoning_config.get('decode_full_output', False)
        
        if decode_full:
            decoded = self.current_tokenizer.decode(outputs[0], **decode_kwargs)
        else:
            input_length = inputs.shape[1] if inputs.dim() > 1 else len(inputs)
            output_ids = outputs[0][input_length:].tolist()
            decoded = self.current_tokenizer.decode(output_ids, **decode_kwargs)
        
        return decoded.strip()
    
    def _query_standard_model(self, prompt: str, max_tokens: int = None, temperature: float = None, **kwargs) -> str:
        """Handle standard model inference"""
        formatted_prompt = self._prepare_chat_prompt(prompt)
        gen_kwargs = self._prepare_generation_kwargs(max_tokens, temperature, **kwargs)
        decode_kwargs = self._prepare_decoding_kwargs()
        
        # Tokenize input if it's still text
        if isinstance(formatted_prompt, str):
            inputs = self.current_tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = inputs.to(self.current_model.device)
        elif isinstance(formatted_prompt, torch.Tensor):
            inputs = formatted_prompt.to(self.current_model.device)
        else:
            raise ValueError(f"Unexpected formatted_prompt type: {type(formatted_prompt)}")
        
        # Generate with no gradients for efficiency
        with torch.no_grad():
            output = self.current_model.generate(**inputs, **gen_kwargs)
            
        if output is None or len(output) == 0:
            raise ValueError("Model generation returned empty output")
        
        # Extract only the new tokens (remove input)
        if hasattr(inputs, 'input_ids'):
            input_length = inputs.input_ids.shape[1]
        elif hasattr(inputs, 'shape'):
            input_length = inputs.shape[1]
        else:
            logger.warning("Cannot determine input length, using full output")
            input_length = 0
        
        output_ids = output[0][input_length:].tolist()
        decoded = self.current_tokenizer.decode(output_ids, **decode_kwargs)
        
        return decoded.strip()
    
    # =============================================================================
    # EMBEDDING MODELS
    # =============================================================================
    
    def load_embedding_model(self):
        """Load embedding model for semantic similarity calculations"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            
            memory_before = MemoryMonitor.get_memory_before_operation()
            
            try:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=Config.EMBEDDING_MODEL,
                    cache_folder=Config.CACHED_MODELS_DIR
                )
                
                logger.info("Embedding model loaded successfully")
                MemoryMonitor.log_memory_usage("Embedding model loading", memory_before)
            
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                logger.info("This may be due to missing dependencies or network issues")
                raise
        else:
            logger.debug("Embedding model already loaded")
        
        return self.embedding_model
    
    # =============================================================================
    # API CLIENT SETUP AND INFERENCE (PRESERVED INTERFACE)
    # =============================================================================
    
    def setup_api_clients(self):
        """Initialize API clients for external model providers"""
        logger.info("Setting up API clients...")
        
        # OpenAI client setup
        if Config.OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Google GenAI client setup
        if Config.GENAI_API_KEY:
            try:
                genai.configure(api_key=Config.GENAI_API_KEY)
                self.genai_client = genai
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
        
        # Initialize unified API manager
        self.api_manager = APIQueryManager(
            self.openai_client, 
            self.genai_client, 
            self.anthropic_client
        )
    
    # Preserved individual API methods for backward compatibility
    def query_openai(self, prompt: str, model: str = 'gpt-4o-mini', max_tokens: int = None, temperature: float = None) -> str:
        """Query OpenAI API for text generation"""
        if self.api_manager:
            return self.api_manager.query_api_model(prompt, 'openai', model, max_tokens, temperature)
        else:
            # Fallback for direct usage
            if self.openai_client is None:
                raise ValueError("OpenAI client not initialized. Check API key.")
            return self.api_manager._query_openai_impl(prompt, model, max_tokens or Config.MAX_NEW_TOKENS, temperature or Config.DEFAULT_TEMPERATURE)
    
    def query_genai(self, prompt: str, model: str = 'gemini-1.5-flash', max_tokens: int = None, temperature: float = None) -> str:
        """Query Google GenAI API for text generation"""
        if self.api_manager:
            return self.api_manager.query_api_model(prompt, 'google', model, max_tokens, temperature)
        else:
            raise ValueError("API manager not initialized. Call setup_api_clients() first.")
    
    def query_anthropic(self, prompt: str, model: str = 'claude-3-5-sonnet-20241022', max_tokens: int = None, temperature: float = None) -> str:
        """Query Anthropic Claude API for text generation"""
        if self.api_manager:
            return self.api_manager.query_api_model(prompt, 'anthropic', model, max_tokens, temperature)
        else:
            raise ValueError("API manager not initialized. Call setup_api_clients() first.")
    
    # =============================================================================
    # UTILITY METHODS (PRESERVED FOR COMPATIBILITY)
    # =============================================================================
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available for use"""
        if model_name not in self.models_config:
            return False
        
        model_config = self.models_config[model_name]
        
        if self._is_local_model(model_config):
            use_unsloth = model_config.get('use_unsloth', False)
            
            if use_unsloth and not self._check_unsloth_availability():
                logger.debug(f"Model {model_name} prefers Unsloth but it's not available, checking transformers")
            
            # For finetuned models, check if the finetuned version exists
            if model_config.get('finetuned', False):
                model_path = os.path.join(Config.FINETUNED_MODELS_DIR, 
                                          model_config['model_path'].split('/')[-1] + '_finetuned')
                return os.path.exists(model_path)
            
            return True  # Can try to load with available libraries
            
        elif self._is_api_model(model_config):
            # Check if appropriate API client is available
            provider = model_config.get('provider')
            
            if provider == 'openai':
                return self.openai_client is not None
            elif provider == 'google':
                return self.genai_client is not None
            elif provider == 'anthropic':
                return self.anthropic_client is not None
            
        return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a model"""
        if model_name not in self.models_config:
            return {'available': False, 'error': 'Model not found in configuration'}
        
        config = self.models_config[model_name]
        info = {
            'name': model_name,
            'available': self.is_model_available(model_name),
            'type': config.get('type'),
            'description': config.get('description', 'No description available'),
            'configuration': config
        }
        
        if self._is_local_model(config):
            info.update({
                'loader': 'Unsloth' if config.get('use_unsloth') else 'Transformers',
                'quantization': '4-bit' if config.get('load_in_4bit') else 'Full precision',
                'reasoning_model': self._is_reasoning_model(config),
                'finetuned': config.get('finetuned', False)
            })
        
        elif self._is_api_model(config):
            info.update({
                'provider': config.get('provider'),
                'api_model_name': config.get('model_name')
            })
        
        return info
    
    def requires_local_models(self, model_names: list) -> bool:
        """Check if any of the specified models are local models"""
        for model_name in model_names:
            if model_name in self.models_config:
                if self._is_local_model(self.models_config[model_name]):
                    return True
        return False