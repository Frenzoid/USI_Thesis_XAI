import torch
import os
from typing import Optional, Dict, Any, List
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import google.generativeai as genai

from config import Config
from utils import setup_logging, clear_gpu_memory, get_memory_status

logger = setup_logging("models")

class ModelManager:
    """
    Centralized model management system supporting:
    - Unsloth and standard transformers for local models
    - Reasoning models with thinking budget
    - API models (OpenAI, Google, Anthropic)
    - Embedding models for similarity calculations
    - Hugging Face authentication for restricted repositories
    """
    
    def __init__(self):
        """Initialize model manager"""
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_config = None
        self.embedding_model = None
        
        # API clients
        self.openai_client = None
        self.genai_client = None
        self.anthropic_client = None
        
        # Hugging Face authentication setup
        self._setup_hf_authentication()
        
        self.models_config = Config.load_models_config()
        logger.info("ModelManager initialized")
    
    def _setup_hf_authentication(self):
        """Setup Hugging Face authentication if token is available"""
        if Config.HF_ACCESS_TOKEN:
            try:
                from huggingface_hub import login
                login(token=Config.HF_ACCESS_TOKEN, add_to_git_credential=True)
                logger.info("Hugging Face authentication configured successfully")
            except ImportError:
                logger.warning("huggingface_hub not available - some models may be inaccessible")
            except Exception as e:
                logger.error(f"Failed to setup Hugging Face authentication: {e}")
        else:
            logger.info("No Hugging Face token found - restricted models will be unavailable")
    
    def _check_model_accessibility(self, model_path: str, model_name: str) -> bool:
        """
        Check if a model is accessible, handling authentication errors gracefully.
        
        Args:
            model_path: Path/name of the model to check
            model_name: Human-readable model name for logging
            
        Returns:
            bool: True if model is accessible, False otherwise
        """
        try:
            from huggingface_hub import model_info, HfApi
            
            # Try to get model info to verify accessibility
            api = HfApi(token=Config.HF_ACCESS_TOKEN if Config.HF_ACCESS_TOKEN else None)
            model_info_result = api.model_info(model_path)
            
            # Check if model requires authentication and we don't have a token
            if hasattr(model_info_result, 'gated') and model_info_result.gated and not Config.HF_ACCESS_TOKEN:
                logger.warning(f"Model '{model_name}' ({model_path}) is gated and requires authentication")
                logger.info(f"Add HF_ACCESS_TOKEN to .env file to access gated models")
                return False
            
            logger.debug(f"Model '{model_name}' is accessible")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle different types of access errors
            if any(keyword in error_msg for keyword in ['gated', 'restricted', 'private', 'authentication', 'unauthorized', '401', '403']):
                logger.warning(f"Model '{model_name}' ({model_path}) requires authentication or is restricted")
                if not Config.HF_ACCESS_TOKEN:
                    logger.info("Add HF_ACCESS_TOKEN to .env file to access restricted models")
                else:
                    logger.info("Your HF_ACCESS_TOKEN may not have access to this model")
                return False
                
            elif 'not found' in error_msg or '404' in error_msg:
                logger.error(f"Model '{model_name}' ({model_path}) not found")
                return False
                
            else:
                # For other errors, log warning but don't skip (might be network issues)
                logger.warning(f"Could not verify accessibility for model '{model_name}': {e}")
                return True  # Assume accessible and let the actual loading fail if needed
    
    def cleanup_current_model(self):
        """Clean up currently loaded model to free memory"""
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_config = None
            clear_gpu_memory()
            logger.info("Model cleanup completed")
    
    def _load_with_unsloth(self, model_path: str, config: Dict[str, Any]):
        """Load model using Unsloth FastLanguageModel"""
        try:
            from unsloth import FastLanguageModel
            
            load_args = {
                'model_name': model_path,
                'max_seq_length': Config.MAX_SEQ_LENGTH,
                'load_in_4bit': config.get('load_in_4bit', False),
                'trust_remote_code': config.get('trust_remote_code', False),
                'cache_dir': Config.CACHED_MODELS_DIR
            }
            
            # Add HF token if available
            if Config.HF_ACCESS_TOKEN:
                load_args['token'] = Config.HF_ACCESS_TOKEN
            
            # Handle dtype parameter (Unsloth prefers 'dtype' over 'torch_dtype')
            torch_dtype = config.get('torch_dtype', 'auto')
            if torch_dtype != 'auto':
                load_args['dtype'] = torch_dtype
            
            if config.get('local_files_only', False):
                load_args['local_files_only'] = True
            
            model, tokenizer = FastLanguageModel.from_pretrained(**load_args)
            FastLanguageModel.for_inference(model)
            return model, tokenizer
            
        except Exception as e:
            # Handle authentication errors specifically
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['gated', 'restricted', 'private', 'authentication', 'unauthorized', '401', '403']):
                logger.error(f"Authentication failed for model {model_path}: {e}")
                if not Config.HF_ACCESS_TOKEN:
                    logger.info("Consider adding HF_ACCESS_TOKEN to .env file for restricted models")
                raise ValueError(f"Model requires authentication: {model_path}")
            
            if 'dtype' in str(e):
                # Retry without dtype
                load_args.pop('dtype', None)
                model, tokenizer = FastLanguageModel.from_pretrained(**load_args)
                FastLanguageModel.for_inference(model)
                return model, tokenizer
            raise
    
    def _load_with_transformers(self, model_path: str, config: Dict[str, Any]):
        """Load model using standard transformers"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Common args
        common_args = {
            'trust_remote_code': config.get('trust_remote_code', False),
            'cache_dir': Config.CACHED_MODELS_DIR
        }
        
        # Add HF token if available
        if Config.HF_ACCESS_TOKEN:
            common_args['token'] = Config.HF_ACCESS_TOKEN
        
        if config.get('local_files_only', False):
            common_args['local_files_only'] = True
        
        # Load tokenizer with authentication error handling
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, **common_args)
            
            # Fix missing pad_token issue (common with Mistral and other models)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Set pad_token to eos_token for {model_path}")
                else:
                    # Fallback: add a pad token
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.info(f"Added [PAD] token as pad_token for {model_path}")
                    logger.warning(f"Tokenizer for {model_path} had no pad_token or eos_token; added [PAD]")
                    
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['gated', 'restricted', 'private', 'authentication', 'unauthorized', '401', '403']):
                logger.error(f"Authentication failed for tokenizer {model_path}: {e}")
                if not Config.HF_ACCESS_TOKEN:
                    logger.info("Consider adding HF_ACCESS_TOKEN to .env file for restricted models")
                raise ValueError(f"Tokenizer requires authentication: {model_path}")
            raise
        
        # Load model with additional args
        model_args = {
            **common_args,
            'torch_dtype': config.get('torch_dtype', 'auto'),
            'device_map': config.get('device_map', 'auto')
        }
        
        # Add quantization if needed
        if config.get('load_in_4bit', False):
            try:
                from transformers import BitsAndBytesConfig
                model_args['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            except ImportError:
                logger.warning("BitsAndBytes not available, loading without quantization")
        
        # Load model with authentication error handling
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['gated', 'restricted', 'private', 'authentication', 'unauthorized', '401', '403']):
                logger.error(f"Authentication failed for model {model_path}: {e}")
                if not Config.HF_ACCESS_TOKEN:
                    logger.info("Consider adding HF_ACCESS_TOKEN to .env file for restricted models")
                raise ValueError(f"Model requires authentication: {model_path}")
            raise
        
        return model, tokenizer
    
    def load_open_source_model(self, model_name: str, model_path: str):
        """Load a local model using the appropriate method with authentication handling"""
        if model_name not in self.models_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.models_config[model_name]
        
        # Check model accessibility first
        if not self._check_model_accessibility(model_path, model_name):
            raise ValueError(f"Model '{model_name}' is not accessible - check authentication or permissions")
        
        # Apply defaults
        config = {
            **Config.get_default_model_loading_config(),
            **config
        }
        
        # Add reasoning config if needed
        if config.get('is_reasoning_model', False):
            reasoning_defaults = Config.get_default_reasoning_config()
            reasoning_config = config.get('reasoning_config', {})
            config['reasoning_config'] = {**reasoning_defaults, **reasoning_config}
        
        # Add generation config
        gen_defaults = Config.get_default_generation_config()
        gen_config = config.get('generation_config', {})
        config['generation_config'] = {**gen_defaults, **gen_config}
        
        logger.info(f"Loading model: {model_path}")
        
        # Clean up existing model
        self.cleanup_current_model()
        
        # Choose loading method
        use_unsloth = config.get('use_unsloth', False)
        
        try:
            if use_unsloth:
                try:
                    model, tokenizer = self._load_with_unsloth(model_path, config)
                    logger.info("Loaded with Unsloth")
                except ImportError:
                    logger.warning("Unsloth not available, using transformers")
                    model, tokenizer = self._load_with_transformers(model_path, config)
                except ValueError as e:
                    # Re-raise authentication errors
                    if "authentication" in str(e).lower():
                        raise
                    # For other ValueError, try transformers
                    logger.warning(f"Unsloth failed ({e}), trying transformers")
                    model, tokenizer = self._load_with_transformers(model_path, config)
            else:
                model, tokenizer = self._load_with_transformers(model_path, config)
                logger.info("Loaded with transformers")
            
            # Store references
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_config = config
            
            logger.info(f"Successfully loaded model: {model_name}")
            
        except ValueError as e:
            # Authentication errors should be propagated
            if "authentication" in str(e).lower() or "not accessible" in str(e).lower():
                logger.error(f"Failed to load model {model_name}: {e}")
                self.cleanup_current_model()
                raise
            # Other ValueError should also be propagated
            logger.error(f"Failed to load model {model_name}: {e}")
            self.cleanup_current_model()
            raise
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.cleanup_current_model()
            raise
    
    def load_finetuned_model(self, model_path: str):
        """Load a finetuned model from local filesystem"""
        logger.info(f"Loading finetuned model from: {model_path}")
        
        # Create default config for finetuned model
        config = {
            **Config.get_default_model_loading_config(),
            'local_files_only': True,
            'finetuned': True
        }
        
        model_name = f"finetuned_{os.path.basename(model_path)}"
        self.models_config[model_name] = config
        
        self.load_open_source_model(model_name, model_path)
    
    def _prepare_chat_prompt(self, prompt: str) -> str:
        """Prepare prompt using chat template if configured"""
        use_chat_template = self.current_model_config.get('use_chat_template', True)
        
        if not use_chat_template or not hasattr(self.current_tokenizer, 'apply_chat_template'):
            return prompt
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Check if this is a reasoning model
            if self.current_model_config.get('is_reasoning_model', False):
                reasoning_config = self.current_model_config.get('reasoning_config', {})
                thinking_budget = reasoning_config.get('thinking_budget', 512)
                
                # Reasoning models need tokenized input
                return self.current_tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    thinking_budget=thinking_budget
                )
            else:
                # Standard models use text format
                return self.current_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}, using raw prompt")
            return prompt
    
    def query_open_source(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Query the currently loaded local model"""
        if self.current_model is None or self.current_tokenizer is None:
            raise ValueError("No model loaded")
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Prepare generation parameters
        gen_config = self.current_model_config.get('generation_config', {})
        gen_kwargs = gen_config.copy()
        
        if max_tokens is not None:
            gen_kwargs['max_new_tokens'] = max_tokens
        if temperature is not None:
            gen_kwargs['temperature'] = temperature
        
        # Set tokenizer tokens
        if hasattr(self.current_tokenizer, 'pad_token_id') and self.current_tokenizer.pad_token_id is not None:
            gen_kwargs['pad_token_id'] = self.current_tokenizer.pad_token_id
        elif hasattr(self.current_tokenizer, 'eos_token_id'):
            gen_kwargs['pad_token_id'] = self.current_tokenizer.eos_token_id
        
        if hasattr(self.current_tokenizer, 'eos_token_id'):
            gen_kwargs['eos_token_id'] = self.current_tokenizer.eos_token_id
        
        # Remove decode-only parameters
        decode_kwargs = {
            'skip_special_tokens': gen_kwargs.pop('skip_special_tokens', True),
            'clean_up_tokenization_spaces': gen_kwargs.pop('clean_up_tokenization_spaces', True)
        }
        
        try:
            formatted_prompt = self._prepare_chat_prompt(prompt)
            
            # Handle reasoning models vs standard models
            if isinstance(formatted_prompt, torch.Tensor):
                # Reasoning model with tokenized input
                inputs = formatted_prompt.to(self.current_model.device)
                
                with torch.no_grad():
                    outputs = self.current_model.generate(inputs, **gen_kwargs)
                
                # Check if we should decode full output for reasoning models
                reasoning_config = self.current_model_config.get('reasoning_config', {})
                decode_full = reasoning_config.get('decode_full_output', False)
                
                if decode_full:
                    decoded = self.current_tokenizer.decode(outputs[0], **decode_kwargs)
                else:
                    input_length = inputs.shape[1]
                    output_ids = outputs[0][input_length:]
                    decoded = self.current_tokenizer.decode(output_ids, **decode_kwargs)
            else:
                # Standard model with text input
                inputs = self.current_tokenizer(
                    formatted_prompt, 
                    return_tensors="pt",
                    padding=Config.DEFAULT_PADDING,
                    truncation=Config.DEFAULT_TRUNCATION,
                    max_length=Config.DEFAULT_MAX_LENGTH
                ).to(self.current_model.device)
                
                with torch.no_grad():
                    outputs = self.current_model.generate(**inputs, **gen_kwargs)
                
                # Extract only new tokens
                input_length = inputs.input_ids.shape[1]
                output_ids = outputs[0][input_length:]
                decoded = self.current_tokenizer.decode(output_ids, **decode_kwargs)
            
            return decoded.strip()
            
        except Exception as e:
            logger.error(f"Error querying model: {e}")
            raise
    
    def setup_api_clients(self):
        """Initialize API clients for external providers"""
        if Config.OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        if Config.GENAI_API_KEY:
            try:
                genai.configure(api_key=Config.GENAI_API_KEY)
                self.genai_client = genai
                logger.info("Google GenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize GenAI client: {e}")
        
        if Config.ANTHROPIC_API_KEY:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
    
    def query_api_model(self, prompt: str, provider: str, model: str, 
                       max_tokens: int = None, temperature: float = None) -> str:
        """Query API model with unified interface"""
        max_tokens = max_tokens or Config.MAX_NEW_TOKENS
        temperature = temperature or Config.DEFAULT_TEMPERATURE
        
        if provider == 'openai':
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized")
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
            
        elif provider == 'google':
            if not self.genai_client:
                raise ValueError("GenAI client not initialized")
            
            model_instance = self.genai_client.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config=self.genai_client.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text.strip()
            
        elif provider == 'anthropic':
            if not self.anthropic_client:
                raise ValueError("Anthropic client not initialized")
            
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
            
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def load_embedding_model(self):
        """Load embedding model for semantic similarity with authentication"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            
            try:
                # Check if embedding model needs authentication
                if not self._check_model_accessibility(Config.EMBEDDING_MODEL, "embedding_model"):
                    logger.warning(f"Embedding model {Config.EMBEDDING_MODEL} may not be accessible")
                    # Try to continue anyway, transformers might handle it
                
                model_kwargs = {'cache_folder': Config.CACHED_MODELS_DIR}
                
                # Add HF token if available
                if Config.HF_ACCESS_TOKEN:
                    model_kwargs['model_kwargs'] = {'token': Config.HF_ACCESS_TOKEN}
                
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=Config.EMBEDDING_MODEL,
                    **model_kwargs
                )
                logger.info("Embedding model loaded successfully")
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['gated', 'restricted', 'private', 'authentication', 'unauthorized', '401', '403']):
                    logger.error(f"Embedding model requires authentication: {e}")
                    if not Config.HF_ACCESS_TOKEN:
                        logger.info("Consider adding HF_ACCESS_TOKEN to .env file")
                    raise ValueError(f"Embedding model requires authentication: {Config.EMBEDDING_MODEL}")
                else:
                    logger.error(f"Failed to load embedding model: {e}")
                    raise
        
        return self.embedding_model
    
    def get_inaccessible_models(self) -> List[str]:
        """
        Get list of models from config that are not accessible due to authentication issues.
        
        Returns:
            List[str]: List of model names that are not accessible
        """
        inaccessible_models = []
        
        for model_name, model_config in self.models_config.items():
            # Only check local models
            if model_config.get('type') != 'local':
                continue
                
            model_path = model_config.get('model_path', '')
            if model_path and not self._check_model_accessibility(model_path, model_name):
                inaccessible_models.append(model_name)
        
        return inaccessible_models