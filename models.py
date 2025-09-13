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
        
        self.models_config = Config.load_models_config()
        logger.info("ModelManager initialized")
    
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
        
        if config.get('local_files_only', False):
            common_args['local_files_only'] = True
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, **common_args)
        
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
        
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)
        return model, tokenizer
    
    def load_open_source_model(self, model_name: str, model_path: str):
        """Load a local model using the appropriate method"""
        if model_name not in self.models_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.models_config[model_name]
        
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
            else:
                model, tokenizer = self._load_with_transformers(model_path, config)
                logger.info("Loaded with transformers")
            
            # Store references
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_config = config
            
            logger.info(f"Successfully loaded model: {model_name}")
            
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
                    padding=True,
                    truncation=True
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
        """Load embedding model for semantic similarity"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            
            try:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=Config.EMBEDDING_MODEL,
                    cache_folder=Config.CACHED_MODELS_DIR
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        
        return self.embedding_model