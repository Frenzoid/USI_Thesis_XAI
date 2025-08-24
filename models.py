import torch
import os
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from google import genai
from google.genai import types # Correctly import 'types'

from config import Config
from utils import setup_logging, clear_gpu_memory, show_gpu_stats

logger = setup_logging("models")

class ModelManager:
    """Centralized model management system with JSON configuration support"""
    
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.embedding_model = None
        
        # API clients
        self.openai_client = None
        self.genai_client = None
        self.anthropic_client = None
        
        # Load model configurations
        self.models_config = Config.load_models_config()
        
        # Check Unsloth availability for local models
        self.unsloth_available = self._check_unsloth_availability()
        
        logger.info("ModelManager initialized")
    
    def _check_unsloth_availability(self):
        """Check if Unsloth is available for open-source models"""
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
        """Clean up currently loaded model to free memory"""
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
        """Load embedding model for similarity calculations"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        else:
            logger.info("Embedding model already loaded")
        
        return self.embedding_model
    
    def load_open_source_model(self, model_name: str, model_path: str):
        """Load an open source model using Unsloth"""
        if not self.unsloth_available:
            raise ImportError("Unsloth not available. Cannot load open source models.")
        
        if model_name not in self.models_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.models_config[model_name]
        
        if model_config['type'] != 'local':
            raise ValueError(f"Model {model_name} is not a local model")
        
        # Clean up current model
        self.cleanup_current_model()
        
        logger.info(f"Loading open-source model: {model_path}")
        
        try:
            from unsloth import FastLanguageModel
            
            self.current_model, self.current_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=Config.MAX_SEQ_LENGTH,
                load_in_4bit=True,
                cache_dir=Config.MODELS_CACHE_DIR
            )
            
            # Set for inference
            FastLanguageModel.for_inference(self.current_model)
            
            self.current_model_name = model_name
            logger.info(f"Successfully loaded model: {model_name}")
            show_gpu_stats()
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def load_finetuned_model(self, model_path: str):
        """Load a finetuned model from local path"""
        if not self.unsloth_available:
            raise ImportError("Unsloth not available. Cannot load finetuned models.")
        
        # Clean up current model
        self.cleanup_current_model()
        
        logger.info(f"Loading finetuned model from: {model_path}")
        
        try:
            from unsloth import FastLanguageModel
            
            self.current_model, self.current_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                cache_dir=Config.MODELS_CACHE_DIR,
                local_files_only=True,
                max_seq_length=Config.MAX_SEQ_LENGTH,
                load_in_4bit=True
            )
            
            # Set for inference
            FastLanguageModel.for_inference(self.current_model)
            
            self.current_model_name = f"finetuned_{os.path.basename(model_path)}"
            logger.info(f"Successfully loaded finetuned model from: {model_path}")
            show_gpu_stats()
            
        except Exception as e:
            logger.error(f"Failed to load finetuned model from {model_path}: {e}")
            raise
    
    def setup_api_clients(self):
        """Setup API clients for external services"""
        logger.info("Setting up API clients...")
        
        # OpenAI
        if Config.OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Google GenAI
        if Config.GENAI_API_KEY:
            try:
                # Use genai.Client() for the newer SDK
                self.genai_client = genai.Client(api_key=Config.GENAI_API_KEY)
                logger.info("Google GenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize GenAI client: {e}")
        
        # Anthropic
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
        """Query currently loaded open source model"""
        if self.current_model is None or self.current_tokenizer is None:
            raise ValueError("No open source model loaded")
        
        max_tokens = max_tokens or Config.MAX_NEW_TOKENS
        temperature = temperature or Config.DEFAULT_TEMPERATURE
        
        logger.debug(f"Querying open-source model with prompt length: {len(prompt)}")
        
        try:
            inputs = self.current_tokenizer(prompt, return_tensors="pt").to(self.current_model.device)
            
            with torch.no_grad():
                output = self.current_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.current_tokenizer.eos_token_id
                )
            
            # Decode and remove prompt
            decoded = self.current_tokenizer.decode(output[0], skip_special_tokens=True)
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):].strip()
            
            logger.debug(f"Generated response length: {len(decoded)}")
            return decoded
            
        except Exception as e:
            logger.error(f"Error querying open-source model: {e}")
            raise
    
    def query_openai(self, prompt: str, model: str = 'gpt-4o-mini', max_tokens: int = None, temperature: float = None) -> str:
        """Query OpenAI API"""
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
        """Query Google GenAI API"""
        if self.genai_client is None:
            raise ValueError("GenAI client not initialized. Check API key.")
        
        temperature = temperature or Config.DEFAULT_TEMPERATURE
        max_tokens = max_tokens or Config.MAX_NEW_TOKENS
        
        logger.debug(f"Querying GenAI model {model} with prompt length: {len(prompt)}")
        
        try:

            # Use the newer, correct syntax for `generate_content`
            response = self.genai_client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
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
        """Query Anthropic Claude API"""
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
        """Check if a specific model is available"""
        if model_name not in self.models_config:
            return False
        
        model_config = self.models_config[model_name]
        
        if model_config['type'] == 'local':
            # Check if Unsloth is available and if finetuned model exists (if needed)
            if not self.unsloth_available:
                return False
            
            if model_config.get('finetuned', False):
                # Check if finetuned model exists
                model_path = os.path.join(Config.FINETUNED_MODELS_DIR, 
                                          model_config['model_path'].split('/')[-1] + '_finetuned')
                return os.path.exists(model_path)
            
            return True
            
        elif model_config['type'] == 'api':
            provider = model_config['provider']
            
            if provider == 'openai':
                return self.openai_client is not None
            elif provider == 'google':
                return self.genai_client is not None
            elif provider == 'anthropic':
                return self.anthropic_client is not None
            
        return False
    
    def get_available_models(self) -> dict:
        """Get dictionary of all available models"""
        available = {
            'local': [],
            'api': []
        }
        
        for model_name, model_config in self.models_config.items():
            if self.is_model_available(model_name):
                available[model_config['type']].append(model_name)
        
        return available
    
    def get_model_info(self, model_name: str = None) -> dict:
        """Get information about models"""
        if model_name:
            if model_name not in self.models_config:
                return {"error": f"Unknown model: {model_name}"}
            
            info = dict(self.models_config[model_name])
            info['available'] = self.is_model_available(model_name)
            info['currently_loaded'] = (self.current_model_name == model_name)
            
            return info
        else:
            # Return info about all models
            info = {
                'current_model': self.current_model_name,
                'model_loaded': self.current_model is not None,
                'embedding_model_loaded': self.embedding_model is not None,
                'api_clients': {
                    'openai': self.openai_client is not None,
                    'genai': self.genai_client is not None,
                    'anthropic': self.anthropic_client is not None
                },
                'unsloth_available': self.unsloth_available,
                'available_models': self.get_available_models(),
                'total_models_configured': len(self.models_config)
            }
            
            return info