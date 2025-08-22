import torch
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from google import genai

from config import Config
from utils import setup_logging, clear_gpu_memory, show_gpu_stats

logger = setup_logging("models")

class ModelManager:
    """Centralized model management system"""
    
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.embedding_model = None
        
        # API clients
        self.openai_client = None
        self.genai_client = None
        self.anthropic_client = None
        
        # Enhanced model configurations
        self.open_source_models = {
            'mistral-7b': 'unsloth/mistral-7b-v0.3-bnb-4bit',
            'mistral-7b-instruct': 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit',
            'llama3-8b': 'unsloth/llama-3-8b-bnb-4bit',
            'llama3-8b-instruct': 'unsloth/llama-3-8b-instruct-bnb-4bit',
            'llama3.2-3b': 'unsloth/llama-3.2-3b-instruct-bnb-4bit',
            'llama3.2-1b': 'unsloth/llama-3.2-1b-instruct-bnb-4bit',
            'qwen2-7b': 'unsloth/qwen2-7b-bnb-4bit',
            'qwen2-7b-instruct': 'unsloth/qwen2-7b-instruct-bnb-4bit',
            'phi3-mini': 'unsloth/phi-3-mini-4k-instruct-bnb-4bit',
            'phi3-medium': 'unsloth/phi-3-medium-4k-instruct-bnb-4bit',
            'gemma-7b': 'unsloth/gemma-7b-bnb-4bit',
            'gemma-7b-instruct': 'unsloth/gemma-7b-it-bnb-4bit',
            'codellama-7b': 'unsloth/codellama-7b-bnb-4bit',
            'tinyllama-1b': 'unsloth/tinyllama-bnb-4bit'
        }
        
        self.api_models = {
            'gpt-4o': 'gpt-4o',
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-4-turbo': 'gpt-4-turbo',
            'gpt-3.5-turbo': 'gpt-3.5-turbo',
            'gemini-1.5-pro': 'gemini-1.5-pro',
            'gemini-1.5-flash': 'gemini-1.5-flash',
            'gemini-1.0-pro': 'gemini-1.0-pro',
            'claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
            'claude-3-opus': 'claude-3-opus-20240229',
            'claude-3-haiku': 'claude-3-haiku-20240307'
        }
        
        logger.info("ModelManager initialized")
        self._check_unsloth_availability()
    
    def _check_unsloth_availability(self):
        """Check if Unsloth is available for open-source models"""
        try:
            from unsloth import FastLanguageModel, is_bfloat16_supported
            from trl import SFTTrainer
            from transformers import TrainingArguments
            self.unsloth_available = True
            logger.info("Unsloth available for open-source model training")
        except ImportError:
            self.unsloth_available = False
            logger.warning("Unsloth not available. Fine-tuning experiments will be disabled.")
    
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

    def load_open_source_model(self, model_key: str):
        """Load an open source model using Unsloth"""
        if not self.unsloth_available:
            raise ImportError("Unsloth not available. Cannot load open source models.")
        
        if model_key not in self.open_source_models:
            available_models = list(self.open_source_models.keys())
            raise ValueError(f"Unknown model: {model_key}. Available: {available_models}")
        
        # Clean up current model
        self.cleanup_current_model()
        
        model_name = self.open_source_models[model_key]
        logger.info(f"Loading open-source model: {model_name}")
        
        try:
            from unsloth import FastLanguageModel
            
            self.current_model, self.current_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=Config.MAX_SEQ_LENGTH,
                load_in_4bit=True,
                cache_dir=Config.MODELS_DIR
            )
            
            # Set for inference
            FastLanguageModel.for_inference(self.current_model)
            
            self.current_model_name = model_key
            logger.info(f"Successfully loaded model: {model_key}")
            show_gpu_stats()
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
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
                self.genai_client = genai.Client(api_key=Config.GENAI_API_KEY)
                logger.info("Google GenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize GenAI client: {e}")
        
        # Anthropic would go here if available
        if Config.ANTHROPIC_API_KEY:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")

    def query_open_source(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Query currently loaded open source model"""
        if self.current_model is None or self.current_tokenizer is None:
            raise ValueError("No open source model loaded")
        
        max_tokens = max_tokens or Config.MAX_NEW_TOKENS
        temperature = temperature or Config.TEMPERATURE
        
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
        temperature = temperature or Config.TEMPERATURE
        
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
        
        temperature = temperature or Config.TEMPERATURE
        max_tokens = max_tokens or Config.MAX_NEW_TOKENS
        
        logger.debug(f"Querying GenAI model {model} with prompt length: {len(prompt)}")
        
        try:
            response = self.genai_client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai.GenerateContentConfig(
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
    
    def is_model_available(self, model_name: str, model_type: str) -> bool:
        """Check if a specific model is available"""
        if model_type == 'open_source':
            return model_name in self.open_source_models and self.unsloth_available
        elif model_type == 'api':
            if 'gpt' in model_name:
                return self.openai_client is not None
            elif 'gemini' in model_name:
                return self.genai_client is not None
            elif 'claude' in model_name:
                return self.anthropic_client is not None
        return False
    
    def get_available_models(self) -> dict:
        """Get dictionary of all available models"""
        available = {
            'open_source': [],
            'api': []
        }
        
        if self.unsloth_available:
            available['open_source'] = list(self.open_source_models.keys())
        
        if self.openai_client:
            available['api'].extend([m for m in self.api_models.keys() if 'gpt' in m])
        
        if self.genai_client:
            available['api'].extend([m for m in self.api_models.keys() if 'gemini' in m])
        
        if self.anthropic_client:
            available['api'].extend([m for m in self.api_models.keys() if 'claude' in m])
        
        return available
    
    def get_model_info(self) -> dict:
        """Get information about currently loaded model"""
        info = {
            'current_model': self.current_model_name,
            'model_loaded': self.current_model is not None,
            'embedding_model_loaded': self.embedding_model is not None,
            'api_clients': {
                'openai': self.openai_client is not None,
                'genai': self.genai_client is not None,
                'anthropic': self.anthropic_client is not None
            },
            'unsloth_available': self.unsloth_available
        }
        
        return info
