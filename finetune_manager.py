import os
import json
import torch
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random

from config import Config
from utils import setup_logging, clear_gpu_memory, check_gpu_availability
from dataset_manager import DatasetManager
from prompt_manager import PromptManager
from models import ModelManager

logger = setup_logging("finetune_manager")

class FinetuneManager:
    """
    Centralized finetuning management that integrates with existing system infrastructure.
    
    Key features:
    - Reuses existing PromptManager and DatasetManager for consistency
    - Supports both Unsloth and HuggingFace training pipelines
    - Auto-generates training data using existing prompt templates
    - Handles model registration and integration
    - Provides comprehensive validation and progress tracking
    """
    
    def __init__(self):
        """Initialize finetune manager with existing system components"""
        self.dataset_manager = DatasetManager()
        self.prompt_manager = PromptManager()
        self.model_manager = ModelManager()
        
        # Load configurations
        self.finetune_config = self._load_finetune_config()
        self.models_config = Config.load_models_config()
        
        logger.info(f"FinetuneManager initialized with {len(self.finetune_config)} finetune configurations")
    
    def _load_finetune_config(self) -> Dict[str, Any]:
        """Load finetune configuration from JSON file"""
        finetune_json_path = Config.FINETUNES_JSON
        
        try:
            with open(finetune_json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Finetune configuration file not found: {finetune_json_path}")
            logger.info("Creating empty finetune configuration")
            return {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in finetune configuration: {e}")
    
    def validate_finetune_request(self, model_name: str, tune_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a finetune request before starting training.
        
        Args:
            model_name: Name of base model to finetune
            tune_name: Name of finetune configuration
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Check if tune configuration exists
        if tune_name not in self.finetune_config:
            errors.append(f"Unknown finetune configuration: {tune_name}")
            return False, errors
        
        # Check if base model exists and is local
        if model_name not in self.models_config:
            errors.append(f"Unknown base model: {model_name}")
            return False, errors
        
        model_config = self.models_config[model_name]
        if model_config.get('type') != 'local':
            errors.append(f"Can only finetune local models, but {model_name} is type: {model_config.get('type')}")
            return False, errors
        
        # Check GPU availability
        gpu_status = check_gpu_availability()
        if not gpu_status['cuda_available']:
            errors.append("CUDA GPU required for finetuning but not available")
            return False, errors
        
        if not gpu_status['can_run_local_models']:
            errors.append(f"Insufficient GPU memory ({gpu_status['total_memory']:.1f} GB) for finetuning")
        
        # Validate finetune configuration
        tune_config = self.finetune_config[tune_name]
        
        setup_name = tune_config.get('setup')
        if not setup_name:
            errors.append(f"Finetune config '{tune_name}' missing required 'setup' field")
            return False, errors
        
        prompt_name = tune_config.get('prompt')
        if not prompt_name:
            errors.append(f"Finetune config '{tune_name}' missing required 'prompt' field")
            return False, errors
        
        # Validate setup and prompt compatibility
        try:
            prompt_requirements = self.prompt_manager.get_prompt_field_requirements(prompt_name)
            if prompt_requirements.get('setup_name') != setup_name:
                errors.append(f"Prompt '{prompt_name}' is not compatible with setup '{setup_name}'")
        except Exception as e:
            errors.append(f"Error validating prompt '{prompt_name}': {e}")
        
        # Check if dataset is available
        if not self.dataset_manager.is_dataset_downloaded(setup_name):
            errors.append(f"Dataset for setup '{setup_name}' not downloaded")
        
        # Check disk space (rough estimate)
        try:
            import shutil
            free_space_gb = shutil.disk_usage(Config.FINETUNED_MODELS_DIR)[2] / (1024**3)
            if free_space_gb < 10:  # Need at least 10GB free
                errors.append(f"Insufficient disk space ({free_space_gb:.1f} GB free, need at least 10 GB)")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        return len(errors) == 0, errors
    
    def prepare_training_data(self, tune_name: str, max_samples: Optional[int] = None) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
        """
        Prepare training and validation data using existing system components.
        
        Args:
            tune_name: Name of finetune configuration
            max_samples: Optional limit on number of samples
            
        Returns:
            Tuple[List[Dict], List[Dict], Dict]: (train_data, val_data, info)
        """
        tune_config = self.finetune_config[tune_name]
        setup_name = tune_config['setup']
        prompt_name = tune_config['prompt']
        
        training_config = tune_config.get('training_config', {})
        train_test_split = training_config.get('train_test_split', 0.8)
        validation_split = training_config.get('validation_split', 0.1)
        config_max_samples = training_config.get('max_samples')
        
        # Use the smaller of provided max_samples or config max_samples
        if max_samples is not None and config_max_samples is not None:
            max_samples = min(max_samples, config_max_samples)
        elif config_max_samples is not None:
            max_samples = config_max_samples
        
        logger.info(f"Preparing training data for {tune_name}: setup={setup_name}, prompt={prompt_name}")
        
        # Load dataset
        dataset = self.dataset_manager.load_dataset(setup_name)
        if dataset is None:
            raise ValueError(f"Could not load dataset for setup: {setup_name}")
        
        # Apply dataset filtering/pruning if configured
        if self.dataset_manager.has_pruning_config(setup_name):
            dataset, skipped_count, _ = self.dataset_manager.filter_dataset_rows(dataset, setup_name)
            logger.info(f"Applied pruning: kept {len(dataset)} rows, skipped {skipped_count} rows")
        
        # Limit samples if specified
        if max_samples and len(dataset) > max_samples:
            logger.info(f"Limiting dataset from {len(dataset)} to LAST {max_samples} samples for training")
            dataset = dataset.tail(max_samples).reset_index(drop=True)
                
        # Generate training examples using existing prompt infrastructure
        training_examples = []
        failed_count = 0
        
        prompt_requirements = self.prompt_manager.get_prompt_field_requirements(prompt_name)
        prompt_mode = prompt_requirements['mode']
        
        logger.info(f"Generating training examples using mode: {prompt_mode}")
        
        for idx, row in dataset.iterrows():
            try:
                # Generate input using existing prompt formatting
                input_text = self.prompt_manager.prepare_prompt_for_row(
                    prompt_name=prompt_name,
                    row=row,
                    setup_name=setup_name,
                    mode=prompt_mode,
                    dataset=dataset,
                    few_shot_row=None  # Don't use few-shot for training data generation
                )
                
                # Get expected output using existing data extraction
                output_text = self.dataset_manager.get_expected_answer(row, setup_name)
                
                if not output_text or output_text.strip() == "":
                    failed_count += 1
                    logger.debug(f"Skipping row {idx}: empty expected answer")
                    continue
                
                training_examples.append({
                    'input': input_text,
                    'output': output_text,
                    'row_index': idx
                })
                
            except Exception as e:
                failed_count += 1
                logger.debug(f"Error processing row {idx}: {e}")
                continue
        
        if not training_examples:
            raise ValueError("No valid training examples could be generated")
        
        logger.info(f"Generated {len(training_examples)} training examples ({failed_count} failed)")
        
        # Split into train/validation sets
        random.seed(Config.RANDOM_SEED)
        random.shuffle(training_examples)
        
        total_examples = len(training_examples)
        train_size = int(total_examples * train_test_split)
        val_size = int(total_examples * validation_split)
        
        # Ensure we have at least some validation data
        if val_size == 0 and total_examples > 1:
            val_size = 1
            train_size = total_examples - 1
        
        train_data = training_examples[:train_size]
        val_data = training_examples[train_size:train_size + val_size]
        
        info = {
            'tune_name': tune_name,
            'setup_name': setup_name,
            'prompt_name': prompt_name,
            'prompt_mode': prompt_mode,
            'total_examples': total_examples,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'failed_count': failed_count,
            'original_dataset_size': len(dataset)
        }
        
        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} validation")
        
        return train_data, val_data, info
    
    def format_training_data(self, training_data: List[Dict], model_config: Dict[str, Any]) -> List[Dict]:
        """
        Format training data according to model requirements (Unsloth vs HuggingFace).
        
        Args:
            training_data: List of training examples with 'input' and 'output' fields
            model_config: Configuration of the base model being finetuned
            
        Returns:
            List[Dict]: Formatted training data ready for the training pipeline
        """
        use_unsloth = model_config.get('use_unsloth', False)
        use_chat_template = model_config.get('use_chat_template', True)
        
        formatted_data = []
        
        if use_unsloth and use_chat_template:
            # Unsloth format with chat messages
            logger.info("Formatting training data for Unsloth with chat templates")
            for example in training_data:
                formatted_example = {
                    "conversations": [
                        {"from": "human", "value": example['input']},
                        {"from": "gpt", "value": example['output']}
                    ]
                }
                formatted_data.append(formatted_example)
                
        elif use_chat_template:
            # HuggingFace format with chat messages
            logger.info("Formatting training data for HuggingFace with chat templates")
            for example in training_data:
                formatted_example = {
                    "messages": [
                        {"role": "user", "content": example['input']},
                        {"role": "assistant", "content": example['output']}
                    ]
                }
                formatted_data.append(formatted_example)
                
        else:
            # Simple text format
            logger.info("Formatting training data as simple text")
            for example in training_data:
                formatted_example = {
                    "text": f"### Human: {example['input']}\n\n### Assistant: {example['output']}"
                }
                formatted_data.append(formatted_example)
        
        return formatted_data
    
    def run_unsloth_training(self, model_name: str, tune_name: str, train_data: List[Dict], 
                           val_data: List[Dict], hyperparameters: Dict[str, Any]) -> str:
        """
        Run training using Unsloth framework.
        
        Args:
            model_name: Name of base model
            tune_name: Name of finetune configuration  
            train_data: Formatted training data
            val_data: Formatted validation data
            hyperparameters: Training hyperparameters
            
        Returns:
            str: Path to saved finetuned model
        """
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from datasets import Dataset
        except ImportError as e:
            raise ImportError(f"Unsloth training requires additional packages: {e}\nInstall with: pip install unsloth trl")
        
        model_config = self.models_config[model_name]
        model_path = model_config['model_path']
        
        logger.info(f"Starting Unsloth training: {model_name} -> {tune_name}")
        
        # Clear any existing models
        self.model_manager.cleanup_current_model()
        clear_gpu_memory()
        
        # Load model for training
        logger.info(f"Loading model for training: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=hyperparameters.get('max_seq_length', Config.MAX_SEQ_LENGTH),
            load_in_4bit=model_config.get('load_in_4bit', False),
            token=Config.HF_ACCESS_TOKEN if Config.HF_ACCESS_TOKEN else None,
            cache_dir=Config.CACHED_MODELS_DIR
        )
        
        # Configure LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=hyperparameters.get('lora_r', 16),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=hyperparameters.get('lora_alpha', 32),
            lora_dropout=hyperparameters.get('lora_dropout', 0.1),
            bias="none",
            use_gradient_checkpointing="unsloth"
        )
        
        # Prepare datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data) if val_data else None
        
        # Define formatting function for Unsloth
        # FIX: Check if model has chat template before using it
        has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
        
        def formatting_func(examples):
            """Format training examples for Unsloth"""
            # Check if this is a single example or batched
            # Single: {"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}
            # Batched: {"conversations": [[{"from": "human", ...}, ...], [{"from": "human", ...}, ...]]}
            
            if "conversations" not in examples:
                raise ValueError("Examples must contain 'conversations' field")
            
            conversations_data = examples["conversations"]
            
            # Detect if this is a single example or batch
            # Single example: conversations is a list of dicts (conversation turns)
            # Batched: conversations is a list of lists of dicts
            is_single = False
            if conversations_data and isinstance(conversations_data[0], dict):
                # First element is a dict, so this is a single conversation
                is_single = True
                conversations_list = [conversations_data]
            else:
                # First element is a list, so this is batched
                conversations_list = conversations_data
            
            texts = []
            
            for conversations in conversations_list:
                if has_chat_template:
                    try:
                        text = tokenizer.apply_chat_template(
                            conversations,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                    except Exception as e:
                        logger.warning(f"Chat template failed, using simple format: {e}")
                        # Fallback to simple format
                        text = ""
                        for msg in conversations:
                            role = "Human" if msg.get("from") == "human" else "Assistant"
                            text += f"### {role}: {msg.get('value', '')}\n\n"
                else:
                    # Simple format without chat template
                    text = ""
                    for msg in conversations:
                        role = "Human" if msg.get("from") == "human" else "Assistant"
                        text += f"### {role}: {msg.get('value', '')}\n\n"
                texts.append(text)
            
            # Return single text if single example, list if batched
            return texts
        
        # Training arguments
        output_dir = os.path.join(Config.FINETUNED_MODELS_DIR, f"{model_name}_ft_{tune_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            per_device_train_batch_size=hyperparameters.get('batch_size', 4),
            gradient_accumulation_steps=hyperparameters.get('gradient_accumulation_steps', 4),
            warmup_steps=hyperparameters.get('warmup_steps', 100),
            num_train_epochs=hyperparameters.get('num_epochs', 3),
            learning_rate=hyperparameters.get('learning_rate', 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=Config.RANDOM_SEED,
            output_dir=output_dir,
            save_strategy="epoch",
            eval_strategy="epoch" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False,
            report_to="none"  # Disable wandb
        )
        
        # Initialize trainer
        # FIX: Use formatting_prompts_func instead of formatting_func for better compatibility
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            formatting_func=formatting_func,
            max_seq_length=hyperparameters.get('max_seq_length', Config.MAX_SEQ_LENGTH),
            dataset_num_proc=2,
            packing=False,
            args=training_args
        )
        
        # Run training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info(f"Saving finetuned model to: {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Clean up
        del model, tokenizer, trainer
        clear_gpu_memory()
        
        return output_dir
    
    def run_huggingface_training(self, model_name: str, tune_name: str, train_data: List[Dict], 
                                val_data: List[Dict], hyperparameters: Dict[str, Any]) -> str:
        """
        Run training using standard HuggingFace framework.
        
        Args:
            model_name: Name of base model
            tune_name: Name of finetune configuration
            train_data: Formatted training data
            val_data: Formatted validation data  
            hyperparameters: Training hyperparameters
            
        Returns:
            str: Path to saved finetuned model
        """
        try:
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer, TrainingArguments, 
                Trainer, DataCollatorForLanguageModeling
            )
            from datasets import Dataset
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            raise ImportError(f"HuggingFace training requires additional packages: {e}\nInstall with: pip install transformers peft")
        
        model_config = self.models_config[model_name]
        model_path = model_config['model_path']
        
        logger.info(f"Starting HuggingFace training: {model_name} -> {tune_name}")
        
        # Clear any existing models
        self.model_manager.cleanup_current_model()
        clear_gpu_memory()
        
        # Load model and tokenizer
        logger.info(f"Loading model for training: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=Config.HF_ACCESS_TOKEN if Config.HF_ACCESS_TOKEN else None,
            cache_dir=Config.CACHED_MODELS_DIR,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=Config.HF_ACCESS_TOKEN if Config.HF_ACCESS_TOKEN else None,
            cache_dir=Config.CACHED_MODELS_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=hyperparameters.get('lora_r', 16),
            lora_alpha=hyperparameters.get('lora_alpha', 32),
            lora_dropout=hyperparameters.get('lora_dropout', 0.1),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # FIX: Proper batched tokenization function
        has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
        
        def tokenize_function(examples):
            """Tokenize examples with proper batch handling"""
            texts = []
            
            # Check if this is batched data (dict with lists) or single example
            if "messages" in examples:
                messages_list = examples["messages"]
                
                # Handle batched format
                if isinstance(messages_list, list) and len(messages_list) > 0:
                    if isinstance(messages_list[0], list):
                        # Batched: list of message lists
                        for messages in messages_list:
                            if has_chat_template:
                                try:
                                    text = tokenizer.apply_chat_template(
                                        messages,
                                        tokenize=False,
                                        add_generation_prompt=False
                                    )
                                except Exception as e:
                                    logger.warning(f"Chat template failed, using simple format: {e}")
                                    text = ""
                                    for msg in messages:
                                        role = "Human" if msg["role"] == "user" else "Assistant"
                                        text += f"### {role}: {msg['content']}\n\n"
                            else:
                                text = ""
                                for msg in messages:
                                    role = "Human" if msg["role"] == "user" else "Assistant"
                                    text += f"### {role}: {msg['content']}\n\n"
                            texts.append(text)
                    else:
                        # Single example wrapped in dict
                        messages = messages_list
                        if has_chat_template:
                            try:
                                text = tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=False
                                )
                            except Exception as e:
                                logger.warning(f"Chat template failed, using simple format: {e}")
                                text = ""
                                for msg in messages:
                                    role = "Human" if msg["role"] == "user" else "Assistant"
                                    text += f"### {role}: {msg['content']}\n\n"
                        else:
                            text = ""
                            for msg in messages:
                                role = "Human" if msg["role"] == "user" else "Assistant"
                                text += f"### {role}: {msg['content']}\n\n"
                        texts.append(text)
                        
            elif "text" in examples:
                # Simple text format
                if isinstance(examples["text"], list):
                    texts = examples["text"]
                else:
                    texts = [examples["text"]]
            else:
                raise ValueError("Dataset must have either 'messages' or 'text' field")
            
            # Tokenize all texts
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=hyperparameters.get('max_seq_length', Config.MAX_SEQ_LENGTH),
                return_tensors=None  # Return lists, not tensors
            )
            
            # Add labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        
        # FIX: Use batched=True with proper batch size for efficiency
        logger.info("Tokenizing training data...")
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training data"
        )
        
        val_dataset = None
        if val_data:
            val_dataset = Dataset.from_list(val_data)
            logger.info("Tokenizing validation data...")
            val_dataset = val_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=100,
                remove_columns=val_dataset.column_names,
                desc="Tokenizing validation data"
            )
        
        # Training arguments
        output_dir = os.path.join(Config.FINETUNED_MODELS_DIR, f"{model_name}_ft_{tune_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=hyperparameters.get('batch_size', 4),
            gradient_accumulation_steps=hyperparameters.get('gradient_accumulation_steps', 4),
            warmup_steps=hyperparameters.get('warmup_steps', 100),
            num_train_epochs=hyperparameters.get('num_epochs', 3),
            learning_rate=hyperparameters.get('learning_rate', 2e-4),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False,
            report_to="none",
            seed=Config.RANDOM_SEED
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # Run training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info(f"Saving finetuned model to: {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Clean up
        del model, tokenizer, trainer
        clear_gpu_memory()
        
        return output_dir
    
    def register_finetuned_model(self, base_model_name: str, tune_name: str, 
                                model_path: str, info: Dict[str, Any]):
        """
        Register a newly finetuned model in the models configuration.
        
        Args:
            base_model_name: Name of the base model that was finetuned
            tune_name: Name of the finetune configuration used
            model_path: Path to the saved finetuned model
            info: Training information and metadata
        """
        finetuned_model_name = f"{base_model_name}_ft_{tune_name}"
        
        # Get base model config
        base_config = self.models_config[base_model_name].copy()
        
        # FIX: Use absolute path for model_path
        absolute_model_path = os.path.abspath(model_path)
        
        # Create finetuned model config
        finetuned_config = {
            **base_config,
            "model_path": absolute_model_path,
            "description": f"Finetuned {base_config['description']} for {info['setup_name']} task",
            "finetuned": True,
            "base_model": base_model_name,
            "tune_name": tune_name,
            "training_info": {
                "setup_name": info['setup_name'],
                "prompt_name": info['prompt_name'],
                "train_size": info['train_size'],
                "val_size": info['val_size'],
                "timestamp": datetime.now().isoformat()
            },
            "local_files_only": True  # Finetuned models are always local
        }
        
        # Update models config
        models_config_path = Config.MODELS_JSON
        
        # Load current config
        with open(models_config_path, 'r') as f:
            current_models = json.load(f)
        
        # Add new model
        current_models[finetuned_model_name] = finetuned_config
        
        # Save updated config
        with open(models_config_path, 'w') as f:
            json.dump(current_models, f, indent=2)
        
        # Update our cached config
        self.models_config[finetuned_model_name] = finetuned_config
        
        logger.info(f"Registered finetuned model: {finetuned_model_name}")
        logger.info(f"Model path: {absolute_model_path}")
        logger.info(f"Training details: {info['train_size']} train samples, {info['val_size']} val samples")
    
    def run_finetune(self, model_name: str, tune_name: str, max_samples: Optional[int] = None) -> bool:
        """
        Run complete finetuning pipeline.
        
        Args:
            model_name: Name of base model to finetune
            tune_name: Name of finetune configuration
            max_samples: Optional limit on training samples
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate request
            is_valid, errors = self.validate_finetune_request(model_name, tune_name)
            if not is_valid:
                for error in errors:
                    logger.error(error)
                return False
            
            logger.info(f"Starting finetune: {model_name} -> {tune_name}")
            
            # Get configurations
            tune_config = self.finetune_config[tune_name]
            model_config = self.models_config[model_name]
            hyperparameters = tune_config.get('hyperparameters', {})
            
            # Prepare training data
            logger.info("Preparing training data...")
            train_data, val_data, info = self.prepare_training_data(tune_name, max_samples)
            
            # Format data for training framework
            logger.info("Formatting training data...")
            formatted_train_data = self.format_training_data(train_data, model_config)
            formatted_val_data = self.format_training_data(val_data, model_config) if val_data else []
            
            # Run training based on model configuration
            use_unsloth = model_config.get('use_unsloth', False)
            
            if use_unsloth:
                logger.info("Using Unsloth training pipeline")
                model_path = self.run_unsloth_training(
                    model_name, tune_name, formatted_train_data, formatted_val_data, hyperparameters
                )
            else:
                logger.info("Using HuggingFace training pipeline") 
                model_path = self.run_huggingface_training(
                    model_name, tune_name, formatted_train_data, formatted_val_data, hyperparameters
                )
            
            # Register the finetuned model
            logger.info("Registering finetuned model...")
            self.register_finetuned_model(model_name, tune_name, model_path, info)
            
            logger.info(f"Finetuning completed successfully!")
            logger.info(f"Finetuned model: {model_name}_ft_{tune_name}")
            logger.info(f"Model path: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Finetuning failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            # Always clean up GPU memory
            clear_gpu_memory()
    
    def list_finetune_configurations(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available finetune configurations.
        
        Returns:
            Dict[str, Dict]: Configuration details for each finetune setup
        """
        configurations = {}
        
        for tune_name, config in self.finetune_config.items():
            info = {
                'description': config.get('description', 'No description'),
                'setup': config.get('setup'),
                'prompt': config.get('prompt'),
                'hyperparameters': config.get('hyperparameters', {}),
                'training_config': config.get('training_config', {})
            }
            configurations[tune_name] = info
        
        return configurations
    
    def list_finetuned_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all finetuned models in the system.
        
        Returns:
            Dict[str, Dict]: Information about each finetuned model
        """
        finetuned_models = {}
        
        for model_name, config in self.models_config.items():
            if config.get('finetuned', False):
                info = {
                    'description': config.get('description'),
                    'base_model': config.get('base_model'),
                    'tune_name': config.get('tune_name'),
                    'model_path': config.get('model_path'),
                    'training_info': config.get('training_info', {})
                }
                finetuned_models[model_name] = info
        
        return finetuned_models