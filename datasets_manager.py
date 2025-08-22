import os
import json
import pandas as pd
from typing import Dict, Optional, List, Any
from config import Config
from utils import setup_logging, download_dataset
from prompts import PromptManager

logger = setup_logging("dataset_manager")

class DatasetManager:
    """Centralized dataset management with flexible field mappings"""
    
    def __init__(self, prompt_manager: PromptManager):
        self.datasets = {}
        self.current_dataset = None
        self.current_dataset_name = None
        self.current_dataset_config = None
        self.prompt_manager = prompt_manager
        
        # Load dataset configurations from JSON file
        self.dataset_configs = self.load_dataset_configs()
        
        logger.info(f"DatasetManager initialized with {len(self.dataset_configs)} dataset configurations")
    
    def load_dataset_configs(self) -> Dict[str, Dict]:
        """Load dataset configurations from datasets.json"""
        try:
            with open(Config.DATASETS_JSON, 'r') as f:
                datasets_list = json.load(f)
            
            # Convert list to dict with name as key
            configs = {}
            for dataset in datasets_list:
                name = dataset['name'].lower().replace('-', '_')  # Normalize name
                configs[name] = dataset
                
            logger.info(f"Loaded {len(configs)} dataset configurations from {Config.DATASETS_JSON}")
            return configs
            
        except FileNotFoundError:
            logger.error(f"Dataset configuration file {Config.DATASETS_JSON} not found")
            return {}
        except Exception as e:
            logger.error(f"Error loading dataset configurations: {e}")
            return {}
    
    def ensure_dataset_downloaded(self, dataset_name: str) -> bool:
        """Ensure dataset is downloaded, download if necessary"""
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        config = self.dataset_configs[dataset_name]
        dataset_path = self.get_dataset_path(dataset_name)
        
        # Check if dataset file exists
        if os.path.exists(dataset_path):
            logger.info(f"Dataset {dataset_name} already exists at {dataset_path}")
            return True
        
        # Check if storage folder exists but file doesn't
        storage_path = os.path.join(Config.DATA_DIR, config['storage_folder'])
        if os.path.exists(storage_path):
            logger.warning(f"Storage folder exists but dataset file not found at {dataset_path}")
            logger.info("Checking for alternative paths...")
            
            # Try to find CSV files in the storage folder
            csv_files = []
            for root, dirs, files in os.walk(storage_path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if csv_files:
                logger.info(f"Found {len(csv_files)} CSV files in storage folder:")
                for csv_file in csv_files:
                    logger.info(f"  - {csv_file}")
                logger.warning("Please update the 'inner_path' in datasets.json if needed")
        
        # Download dataset
        logger.info(f"Downloading dataset {dataset_name}...")
        try:
            # Create a temporary datasets list for download_dataset function
            temp_config = [{
                'name': config['name'],
                'link': config['link'],
                'storage_folder': config['storage_folder']
            }]
            
            # Write temporary config file
            temp_config_path = 'temp_dataset_config.json'
            with open(temp_config_path, 'w') as f:
                json.dump(temp_config, f)
            
            # Download
            download_dataset(temp_config_path, Config.DATA_DIR)
            
            # Clean up temp file
            os.remove(temp_config_path)
            
            # Check if download was successful
            if os.path.exists(dataset_path):
                logger.info(f"Successfully downloaded dataset {dataset_name}")
                return True
            else:
                logger.error(f"Download completed but file not found at {dataset_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {e}")
            return False
    
    def get_dataset_path(self, dataset_name: str) -> str:
        """Get the full path to a dataset file"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        return os.path.join(Config.DATA_DIR, config['storage_folder'], config['inner_path'])
    
    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load a specific dataset by name"""
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Normalize dataset name
        dataset_name = dataset_name.lower().replace('-', '_')
        
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            logger.info(f"Available datasets: {list(self.dataset_configs.keys())}")
            return None
        
        # Ensure dataset is downloaded
        if not self.ensure_dataset_downloaded(dataset_name):
            logger.error(f"Could not ensure dataset {dataset_name} is available")
            return None
        
        # Load the dataset
        dataset_path = self.get_dataset_path(dataset_name)
        
        try:
            dataset = pd.read_csv(dataset_path)
            logger.info(f"Loaded {dataset_name} dataset with {len(dataset)} entries")
            logger.info(f"Columns: {list(dataset.columns)}")
            
            # Store dataset and configuration
            self.datasets[dataset_name] = dataset
            self.current_dataset = dataset
            self.current_dataset_name = dataset_name
            self.current_dataset_config = self.dataset_configs[dataset_name]
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name} from {dataset_path}: {e}")
            return None
    
    def validate_prompt_compatibility(self, dataset_name: str, prompt_key: str) -> bool:
        """Check if a prompt is compatible with a dataset"""
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        config = self.dataset_configs[dataset_name]
        compatible_prompts = config.get('compatible_prompts', [])
        
        # Also allow general prompts that don't have dataset-specific prefixes
        general_prompts = [p for p in self.prompt_manager.prompts.keys() 
                          if not any(p.startswith(prefix) for prefix in ['gmeg_', 'fungi_', 'hate_', 'hardness_', 'reframing_'])]
        
        is_compatible = prompt_key in compatible_prompts or prompt_key in general_prompts
        
        if not is_compatible:
            logger.warning(f"Prompt '{prompt_key}' not compatible with dataset '{dataset_name}'")
            logger.info(f"Compatible prompts: {compatible_prompts}")
            logger.info(f"General prompts: {general_prompts}")
        
        return is_compatible
    
    def prepare_question_from_row(self, row: pd.Series, dataset_name: str = None) -> str:
        """Prepare question text from a dataset row using field mapping"""
        dataset_name = dataset_name or self.current_dataset_name
        
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        field_mapping = config['field_mapping']
        
        # Get question fields
        question_fields = field_mapping['question_fields']
        question_template = field_mapping['question_template']
        
        # Build kwargs for template
        template_kwargs = {}
        for field in question_fields:
            if field in row:
                template_kwargs[field] = str(row[field])
            else:
                logger.warning(f"Field '{field}' not found in row {row.name} for dataset {dataset_name}")
                template_kwargs[field] = "N/A"
        
        try:
            question = question_template.format(**template_kwargs)
            logger.debug(f"Prepared question for row {row.name} in dataset {dataset_name}")
            return question
        except Exception as e:
            logger.error(f"Error formatting question template for row {row.name}: {e}")
            # Fallback: just concatenate the fields
            fallback_question = " | ".join([f"{field}: {template_kwargs.get(field, 'N/A')}" 
                                          for field in question_fields])
            logger.warning(f"Using fallback question format: {fallback_question}")
            return fallback_question
    
    def prepare_prompt(self, row: pd.Series, prompt_key: str, dataset_name: str = None, **kwargs) -> str:
        """Prepare prompt using generic field mapping"""
        dataset_name = dataset_name or self.current_dataset_name
        
        if not dataset_name:
            raise ValueError("No dataset specified and no current dataset set")
        
        # Check prompt compatibility
        if not self.validate_prompt_compatibility(dataset_name, prompt_key):
            logger.warning(f"Using potentially incompatible prompt '{prompt_key}' with dataset '{dataset_name}'")
        
        # Prepare question text
        question_text = self.prepare_question_from_row(row, dataset_name)
        
        # For generic prompts, use standard field names
        prompt_kwargs = {
            'question_text': question_text,
            'input_text': question_text,  # Alternative name
            **kwargs
        }
        
        # For dataset-specific prompts, also include original fields
        if dataset_name in self.dataset_configs:
            config = self.dataset_configs[dataset_name]
            field_mapping = config['field_mapping']
            
            for field in field_mapping['question_fields']:
                if field in row:
                    prompt_kwargs[field] = str(row[field])
        
        # Handle few-shot prompts
        if 'few_shot' in prompt_key and 'few_shot_examples' not in prompt_kwargs:
            if self.current_dataset is not None:
                prompt_kwargs['few_shot_examples'] = self.prompt_manager.generate_few_shot_examples(
                    self.current_dataset, n_examples=3
                )
        
        try:
            prompt = self.prompt_manager.get_prompt(prompt_key, **prompt_kwargs)
            logger.debug(f"Prepared prompt for row {row.name} using template {prompt_key}")
            return prompt
        except Exception as e:
            logger.error(f"Error preparing prompt for row {row.name}: {e}")
            raise
    
    def get_expected_output(self, row: pd.Series, dataset_name: str = None) -> str:
        """Get expected output for evaluation based on dataset configuration"""
        dataset_name = dataset_name or self.current_dataset_name
        
        if not dataset_name or dataset_name not in self.dataset_configs:
            logger.warning(f"Unknown dataset: {dataset_name}, using fallback fields")
            return self._get_fallback_expected_output(row)
        
        config = self.dataset_configs[dataset_name]
        field_mapping = config['field_mapping']
        answer_field = field_mapping['answer_field']
        
        if answer_field in row:
            result = str(row[answer_field])
            logger.debug(f"Retrieved expected output for row {row.name} from field '{answer_field}'")
            return result
        else:
            logger.warning(f"Answer field '{answer_field}' not found in row {row.name}")
            return self._get_fallback_expected_output(row)
    
    def _get_fallback_expected_output(self, row: pd.Series) -> str:
        """Get expected output using fallback field names"""
        fallback_fields = ['explanation', 'answer', 'target', 'label', 'human_explanation', 'expected_output']
        
        for field in fallback_fields:
            if field in row:
                result = str(row[field])
                logger.debug(f"Retrieved expected output from fallback field '{field}' for row {row.name}")
                return result
        
        logger.warning(f"No expected output field found for row {row.name}")
        return "No explanation available"
    
    def validate_dataset(self, dataset: pd.DataFrame, dataset_name: str = None) -> bool:
        """Validate that dataset has required fields for evaluation"""
        dataset_name = dataset_name or self.current_dataset_name
        
        if not dataset_name:
            logger.warning("No dataset name provided for validation")
            return True  # Allow validation to pass
        
        logger.info(f"Validating dataset '{dataset_name}'")
        
        if dataset_name not in self.dataset_configs:
            logger.warning(f"Unknown dataset '{dataset_name}', performing basic validation")
            return len(dataset) > 0
        
        config = self.dataset_configs[dataset_name]
        field_mapping = config['field_mapping']
        
        # Check required fields
        required_fields = field_mapping['question_fields'] + [field_mapping['answer_field']]
        missing_fields = [field for field in required_fields if field not in dataset.columns]
        
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")
            logger.info(f"Available columns: {list(dataset.columns)}")
            
            # Check if we can find similar field names
            available_cols = list(dataset.columns)
            for missing_field in missing_fields:
                similar_cols = [col for col in available_cols 
                               if missing_field.lower() in col.lower() or col.lower() in missing_field.lower()]
                if similar_cols:
                    logger.info(f"Similar columns to '{missing_field}': {similar_cols}")
            
            return False
        
        logger.info("Dataset validation completed successfully")
        return True
    
    def get_sample(self, dataset_name: str = None, size: int = None) -> pd.DataFrame:
        """Get a sample from the dataset"""
        if dataset_name:
            dataset = self.datasets.get(dataset_name)
        else:
            dataset = self.current_dataset
        
        if dataset is None:
            raise ValueError("No dataset loaded")
        
        size = size or Config.SAMPLE_SIZE
        sample_size = min(size, len(dataset))
        
        logger.info(f"Sampling {sample_size} entries from dataset of {len(dataset)} total entries")
        return dataset.sample(sample_size)
    
    def get_dataset_info(self, dataset_name: str = None) -> Dict:
        """Get information about a dataset"""
        if dataset_name:
            dataset = self.datasets.get(dataset_name)
            config = self.dataset_configs.get(dataset_name, {})
            name = dataset_name
        else:
            dataset = self.current_dataset
            config = self.current_dataset_config or {}
            name = self.current_dataset_name or "current"
        
        if dataset is None:
            return {"error": "No dataset loaded"}
        
        info = {
            'name': name,
            'shape': dataset.shape,
            'columns': list(dataset.columns),
            'memory_usage': dataset.memory_usage(deep=True).sum(),
            'dtypes': dataset.dtypes.to_dict(),
            'null_counts': dataset.isnull().sum().to_dict(),
            'sample_row': dataset.iloc[0].to_dict() if len(dataset) > 0 else None,
            'config': config
        }
        
        # Add dataset-specific analysis
        if config:
            field_mapping = config.get('field_mapping', {})
            answer_field = field_mapping.get('answer_field')
            
            if answer_field and answer_field in dataset.columns:
                # Count NA/missing annotations
                na_patterns = ['na', 'n/a', 'not applicable', 'not annotatable', '', 'none']
                na_count = dataset[answer_field].fillna('').astype(str).str.lower().str.strip().isin(na_patterns).sum()
                info['na_annotations'] = int(na_count)
                info['valid_annotations'] = len(dataset) - int(na_count)
                
                # Get answer field statistics
                answer_lengths = dataset[answer_field].fillna('').astype(str).str.len()
                info['answer_stats'] = {
                    'mean_length': answer_lengths.mean(),
                    'median_length': answer_lengths.median(),
                    'min_length': answer_lengths.min(),
                    'max_length': answer_lengths.max()
                }
        
        return info
    
    def prepare_dataset_for_experiment(self, dataset_name: str, prompt_key: str, 
                                     sample_size: int = None) -> Dict:
        """Prepare dataset for experiment with prompts and expected outputs"""
        logger.info(f"Preparing dataset '{dataset_name}' for experiment with prompt '{prompt_key}'")
        
        # Load dataset if not already loaded
        if dataset_name not in self.datasets:
            dataset = self.load_dataset(dataset_name)
            if dataset is None:
                raise ValueError(f"Could not load dataset '{dataset_name}'")
        else:
            dataset = self.datasets[dataset_name]
        
        # Validate dataset
        if not self.validate_dataset(dataset, dataset_name):
            logger.warning("Dataset validation failed, continuing with available fields...")
        
        # Check prompt compatibility
        if not self.validate_prompt_compatibility(dataset_name, prompt_key):
            logger.warning(f"Prompt '{prompt_key}' may not be fully compatible with dataset '{dataset_name}'")
        
        # Get sample
        sample_data = self.get_sample(dataset_name, sample_size)
        logger.info(f"Using {len(sample_data)} samples for experiment")
        
        # Prepare all prompts and expected outputs
        prepared_data = {
            'prompts': [],
            'expected_outputs': [],
            'sample_indices': [],
            'row_data': [],
            'questions': []  # Store the extracted questions for analysis
        }
        
        for idx, row in sample_data.iterrows():
            try:
                # Prepare prompt
                prompt = self.prepare_prompt(row, prompt_key, dataset_name)
                expected_output = self.get_expected_output(row, dataset_name)
                question = self.prepare_question_from_row(row, dataset_name)
                
                prepared_data['prompts'].append(prompt)
                prepared_data['expected_outputs'].append(expected_output)
                prepared_data['sample_indices'].append(idx)
                prepared_data['row_data'].append(row.to_dict())
                prepared_data['questions'].append(question)
                
            except Exception as e:
                logger.error(f"Error preparing data for row {idx}: {e}")
                continue
        
        logger.info(f"Successfully prepared {len(prepared_data['prompts'])} samples")
        return prepared_data
    
    def get_available_datasets(self) -> Dict[str, Dict]:
        """Get information about all available datasets"""
        available = {}
        
        for dataset_name, config in self.dataset_configs.items():
            dataset_path = self.get_dataset_path(dataset_name)
            available[dataset_name] = {
                'config': config,
                'path': dataset_path,
                'exists': os.path.exists(dataset_path),
                'loaded': dataset_name in self.datasets,
                'download_link': config.get('link', 'N/A')
            }
        
        return available
    
    def list_loaded_datasets(self) -> List[str]:
        """List names of currently loaded datasets"""
        return list(self.datasets.keys())
    
    def get_compatible_prompts(self, dataset_name: str) -> List[str]:
        """Get list of prompts compatible with a dataset"""
        if dataset_name not in self.dataset_configs:
            return []
        
        config = self.dataset_configs[dataset_name]
        compatible_prompts = config.get('compatible_prompts', [])
        
        # Add general prompts
        general_prompts = [p for p in self.prompt_manager.prompts.keys() 
                          if not any(p.startswith(prefix) for prefix in ['gmeg_', 'fungi_', 'hate_', 'hardness_', 'reframing_'])]
        
        return compatible_prompts + general_prompts