import os
import pandas as pd
import subprocess
from typing import Dict, Optional, List
from urllib.parse import urlparse

from config import Config
from utils import setup_logging

logger = setup_logging("dataset_manager")

class DatasetManager:
    """Centralized dataset management with JSON configuration support"""
    
    def __init__(self):
        self.datasets = {}
        self.datasets_config = Config.load_datasets_config()
        
        logger.info(f"DatasetManager initialized with {len(self.datasets_config)} dataset configurations")
    
    def get_dataset_path(self, dataset_name: str) -> str:
        """Get the full path to a dataset CSV file"""
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_config = self.datasets_config[dataset_name]
        return os.path.join(Config.DATA_DIR, dataset_config['download_path'], dataset_config['csv_file'])
    
    def is_dataset_downloaded(self, dataset_name: str) -> bool:
        """Check if dataset is already downloaded"""
        dataset_path = self.get_dataset_path(dataset_name)
        return os.path.exists(dataset_path)
    
    def download_dataset(self, dataset_name: str) -> bool:
        """Download dataset if not already present"""
        if dataset_name not in self.datasets_config:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        if self.is_dataset_downloaded(dataset_name):
            logger.info(f"Dataset {dataset_name} already downloaded")
            return True
        
        dataset_config = self.datasets_config[dataset_name]
        download_url = dataset_config['download_link']
        download_path = os.path.join(Config.DATA_DIR, dataset_config['download_path'])
        
        logger.info(f"Downloading dataset {dataset_name} from {download_url}")
        
        # Create download directory
        os.makedirs(download_path, exist_ok=True)
        
        try:
            # Determine file extension from URL
            parsed_url = urlparse(download_url)
            url_path = parsed_url.path
            
            if url_path.endswith('.zip'):
                # Download and extract ZIP file
                zip_filename = os.path.join(download_path, f"{dataset_name}.zip")
                
                # Download
                subprocess.run([
                    "wget", "-q", "-O", zip_filename, download_url
                ], check=True, timeout=600)  # 10 minute timeout
                
                # Extract
                subprocess.run([
                    "unzip", "-q", zip_filename, "-d", download_path
                ], check=True, timeout=300)  # 5 minute timeout
                
                # Clean up zip file
                os.remove(zip_filename)
                
                logger.info(f"Successfully downloaded and extracted: {dataset_name}")
                
            else:
                # Direct file download
                output_filename = os.path.join(download_path, dataset_config['csv_file'])
                output_dir = os.path.dirname(output_filename)
                os.makedirs(output_dir, exist_ok=True)
                
                subprocess.run([
                    "wget", "-q", "-O", output_filename, download_url
                ], check=True, timeout=600)
                
                logger.info(f"Successfully downloaded: {dataset_name}")
            
            # Verify download
            if self.is_dataset_downloaded(dataset_name):
                return True
            else:
                logger.error(f"Download completed but dataset file not found: {dataset_name}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Download timeout for dataset: {dataset_name}")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed for dataset {dataset_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading dataset {dataset_name}: {e}")
            return False
    
    def load_dataset(self, dataset_name: str, ensure_download: bool = True) -> Optional[pd.DataFrame]:
        """Load a dataset by name"""
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name not in self.datasets_config:
            logger.error(f"Unknown dataset: {dataset_name}")
            return None
        
        # Download if needed
        if ensure_download and not self.is_dataset_downloaded(dataset_name):
            if not self.download_dataset(dataset_name):
                logger.error(f"Could not download dataset: {dataset_name}")
                return None
        
        # Load dataset
        dataset_path = self.get_dataset_path(dataset_name)
        
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded {dataset_name} dataset with {len(df)} rows and {len(df.columns)} columns")
            
            # Store in cache
            self.datasets[dataset_name] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name} from {dataset_path}: {e}")
            return None
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a dataset"""
        if dataset_name not in self.datasets_config:
            return {"error": f"Unknown dataset: {dataset_name}"}
        
        config = self.datasets_config[dataset_name]
        dataset_path = self.get_dataset_path(dataset_name)
        is_downloaded = self.is_dataset_downloaded(dataset_name)
        is_loaded = dataset_name in self.datasets
        
        info = {
            'name': dataset_name,
            'description': config['description'],
            'download_link': config['download_link'],
            'download_path': config['download_path'],
            'csv_file': config['csv_file'],
            'full_path': dataset_path,
            'question_fields': config['question_fields'],
            'answer_field': config['answer_field'],
            'is_downloaded': is_downloaded,
            'is_loaded': is_loaded
        }
        
        # Add dataset statistics if loaded
        if is_loaded:
            df = self.datasets[dataset_name]
            info.update({
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'columns': list(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_values': df.isnull().sum().to_dict()
            })
            
            # Analyze answer field
            answer_field = config['answer_field']
            if answer_field in df.columns:
                answer_series = df[answer_field]
                
                # Count NA/empty answers
                na_patterns = ['na', 'n/a', 'not applicable', 'not annotatable', '', 'none']
                na_mask = answer_series.fillna('').astype(str).str.lower().str.strip().isin(na_patterns)
                na_count = na_mask.sum()
                
                info['answer_field_stats'] = {
                    'total_answers': len(answer_series),
                    'valid_answers': len(answer_series) - na_count,
                    'na_answers': int(na_count),
                    'avg_answer_length': answer_series.fillna('').astype(str).str.len().mean(),
                    'max_answer_length': answer_series.fillna('').astype(str).str.len().max()
                }
        elif is_downloaded:
            # Try to get basic info without fully loading
            try:
                df_sample = pd.read_csv(dataset_path, nrows=5)
                info.update({
                    'columns': list(df_sample.columns),
                    'sample_row': df_sample.iloc[0].to_dict() if len(df_sample) > 0 else None
                })
            except Exception as e:
                logger.warning(f"Could not read sample from {dataset_name}: {e}")
        
        return info
    
    def validate_dataset_fields(self, dataset_name: str) -> bool:
        """Validate that dataset has required fields"""
        if dataset_name not in self.datasets_config:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        # Load dataset if not already loaded
        if dataset_name not in self.datasets:
            df = self.load_dataset(dataset_name)
            if df is None:
                return False
        else:
            df = self.datasets[dataset_name]
        
        config = self.datasets_config[dataset_name]
        required_fields = config['question_fields'] + [config['answer_field']]
        
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            logger.error(f"Dataset {dataset_name} missing required fields: {missing_fields}")
            logger.info(f"Available columns: {list(df.columns)}")
            
            # For GMEG, give specific guidance
            if dataset_name == 'gmeg':
                logger.info("Expected GMEG fields: 'original', 'revised', 'please_explain_the_revisions_write_na_if_not_annotatable'")
            
            return False
        
        logger.info(f"Dataset {dataset_name} validation passed")
        return True
    
    def get_sample_data(self, dataset_name: str, size: int, random_state: int = None) -> Optional[pd.DataFrame]:
        """Get a sample from the dataset"""
        logger.info(f"Getting sample of size {size} from dataset {dataset_name}")
        
        # Load dataset if not already loaded
        if dataset_name not in self.datasets:
            df = self.load_dataset(dataset_name)
            if df is None:
                return None
        else:
            df = self.datasets[dataset_name]
        
        # Sample data
        if size >= len(df):
            logger.info(f"Requested sample size ({size}) >= dataset size ({len(df)}), returning full dataset")
            return df.copy()
        else:
            sample_df = df.sample(n=size, random_state=random_state or Config.RANDOM_SEED).reset_index(drop=True)
            logger.info(f"Sampled {size} rows from dataset of {len(df)} total rows")
            return sample_df
    
    def prepare_question_text(self, row: pd.Series, dataset_name: str) -> str:
        """Prepare question text from a dataset row"""
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.datasets_config[dataset_name]
        question_fields = config['question_fields']
        
        # For GMEG dataset, concatenate original and revised texts
        if dataset_name == 'gmeg':
            if len(question_fields) >= 2:
                original = str(row.get(question_fields[0], ''))
                revised = str(row.get(question_fields[1], ''))
                return f"Original: {original}\nRevised: {revised}"
        
        # Default: concatenate all question fields
        question_parts = []
        for field in question_fields:
            if field in row and not pd.isna(row[field]):
                question_parts.append(f"{field}: {str(row[field])}")
        
        return " | ".join(question_parts)
    
    def get_expected_answer(self, row: pd.Series, dataset_name: str) -> str:
        """Get expected answer from a dataset row"""
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.datasets_config[dataset_name]
        answer_field = config['answer_field']
        
        if answer_field in row and not pd.isna(row[answer_field]):
            return str(row[answer_field])
        else:
            return ""
    
    def get_available_datasets(self) -> Dict[str, Dict]:
        """Get information about all configured datasets"""
        available = {}
        
        for dataset_name in self.datasets_config.keys():
            available[dataset_name] = self.get_dataset_info(dataset_name)
        
        return available
    
    def list_downloaded_datasets(self) -> List[str]:
        """List names of downloaded datasets"""
        downloaded = []
        
        for dataset_name in self.datasets_config.keys():
            if self.is_dataset_downloaded(dataset_name):
                downloaded.append(dataset_name)
        
        return downloaded
    
    def list_loaded_datasets(self) -> List[str]:
        """List names of currently loaded datasets"""
        return list(self.datasets.keys())
    
    def cleanup_datasets(self):
        """Clear loaded datasets from memory"""
        logger.info(f"Cleaning up {len(self.datasets)} loaded datasets")
        self.datasets.clear()
        logger.info("Dataset cleanup completed")