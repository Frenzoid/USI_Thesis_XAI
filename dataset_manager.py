import os
import pandas as pd
import subprocess
import shutil
from typing import Dict, Optional, List
from urllib.parse import urlparse

from config import Config
from utils import setup_logging

logger = setup_logging("dataset_manager")

class DatasetManager:
    """
    Centralized dataset management with JSON configuration support.
    
    This class handles:
    1. Loading dataset configurations from JSON
    2. Downloading datasets from configured sources
    3. Loading and caching datasets in memory
    4. Validating dataset structure and fields
    5. Preparing data samples for experiments
    6. Dataset-specific field mapping and processing
    """
    
    def __init__(self):
        """Initialize dataset manager with configuration from JSON"""
        self.datasets = {}  # Cache for loaded datasets
        self.datasets_config = Config.load_datasets_config()
        
        logger.info(f"DatasetManager initialized with {len(self.datasets_config)} dataset configurations")
    
    # =============================================================================
    # PATH AND FILE MANAGEMENT
    # =============================================================================
    
    def get_dataset_path(self, dataset_name: str) -> str:
        """
        Get the full filesystem path to a dataset CSV file.
        
        Args:
            dataset_name: Name of dataset from configuration
            
        Returns:
            str: Full path to the CSV file
            
        Raises:
            ValueError: If dataset name is unknown
        """
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_config = self.datasets_config[dataset_name]
        return os.path.join(Config.DATA_DIR, dataset_config['download_path'], dataset_config['csv_file'])
    
    def is_dataset_downloaded(self, dataset_name: str) -> bool:
        """
        Check if a dataset has been downloaded and the CSV file exists.
        
        Args:
            dataset_name: Name of dataset to check
            
        Returns:
            bool: True if dataset file exists, False otherwise
        """
        dataset_path = self.get_dataset_path(dataset_name)
        return os.path.exists(dataset_path)
    
    # =============================================================================
    # DATASET DOWNLOADING
    # =============================================================================
    
    def download_dataset(self, dataset_name: str) -> bool:
        """
        Download dataset from configured URL if not already present.
        
        Handles both direct file downloads and ZIP archives that need extraction.
        Uses wget and unzip commands with Python fallbacks.
        
        Args:
            dataset_name: Name of dataset to download
            
        Returns:
            bool: True if download successful, False otherwise
        """
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
            # Check if required tools are available
            self._check_system_dependencies()
            
            # Determine file type from URL
            parsed_url = urlparse(download_url)
            url_path = parsed_url.path
            
            if url_path.endswith('.zip'):
                # Download and extract ZIP file
                zip_filename = os.path.join(download_path, f"{dataset_name}.zip")
                
                # Download with wget or urllib fallback
                self._download_file(download_url, zip_filename)
                
                # Extract with unzip or Python zipfile as fallback
                self._extract_zip(zip_filename, download_path)
                
                # Clean up zip file
                os.remove(zip_filename)
                
                logger.info(f"Successfully downloaded and extracted: {dataset_name}")
                
            else:
                # Direct file download
                output_filename = os.path.join(download_path, dataset_config['csv_file'])
                output_dir = os.path.dirname(output_filename)
                os.makedirs(output_dir, exist_ok=True)
                
                self._download_file(download_url, output_filename)
                
                logger.info(f"Successfully downloaded: {dataset_name}")
            
            # Verify download succeeded
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
    
    def _check_system_dependencies(self):
        """Check if required system tools are available"""
        try:
            subprocess.run(["wget", "--version"], capture_output=True, check=True)
            logger.debug("wget is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("wget not found, will use Python urllib as fallback")
        
        try:
            subprocess.run(["unzip", "-v"], capture_output=True, check=True)
            logger.debug("unzip is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("unzip not found, will use Python zipfile as fallback")
    
    def _download_file(self, url: str, output_path: str):
        """Download file using wget or urllib fallback"""
        try:
            # Try wget first (10 minute timeout)
            subprocess.run([
                "wget", "-q", "-O", output_path, url
            ], check=True, timeout=600)
            logger.debug(f"Downloaded using wget: {output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to urllib
            logger.info("Using urllib fallback for download")
            import urllib.request
            urllib.request.urlretrieve(url, output_path)
            logger.debug(f"Downloaded using urllib: {output_path}")
    
    def _extract_zip(self, zip_path: str, extract_path: str):
        """Extract ZIP file using unzip or zipfile fallback"""
        try:
            # Try unzip first (5 minute timeout)
            subprocess.run([
                "unzip", "-q", zip_path, "-d", extract_path
            ], check=True, timeout=300)
            logger.debug(f"Extracted using unzip: {zip_path}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to Python zipfile
            logger.info("Using zipfile fallback for extraction")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            logger.debug(f"Extracted using zipfile: {zip_path}")
    
    # =============================================================================
    # DATASET LOADING AND CACHING
    # =============================================================================
    
    def load_dataset(self, dataset_name: str, ensure_download: bool = True) -> Optional[pd.DataFrame]:
        """
        Load a dataset by name, with automatic downloading if needed.
        
        Datasets are cached in memory after loading for efficiency.
        
        Args:
            dataset_name: Name of dataset to load
            ensure_download: Whether to download dataset if not found locally
            
        Returns:
            pandas.DataFrame: Loaded dataset, or None if failed
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name not in self.datasets_config:
            logger.error(f"Unknown dataset: {dataset_name}")
            return None
        
        # Download if needed and requested
        if ensure_download and not self.is_dataset_downloaded(dataset_name):
            if not self.download_dataset(dataset_name):
                logger.error(f"Could not download dataset: {dataset_name}")
                return None
        
        # Load dataset from file
        dataset_path = self.get_dataset_path(dataset_name)
        
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded {dataset_name} dataset with {len(df)} rows and {len(df.columns)} columns")
            
            # Cache in memory for reuse
            self.datasets[dataset_name] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name} from {dataset_path}: {e}")
            return None
    
    # =============================================================================
    # DATASET INFORMATION AND ANALYSIS
    # =============================================================================
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get comprehensive information about a dataset.
        
        Includes configuration, download status, statistics, and field analysis.
        
        Args:
            dataset_name: Name of dataset to analyze
            
        Returns:
            dict: Comprehensive dataset information
        """
        if dataset_name not in self.datasets_config:
            return {"error": f"Unknown dataset: {dataset_name}"}
        
        config = self.datasets_config[dataset_name]
        dataset_path = self.get_dataset_path(dataset_name)
        is_downloaded = self.is_dataset_downloaded(dataset_name)
        is_loaded = dataset_name in self.datasets
        
        # Basic information
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
        
        # Add detailed statistics if dataset is loaded
        if is_loaded:
            df = self.datasets[dataset_name]
            info.update({
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'columns': list(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_values': df.isnull().sum().to_dict()
            })
            
            # Analyze answer field for quality assessment
            answer_field = config['answer_field']
            if answer_field in df.columns:
                answer_series = df[answer_field]
                
                # Count NA/empty answers (common patterns for unannotated data)
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
            # Try to get basic info without fully loading (for large datasets)
            try:
                df_sample = pd.read_csv(dataset_path, nrows=5)
                info.update({
                    'columns': list(df_sample.columns),
                    'sample_row': df_sample.iloc[0].to_dict() if len(df_sample) > 0 else None
                })
            except Exception as e:
                logger.warning(f"Could not read sample from {dataset_name}: {e}")
        
        return info
    
    # =============================================================================
    # DATASET VALIDATION
    # =============================================================================
    
    def validate_dataset_fields(self, dataset_name: str) -> bool:
        """
        Validate that dataset has all required fields for processing.
        
        Checks that question fields and answer field exist in the dataset.
        Provides helpful error messages for troubleshooting.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            bool: True if all required fields are present, False otherwise
        """
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
            
            # Provide dataset-specific guidance
            if dataset_name == 'gmeg':
                logger.info("Expected GMEG fields: 'original', 'revised', 'please_explain_the_revisions_write_na_if_not_annotatable'")
            
            return False
        
        logger.info(f"Dataset {dataset_name} validation passed")
        return True
    
    # =============================================================================
    # DATA SAMPLING AND PREPARATION
    # =============================================================================
    
    def get_sample_data(self, dataset_name: str, size: int, random_state: int = None) -> Optional[pd.DataFrame]:
        """
        Get a random sample from the dataset for experiments.
        
        Args:
            dataset_name: Name of dataset to sample from
            size: Number of samples to return
            random_state: Random seed for reproducibility
            
        Returns:
            pandas.DataFrame: Sampled data, or None if failed
        """
        logger.info(f"Getting sample of size {size} from dataset {dataset_name}")
        
        # Load dataset if not already loaded
        if dataset_name not in self.datasets:
            df = self.load_dataset(dataset_name)
            if df is None:
                return None
        else:
            df = self.datasets[dataset_name]
        
        # Handle case where requested sample is larger than dataset
        if size >= len(df):
            logger.info(f"Requested sample size ({size}) >= dataset size ({len(df)}), returning full dataset")
            return df.copy()
        else:
            sample_df = df.sample(n=size, random_state=random_state or Config.RANDOM_SEED).reset_index(drop=True)
            logger.info(f"Sampled {size} rows from dataset of {len(df)} total rows")
            return sample_df
    
    # =============================================================================
    # DATASET-SPECIFIC DATA PROCESSING
    # =============================================================================
    
    def prepare_question_text(self, row: pd.Series, dataset_name: str) -> str:
        """
        Prepare question text from a dataset row for prompt generation.
        
        Different datasets have different field structures, so this method
        handles dataset-specific formatting.
        
        Args:
            row: Single row from dataset
            dataset_name: Name of dataset (for format handling)
            
        Returns:
            str: Formatted question text
            
        Raises:
            ValueError: If dataset name is unknown
        """
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.datasets_config[dataset_name]
        question_fields = config['question_fields']
        
        # Dataset-specific formatting
        if dataset_name == 'gmeg':
            # GMEG has original and revised texts for comparison
            if len(question_fields) >= 2:
                original = str(row.get(question_fields[0], ''))
                revised = str(row.get(question_fields[1], ''))
                return f"Original: {original}\nRevised: {revised}"
        
        # Default formatting: concatenate all question fields
        question_parts = []
        for field in question_fields:
            if field in row and not pd.isna(row[field]):
                question_parts.append(f"{field}: {str(row[field])}")
        
        return " | ".join(question_parts)
    
    def get_expected_answer(self, row: pd.Series, dataset_name: str) -> str:
        """
        Extract expected answer from a dataset row.
        
        Args:
            row: Single row from dataset
            dataset_name: Name of dataset (for field mapping)
            
        Returns:
            str: Expected answer text, or empty string if missing
            
        Raises:
            ValueError: If dataset name is unknown
        """
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.datasets_config[dataset_name]
        answer_field = config['answer_field']
        
        if answer_field in row and not pd.isna(row[answer_field]):
            return str(row[answer_field])
        else:
            return ""
    
    # =============================================================================
    # DATASET CATALOG AND MANAGEMENT
    # =============================================================================
    
    def get_available_datasets(self) -> Dict[str, Dict]:
        """
        Get information about all configured datasets.
        
        Returns:
            dict: Information for each configured dataset
        """
        available = {}
        
        for dataset_name in self.datasets_config.keys():
            available[dataset_name] = self.get_dataset_info(dataset_name)
        
        return available
    
    def list_downloaded_datasets(self) -> List[str]:
        """
        List names of datasets that have been downloaded locally.
        
        Returns:
            list: Names of downloaded datasets
        """
        downloaded = []
        
        for dataset_name in self.datasets_config.keys():
            if self.is_dataset_downloaded(dataset_name):
                downloaded.append(dataset_name)
        
        return downloaded
    
    def list_loaded_datasets(self) -> List[str]:
        """
        List names of datasets currently loaded in memory.
        
        Returns:
            list: Names of loaded datasets
        """
        return list(self.datasets.keys())
    
    def cleanup_datasets(self):
        """
        Clear all loaded datasets from memory to free RAM.
        
        Useful when working with large datasets or running many experiments.
        """
        logger.info(f"Cleaning up {len(self.datasets)} loaded datasets")
        self.datasets.clear()
        logger.info("Dataset cleanup completed")