import os
import pandas as pd
import subprocess
import shutil
import re
from typing import Dict, Optional, List, Tuple, Any
from urllib.parse import urlparse

from config import Config
from utils import setup_logging

logger = setup_logging("dataset_manager")

class DatasetManager:
    """
    Centralized dataset management with JSON configuration support and generic field handling.
    
    This class handles:
    1. Loading dataset configurations from JSON
    2. Downloading datasets from configured sources
    3. Loading and caching datasets in memory
    4. Validating dataset structure and fields
    5. Preparing data samples for experiments with row pruning
    6. Generic field mapping without hardcoded dataset structures
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
    # ROW PRUNING LOGIC
    # =============================================================================
    
    def _compile_pattern(self, pattern: str) -> Tuple[bool, Any]:
        """
        Compile a pattern, detecting if it's regex or literal.
        
        Args:
            pattern: Pattern string, with optional "regex:" prefix
            
        Returns:
            Tuple[bool, any]: (is_regex, compiled_pattern_or_string)
            
        Raises:
            ValueError: If regex pattern is invalid
        """
        if pattern.startswith("regex:"):
            regex_pattern = pattern[6:]  # Remove "regex:" prefix
            try:
                compiled = re.compile(regex_pattern)
                return True, compiled
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{regex_pattern}': {e}")
        else:
            return False, pattern
    
    def _value_matches_pattern(self, value: str, pattern_info: Tuple[bool, Any]) -> bool:
        """
        Check if a value matches a compiled pattern.
        
        Args:
            value: String value to check
            pattern_info: Tuple from _compile_pattern()
            
        Returns:
            bool: True if value matches pattern
        """
        is_regex, pattern = pattern_info
        
        # Defensive: ensure value is a string
        str_value = str(value) if value is not None else ""
        
        try:
            if is_regex:
                return bool(pattern.search(str_value))
            else:
                return str_value == pattern
        except Exception as e:
            # This should rarely happen, but provides safety net
            logger.warning(f"Error matching pattern {pattern} against value '{str_value}': {e}")
            return False
    
    def _validate_prune_columns(self, dataset_name: str, dataset_columns: List[str]) -> None:
        """
        Validate that prune_row columns exist in the actual dataset.
        
        Args:
            dataset_name: Name of dataset
            dataset_columns: List of actual column names in the dataset
            
        Raises:
            ValueError: If unknown columns are specified in prune_row config
        """
        dataset_config = self.datasets_config[dataset_name]
        prune_config = dataset_config.get('prune_row', {})
        
        if not prune_config:
            return
        
        # Get columns specified in prune config (excluding "*")
        specified_columns = [col for col in prune_config.keys() if col != "*"]
        
        # Find columns that don't exist in the dataset
        unknown_columns = [col for col in specified_columns if col not in dataset_columns]
        
        if unknown_columns:
            logger.warning(f"Dataset '{dataset_name}' prune_row config references unknown columns: {unknown_columns}")
            logger.info(f"Available columns in dataset: {dataset_columns}")
            logger.info("Unknown columns will be ignored during pruning")
            
            # Optionally make this a hard error instead of warning:
            # raise ValueError(f"Dataset '{dataset_name}' prune_row config contains unknown columns: {unknown_columns}")
    
    def has_pruning_config(self, dataset_name: str) -> bool:
        """
        Check if a dataset has pruning configuration.
        
        Args:
            dataset_name: Name of dataset to check
            
        Returns:
            bool: True if dataset has prune_row configuration
        """
        if dataset_name not in self.datasets_config:
            return False
        
        dataset_config = self.datasets_config[dataset_name]
        prune_config = dataset_config.get('prune_row', {})
        
        # Check if prune_row exists and is not empty
        return bool(prune_config)
    
    def should_prune_row(self, row: pd.Series, dataset_name: str) -> Tuple[bool, str]:
        """
        Check if a row should be pruned based on dataset pruning configuration.
        
        Args:
            row: Dataset row to check
            dataset_name: Name of dataset (for pruning config)
            
        Returns:
            Tuple[bool, str]: (should_prune, reason)
            
        Raises:
            ValueError: If dataset unknown or regex pattern invalid
        """
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_config = self.datasets_config[dataset_name]
        prune_config = dataset_config.get('prune_row', {})
        
        if not prune_config:
            return False, ""
        
        # Compile all patterns once for this row check
        compiled_patterns = {}
        try:
            for column, patterns in prune_config.items():
                compiled_patterns[column] = [self._compile_pattern(p) for p in patterns]
        except ValueError as e:
            raise ValueError(f"Dataset '{dataset_name}' pruning config error: {e}")
        
        # Check * rules first (priority)
        if "*" in compiled_patterns:
            for pattern_info in compiled_patterns["*"]:
                # Check this pattern against ALL columns in the row
                for col_name, col_value in row.items():
                    if pd.isna(col_value):
                        col_value = ""
                    
                    if self._value_matches_pattern(str(col_value), pattern_info):
                        is_regex, pattern = pattern_info
                        pattern_str = pattern.pattern if is_regex else pattern
                        return True, f"Global pattern '{pattern_str}' matched column '{col_name}' with value '{col_value}'"
        
        # Check specific column rules
        for column, pattern_list in compiled_patterns.items():
            if column == "*":  # Already checked above
                continue
                
            if column not in row:
                logger.debug(f"Column '{column}' specified in prune config but not found in dataset row")
                continue  # Column doesn't exist in this row, skip
                
            col_value = row[column]
            if pd.isna(col_value):
                col_value = ""
            
            for pattern_info in pattern_list:
                if self._value_matches_pattern(str(col_value), pattern_info):
                    is_regex, pattern = pattern_info
                    pattern_str = pattern.pattern if is_regex else pattern
                    return True, f"Column '{column}' pattern '{pattern_str}' matched value '{col_value}'"
        
        return False, ""
    
    def filter_dataset_rows(self, df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, int, List[str]]:
        """
        Filter dataset rows based on pruning configuration.
        
        Args:
            df: Dataset DataFrame
            dataset_name: Name of dataset (for pruning config)
            
        Returns:
            Tuple[pd.DataFrame, int, List[str]]: (filtered_df, skipped_count, skip_reasons)
            
        Raises:
            ValueError: If dataset unknown or regex pattern invalid
        """
        logger.info(f"Filtering dataset '{dataset_name}' with {len(df)} total rows")
        
        # Check if pruning is configured for this dataset
        if not self.has_pruning_config(dataset_name):
            logger.info(f"No pruning configuration found for dataset '{dataset_name}', keeping all rows")
            return df.copy(), 0, []
        
        logger.info(f"Applying pruning rules for dataset '{dataset_name}'")
        
        # Validate that specified columns exist in the dataset
        self._validate_prune_columns(dataset_name, list(df.columns))
        
        filtered_rows = []
        skip_reasons = []
        skipped_count = 0
        
        for idx, row in df.iterrows():
            should_skip, reason = self.should_prune_row(row, dataset_name)
            
            if should_skip:
                skipped_count += 1
                skip_reasons.append(f"Row {idx}: {reason}")
                logger.debug(f"Skipping row {idx}: {reason}")
            else:
                filtered_rows.append(row)
        
        if filtered_rows:
            filtered_df = pd.DataFrame(filtered_rows).reset_index(drop=True)
        else:
            # Create empty DataFrame with same columns if all rows were filtered
            filtered_df = df.iloc[0:0].copy()
        
        logger.info(f"Filtered dataset: kept {len(filtered_df)} rows, skipped {skipped_count} rows")
        
        if skipped_count > 0:
            # Log first few reasons for debugging
            sample_reasons = skip_reasons[:5]
            logger.info(f"Sample skip reasons: {sample_reasons}")
            if len(skip_reasons) > 5:
                logger.info(f"... and {len(skip_reasons) - 5} more")
        
        return filtered_df, skipped_count, skip_reasons
    
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
            'is_loaded': is_loaded,
            'prune_config': config.get('prune_row', {})
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
                
                info['answer_field_stats'] = {
                    'total_answers': len(answer_series),
                    'null_answers': answer_series.isnull().sum(),
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
            logger.info(f"Expected question fields: {config['question_fields']}")
            logger.info(f"Expected answer field: {config['answer_field']}")
            return False
        
        logger.info(f"Dataset {dataset_name} validation passed")
        return True
    
    # =============================================================================
    # GENERIC DATA PROCESSING - REMOVED HARDCODED LOGIC
    # =============================================================================
    
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