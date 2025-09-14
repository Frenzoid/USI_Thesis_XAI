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
    1. Loading setup configurations from JSON
    2. Downloading datasets from configured sources (CSV and Parquet)
    3. Loading and caching datasets in memory
    4. Validating dataset structure and fields
    5. Preparing data samples for experiments with row pruning
    6. Generic field mapping without hardcoded dataset structures
    7. Support for both ZIP archives and direct file downloads
    """
    
    def __init__(self):
        """Initialize dataset manager with configuration from JSON"""
        self.datasets = {}  # Cache for loaded datasets
        self.setups_config = Config.load_setups_config()
        
        logger.info(f"DatasetManager initialized with {len(self.setups_config)} setup configurations")
    
    # =============================================================================
    # PATH AND FILE MANAGEMENT WITH PARQUET SUPPORT
    # =============================================================================
    
    def get_dataset_path(self, setup_name: str) -> str:
        """
        Get the full filesystem path to a dataset file (CSV or Parquet).
        
        Args:
            setup_name: Name of setup from configuration
            
        Returns:
            str: Full path to the dataset file
            
        Raises:
            ValueError: If setup name is unknown or file configuration is invalid
        """
        if setup_name not in self.setups_config:
            raise ValueError(f"Unknown setup: {setup_name}")
        
        setup_config = self.setups_config[setup_name]
        dataset_config = setup_config.get('dataset', {})
        
        if not dataset_config:
            raise ValueError(f"Setup '{setup_name}' missing 'dataset' configuration")
        
        # Check for either csv_file or parquet_file
        if 'csv_file' in dataset_config:
            file_path = dataset_config['csv_file']
        elif 'parquet_file' in dataset_config:
            file_path = dataset_config['parquet_file']
        else:
            raise ValueError(f"Setup '{setup_name}' dataset config must specify either 'csv_file' or 'parquet_file'")
        
        return os.path.join(Config.DATA_DIR, dataset_config['download_path'], file_path)
    
    def get_dataset_file_type(self, setup_name: str) -> str:
        """
        Determine the file type of a dataset (csv or parquet).
        
        Args:
            setup_name: Name of setup from configuration
            
        Returns:
            str: File type ('csv' or 'parquet')
            
        Raises:
            ValueError: If setup name is unknown or file type cannot be determined
        """
        if setup_name not in self.setups_config:
            raise ValueError(f"Unknown setup: {setup_name}")
        
        setup_config = self.setups_config[setup_name]
        dataset_config = setup_config.get('dataset', {})
        
        if not dataset_config:
            raise ValueError(f"Setup '{setup_name}' missing 'dataset' configuration")
        
        if 'csv_file' in dataset_config:
            return 'csv'
        elif 'parquet_file' in dataset_config:
            return 'parquet'
        else:
            raise ValueError(f"Setup '{setup_name}' dataset config must specify either 'csv_file' or 'parquet_file'")
    
    def is_dataset_downloaded(self, setup_name: str) -> bool:
        """
        Check if a dataset has been downloaded and the file exists.
        
        Args:
            setup_name: Name of setup to check
            
        Returns:
            bool: True if dataset file exists, False otherwise
        """
        try:
            dataset_path = self.get_dataset_path(setup_name)
            return os.path.exists(dataset_path)
        except ValueError:
            return False
    
    # =============================================================================
    # ROW PRUNING LOGIC (unchanged)
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
    
    def _validate_prune_columns(self, setup_name: str, dataset_columns: List[str]) -> None:
        """
        Validate that prune_row columns exist in the actual dataset.
        
        Args:
            setup_name: Name of setup
            dataset_columns: List of actual column names in the dataset
            
        Raises:
            ValueError: If unknown columns are specified in prune_row config
        """
        setup_config = self.setups_config[setup_name]
        prune_config = setup_config.get('prune_row', {})
        
        if not prune_config:
            return
        
        # Get columns specified in prune config (excluding "*")
        specified_columns = [col for col in prune_config.keys() if col != "*"]
        
        # Find columns that don't exist in the dataset
        unknown_columns = [col for col in specified_columns if col not in dataset_columns]
        
        if unknown_columns:
            logger.warning(f"Setup '{setup_name}' prune_row config references unknown columns: {unknown_columns}")
            logger.info(f"Available columns in dataset: {dataset_columns}")
            logger.info("Unknown columns will be ignored during pruning")
            
            # Optionally make this a hard error instead of warning:
            # raise ValueError(f"Setup '{setup_name}' prune_row config contains unknown columns: {unknown_columns}")
    
    def has_pruning_config(self, setup_name: str) -> bool:
        """
        Check if a setup has pruning configuration.
        
        Args:
            setup_name: Name of setup to check
            
        Returns:
            bool: True if setup has prune_row configuration
        """
        if setup_name not in self.setups_config:
            return False
        
        setup_config = self.setups_config[setup_name]
        prune_config = setup_config.get('prune_row', {})
        
        # Check if prune_row exists and is not empty
        return bool(prune_config)
    
    def should_prune_row(self, row: pd.Series, setup_name: str) -> Tuple[bool, str]:
        """
        Check if a row should be pruned based on setup pruning configuration.
        
        Args:
            row: Dataset row to check
            setup_name: Name of setup (for pruning config)
            
        Returns:
            Tuple[bool, str]: (should_prune, reason)
            
        Raises:
            ValueError: If setup unknown or regex pattern invalid
        """
        if setup_name not in self.setups_config:
            raise ValueError(f"Unknown setup: {setup_name}")
        
        setup_config = self.setups_config[setup_name]
        prune_config = setup_config.get('prune_row', {})
        
        if not prune_config:
            return False, ""
        
        # Compile all patterns once for this row check
        compiled_patterns = {}
        try:
            for column, patterns in prune_config.items():
                compiled_patterns[column] = [self._compile_pattern(p) for p in patterns]
        except ValueError as e:
            raise ValueError(f"Setup '{setup_name}' pruning config error: {e}")
        
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
    
    def filter_dataset_rows(self, df: pd.DataFrame, setup_name: str) -> Tuple[pd.DataFrame, int, List[str]]:
        """
        Filter dataset rows based on pruning configuration.
        
        Args:
            df: Dataset DataFrame
            setup_name: Name of setup (for pruning config)
            
        Returns:
            Tuple[pd.DataFrame, int, List[str]]: (filtered_df, skipped_count, skip_reasons)
            
        Raises:
            ValueError: If setup unknown or regex pattern invalid
        """
        logger.info(f"Filtering dataset for setup '{setup_name}' with {len(df)} total rows")
        
        # Check if pruning is configured for this setup
        if not self.has_pruning_config(setup_name):
            logger.info(f"No pruning configuration found for setup '{setup_name}', keeping all rows")
            return df.copy(), 0, []
        
        logger.info(f"Applying pruning rules for setup '{setup_name}'")
        
        # Validate that specified columns exist in the dataset
        self._validate_prune_columns(setup_name, list(df.columns))
        
        filtered_rows = []
        skip_reasons = []
        skipped_count = 0
        
        for idx, row in df.iterrows():
            should_skip, reason = self.should_prune_row(row, setup_name)
            
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
    # DATASET DOWNLOADING WITH ZIP AND DIRECT FILE SUPPORT
    # =============================================================================
    
    def _is_zip_file(self, url: str, filename: str = None) -> bool:
        """
        Determine if a file is a ZIP archive based on URL or filename.
        
        Args:
            url: Download URL
            filename: Optional filename to check
            
        Returns:
            bool: True if file appears to be a ZIP archive
        """
        # Check URL path
        parsed_url = urlparse(url)
        url_path = parsed_url.path.lower()
        
        if url_path.endswith('.zip'):
            return True
        
        # Check filename if provided
        if filename and filename.lower().endswith('.zip'):
            return True
        
        # Check for common ZIP download patterns
        if 'archive' in url_path and ('zip' in url_path or 'main.zip' in url_path):
            return True
        
        return False
    
    def _get_target_filename(self, url: str, setup_name: str, dataset_config: Dict[str, Any]) -> str:
        """
        Determine the target filename for download.
        
        Args:
            url: Download URL
            setup_name: Name of setup
            dataset_config: Dataset configuration
            
        Returns:
            str: Target filename for download
        """
        # For ZIP files, use setup name
        if self._is_zip_file(url):
            return f"{setup_name}.zip"
        
        # For direct files, try to get filename from config or URL
        if 'parquet_file' in dataset_config:
            return dataset_config['parquet_file']
        elif 'csv_file' in dataset_config:
            # If CSV is specified but we're doing direct download, use the filename part
            return os.path.basename(dataset_config['csv_file'])
        
        # Fallback: extract from URL
        parsed_url = urlparse(url)
        url_filename = os.path.basename(parsed_url.path)
        
        # Handle query parameters (like ?download=true)
        if '?' in url_filename:
            url_filename = url_filename.split('?')[0]
        
        return url_filename if url_filename else f"{setup_name}_data"
    
    def download_dataset(self, setup_name: str) -> bool:
        """
        Download dataset from configured URL with support for both ZIP and direct downloads.
        
        Handles both direct file downloads and ZIP archives that need extraction.
        Uses wget and unzip commands with Python fallbacks.
        
        Args:
            setup_name: Name of setup to download
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if setup_name not in self.setups_config:
            logger.error(f"Unknown setup: {setup_name}")
            return False
        
        if self.is_dataset_downloaded(setup_name):
            logger.info(f"Dataset for setup {setup_name} already downloaded")
            return True
        
        setup_config = self.setups_config[setup_name]
        dataset_config = setup_config.get('dataset', {})
        
        if not dataset_config:
            logger.error(f"Setup '{setup_name}' missing 'dataset' configuration")
            return False
        
        download_url = dataset_config.get('download_link')
        if not download_url:
            logger.error(f"Setup '{setup_name}' missing download_link in dataset config")
            return False
        
        download_path = os.path.join(Config.DATA_DIR, dataset_config['download_path'])
        
        logger.info(f"Downloading dataset for setup {setup_name} from {download_url}")
        
        # Create download directory
        os.makedirs(download_path, exist_ok=True)
        
        try:
            # Check if required tools are available
            self._check_system_dependencies()
            
            # Determine target filename and whether it's a ZIP
            target_filename = self._get_target_filename(download_url, setup_name, dataset_config)
            is_zip = self._is_zip_file(download_url, target_filename)
            
            if is_zip:
                # Download and extract ZIP file
                zip_filepath = os.path.join(download_path, target_filename)
                logger.info(f"Downloading ZIP archive: {target_filename}")
                
                # Download with wget or urllib fallback
                self._download_file(download_url, zip_filepath)
                
                # Extract with unzip or Python zipfile as fallback
                self._extract_zip(zip_filepath, download_path)
                
                # Clean up zip file
                os.remove(zip_filepath)
                logger.info(f"Successfully downloaded and extracted dataset for setup: {setup_name}")
                
            else:
                # Direct file download
                output_filepath = os.path.join(download_path, target_filename)
                output_dir = os.path.dirname(output_filepath)
                os.makedirs(output_dir, exist_ok=True)
                
                logger.info(f"Downloading file directly: {target_filename}")
                self._download_file(download_url, output_filepath)
                logger.info(f"Successfully downloaded dataset for setup: {setup_name}")
            
            # Verify download succeeded
            if self.is_dataset_downloaded(setup_name):
                return True
            else:
                logger.error(f"Download completed but dataset file not found for setup: {setup_name}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Download timeout for setup: {setup_name}")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed for setup {setup_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading dataset for setup {setup_name}: {e}")
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
    # DATASET LOADING WITH CSV AND PARQUET SUPPORT
    # =============================================================================
    
    def load_dataset(self, setup_name: str, ensure_download: bool = True) -> Optional[pd.DataFrame]:
        """
        Load a dataset by setup name, with automatic downloading if needed.
        Supports both CSV and Parquet file formats.
        
        Datasets are cached in memory after loading for efficiency.
        
        Args:
            setup_name: Name of setup to load
            ensure_download: Whether to download dataset if not found locally
            
        Returns:
            pandas.DataFrame: Loaded dataset, or None if failed
        """
        logger.info(f"Loading dataset for setup: {setup_name}")
        
        if setup_name not in self.setups_config:
            logger.error(f"Unknown setup: {setup_name}")
            return None
        
        # Download if needed and requested
        if ensure_download and not self.is_dataset_downloaded(setup_name):
            if not self.download_dataset(setup_name):
                logger.error(f"Could not download dataset for setup: {setup_name}")
                return None
        
        # Get dataset path and file type
        dataset_path = self.get_dataset_path(setup_name)
        file_type = self.get_dataset_file_type(setup_name)
        
        try:
            # Load dataset based on file type
            if file_type == 'csv':
                df = pd.read_csv(dataset_path)
                logger.info(f"Loaded CSV dataset for setup {setup_name} with {len(df)} rows and {len(df.columns)} columns")
            elif file_type == 'parquet':
                df = pd.read_parquet(dataset_path)
                logger.info(f"Loaded Parquet dataset for setup {setup_name} with {len(df)} rows and {len(df.columns)} columns")
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Cache in memory for reuse
            self.datasets[setup_name] = df
            
            return df
            
        except ImportError as e:
            if 'parquet' in str(e).lower():
                logger.error(f"Parquet support not available. Install pyarrow: pip install pyarrow")
                logger.error(f"Error: {e}")
            else:
                logger.error(f"Import error loading dataset for setup {setup_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading dataset for setup {setup_name} from {dataset_path}: {e}")
            return None
    
    # =============================================================================
    # DATASET INFORMATION AND ANALYSIS
    # =============================================================================
    
    def get_dataset_info(self, setup_name: str) -> Dict:
        """
        Get comprehensive information about a dataset.
        
        Includes configuration, download status, statistics, and field analysis.
        
        Args:
            setup_name: Name of setup to analyze
            
        Returns:
            dict: Comprehensive dataset information
        """
        if setup_name not in self.setups_config:
            return {"error": f"Unknown setup: {setup_name}"}
        
        config = self.setups_config[setup_name]
        dataset_config = config.get('dataset', {})
        prompt_fields_config = config.get('prompt_fields', {})
        
        is_downloaded = self.is_dataset_downloaded(setup_name)
        is_loaded = setup_name in self.datasets
        
        try:
            dataset_path = self.get_dataset_path(setup_name)
            file_type = self.get_dataset_file_type(setup_name)
        except ValueError as e:
            return {"error": str(e)}
        
        # Basic information
        info = {
            'name': setup_name,
            'description': config.get('description', 'No description'),
            'download_link': dataset_config.get('download_link', ''),
            'download_path': dataset_config.get('download_path', ''),
            'file_type': file_type,
            'file_path': dataset_config.get('csv_file') or dataset_config.get('parquet_file'),
            'full_path': dataset_path,
            'question_fields': prompt_fields_config.get('question_fields', []),
            'answer_field': prompt_fields_config.get('answer_field', ''),
            'is_downloaded': is_downloaded,
            'is_loaded': is_loaded,
            'prune_config': config.get('prune_row', {})
        }
        
        # Add detailed statistics if dataset is loaded
        if is_loaded:
            df = self.datasets[setup_name]
            info.update({
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'columns': list(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_values': df.isnull().sum().to_dict()
            })
            
            # Analyze answer field for quality assessment
            answer_field = prompt_fields_config.get('answer_field', '')
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
                if file_type == 'csv':
                    df_sample = pd.read_csv(dataset_path, nrows=5)
                elif file_type == 'parquet':
                    # For parquet, read with limit - note: nrows doesn't work the same way
                    df_full = pd.read_parquet(dataset_path)
                    df_sample = df_full.head(5)
                else:
                    df_sample = None
                
                if df_sample is not None:
                    info.update({
                        'columns': list(df_sample.columns),
                        'sample_row': df_sample.iloc[0].to_dict() if len(df_sample) > 0 else None
                    })
            except Exception as e:
                logger.warning(f"Could not read sample from setup {setup_name}: {e}")
        
        return info
    
    # =============================================================================
    # DATASET VALIDATION
    # =============================================================================
    
    def validate_dataset_fields(self, setup_name: str) -> bool:
        """
        Validate that dataset has all required fields for processing.
        
        Checks that question fields and answer field exist in the dataset.
        Provides helpful error messages for troubleshooting.
        
        Args:
            setup_name: Name of setup to validate
            
        Returns:
            bool: True if all required fields are present, False otherwise
        """
        if setup_name not in self.setups_config:
            logger.error(f"Unknown setup: {setup_name}")
            return False
        
        # Load dataset if not already loaded
        if setup_name not in self.datasets:
            df = self.load_dataset(setup_name)
            if df is None:
                return False
        else:
            df = self.datasets[setup_name]
        
        config = self.setups_config[setup_name]
        
        if 'prompt_fields' not in config:
            logger.error(f"Setup '{setup_name}' missing required 'prompt_fields' configuration")
            return False
        
        prompt_fields_config = config['prompt_fields']
        
        question_fields = prompt_fields_config['question_fields']
        answer_field = prompt_fields_config['answer_field']
        
        required_fields = question_fields + ([answer_field] if answer_field else [])
        
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            logger.error(f"Dataset for setup {setup_name} missing required fields: {missing_fields}")
            logger.info(f"Available columns: {list(df.columns)}")
            logger.info(f"Expected question fields: {question_fields}")
            logger.info(f"Expected answer field: {answer_field}")
            return False
        
        logger.info(f"Dataset for setup {setup_name} validation passed")
        return True
    
    # =============================================================================
    # GENERIC DATA PROCESSING - REMOVED HARDCODED LOGIC
    # =============================================================================
    
    def get_expected_answer(self, row: pd.Series, setup_name: str) -> str:
        """
        Extract expected answer from a dataset row.
        
        Args:
            row: Single row from dataset
            setup_name: Name of setup (for field mapping)
            
        Returns:
            str: Expected answer text, or empty string if missing
            
        Raises:
            ValueError: If setup name is unknown
        """
        if setup_name not in self.setups_config:
            raise ValueError(f"Unknown setup: {setup_name}")
        
        config = self.setups_config[setup_name]
        prompt_fields_config = config.get('prompt_fields', {})
        answer_field = prompt_fields_config.get('answer_field', '')
        
        if answer_field and answer_field in row and not pd.isna(row[answer_field]):
            return str(row[answer_field])
        else:
            return ""
    
    # =============================================================================
    # DATASET CATALOG AND MANAGEMENT
    # =============================================================================
    
    def get_available_setups(self) -> Dict[str, Dict]:
        """
        Get information about all configured setups.
        
        Returns:
            dict: Information for each configured setup
        """
        available = {}
        
        for setup_name in self.setups_config.keys():
            available[setup_name] = self.get_dataset_info(setup_name)
        
        return available
    
    def list_downloaded_datasets(self) -> List[str]:
        """
        List names of setups that have their datasets downloaded locally.
        
        Returns:
            list: Names of setups with downloaded datasets
        """
        downloaded = []
        
        for setup_name in self.setups_config.keys():
            if self.is_dataset_downloaded(setup_name):
                downloaded.append(setup_name)
        
        return downloaded