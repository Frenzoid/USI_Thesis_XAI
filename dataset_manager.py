import os
import pandas as pd
import subprocess
import shutil
import re
import json
from typing import Dict, Optional, List, Tuple, Any, Union
from urllib.parse import urlparse

from config import Config
from utils import setup_logging

logger = setup_logging("dataset_manager")

class DatasetManager:
    """
    Centralized dataset management with JSON configuration support and generic field handling.
    
    This class handles:
    1. Loading setup configurations from JSON
    2. Downloading datasets from configured sources (CSV, Parquet, JSON, JSONL)
    3. Loading and caching datasets in memory
    4. Validating dataset structure and fields with nested path support
    5. Preparing data samples for experiments with row pruning
    6. Generic field mapping without hardcoded dataset structures
    7. Support for both ZIP archives and direct file downloads
    8. Nested JSON field path resolution (e.g., "context.questions[0]")
    """
    
    def __init__(self):
        """Initialize dataset manager with configuration from JSON"""
        self.datasets = {}  # Cache for loaded datasets
        self.setups_config = Config.load_setups_config()
        
        logger.info(f"DatasetManager initialized with {len(self.setups_config)} setup configurations")
    
    # =============================================================================
    # NESTED FIELD PATH RESOLUTION
    # =============================================================================
    
    def resolve_field_path(self, data: Union[Dict, List, Any], path: str) -> Any:
        """
        Resolve a nested JSON path like 'context.questions[0].text' against data structure.
        
        Args:
            data: The data structure to navigate (dict, list, or primitive)
            path: The field path to resolve (e.g., "context.questions[0]", "user.profile.name")
            
        Returns:
            The value at the specified path, or None if path doesn't exist
            
        Examples:
            resolve_field_path({"context": {"questions": ["Q1", "Q2"]}}, "context.questions[0]") -> "Q1"
            resolve_field_path({"user": {"profile": {"name": "John"}}}, "user.profile.name") -> "John"
        """
        if not path or path == "":
            return data
            
        try:
            # Parse the path into components
            parts = self._parse_field_path(path)
            
            # Navigate through the data structure
            current = data
            for part in parts:
                if isinstance(part, int):
                    # Array index
                    if not isinstance(current, (list, tuple)):
                        logger.debug(f"Expected array at path component but got {type(current)}")
                        return None
                    if part < 0 or part >= len(current):
                        logger.debug(f"Array index {part} out of bounds (length: {len(current)})")
                        return None
                    current = current[part]
                else:
                    # Object property
                    if not isinstance(current, dict):
                        logger.debug(f"Expected object at path component '{part}' but got {type(current)}")
                        return None
                    if part not in current:
                        logger.debug(f"Property '{part}' not found in object")
                        return None
                    current = current[part]
            
            return current
            
        except Exception as e:
            logger.debug(f"Error resolving field path '{path}': {e}")
            return None
    
    def _parse_field_path(self, path: str) -> List[Union[str, int]]:
        """
        Parse a field path into components, handling both object properties and array indices.
        
        Args:
            path: Field path like "context.questions[0].text"
            
        Returns:
            List of path components (strings for properties, ints for array indices)
            
        Examples:
            "context.questions[0]" -> ["context", "questions", 0]
            "user.profile.name" -> ["user", "profile", "name"]
            "items[2].details[1].value" -> ["items", 2, "details", 1, "value"]
        """
        parts = []
        current_part = ""
        i = 0
        
        while i < len(path):
            char = path[i]
            
            if char == '.':
                # End of property name
                if current_part:
                    parts.append(current_part)
                    current_part = ""
            elif char == '[':
                # Start of array index
                if current_part:
                    parts.append(current_part)
                    current_part = ""
                
                # Find matching closing bracket
                j = i + 1
                bracket_count = 1
                index_str = ""
                
                while j < len(path) and bracket_count > 0:
                    if path[j] == '[':
                        bracket_count += 1
                    elif path[j] == ']':
                        bracket_count -= 1
                    
                    if bracket_count > 0:
                        index_str += path[j]
                    j += 1
                
                if bracket_count > 0:
                    raise ValueError(f"Unclosed bracket in path: {path}")
                
                # Convert index to integer
                try:
                    index = int(index_str)
                    parts.append(index)
                except ValueError:
                    raise ValueError(f"Invalid array index '{index_str}' in path: {path}")
                
                i = j - 1  # j-1 because we'll increment i at the end of loop
            else:
                # Regular character in property name
                current_part += char
            
            i += 1
        
        # Add final part if exists
        if current_part:
            parts.append(current_part)
        
        return parts
    
    def extract_field_values(self, row: Union[Dict, pd.Series], field_paths: List[str]) -> List[str]:
        """
        Extract values from multiple field paths in a data row.
        
        Args:
            row: Data row (dict for JSON data, pd.Series for CSV/Parquet data)
            field_paths: List of field paths to extract
            
        Returns:
            List of string values extracted from the specified paths
        """
        values = []
        
        # Convert pandas Series to dict if needed
        if isinstance(row, pd.Series):
            row_data = row.to_dict()
        else:
            row_data = row
        
        for path in field_paths:
            try:
                value = self.resolve_field_path(row_data, path)
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    values.append("")
                else:
                    values.append(str(value))
            except Exception as e:
                logger.warning(f"Error extracting field path '{path}': {e}")
                values.append("")
        
        return values
    
    # =============================================================================
    # PATH AND FILE MANAGEMENT WITH JSON/JSONL SUPPORT
    # =============================================================================
    
    def get_dataset_path(self, setup_name: str) -> str:
        """
        Get the full filesystem path to a dataset file (CSV, Parquet, JSON, or JSONL).
        
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
        
        # Check for supported file types in priority order
        file_type_keys = ['csv_file', 'parquet_file', 'json_file', 'jsonl_file']
        
        for key in file_type_keys:
            if key in dataset_config:
                file_path = dataset_config[key]
                return os.path.join(Config.DATA_DIR, dataset_config['download_path'], file_path)
        
        raise ValueError(f"Setup '{setup_name}' dataset config must specify one of: {file_type_keys}")
    
    def get_dataset_file_type(self, setup_name: str) -> str:
        """
        Determine the file type of a dataset (csv, parquet, json, or jsonl).
        
        Args:
            setup_name: Name of setup from configuration
            
        Returns:
            str: File type ('csv', 'parquet', 'json', or 'jsonl')
            
        Raises:
            ValueError: If setup name is unknown or file type cannot be determined
        """
        if setup_name not in self.setups_config:
            raise ValueError(f"Unknown setup: {setup_name}")
        
        setup_config = self.setups_config[setup_name]
        dataset_config = setup_config.get('dataset', {})
        
        if not dataset_config:
            raise ValueError(f"Setup '{setup_name}' missing 'dataset' configuration")
        
        # Check file types in order
        if 'csv_file' in dataset_config:
            return 'csv'
        elif 'parquet_file' in dataset_config:
            return 'parquet'
        elif 'json_file' in dataset_config:
            return 'json'
        elif 'jsonl_file' in dataset_config:
            return 'jsonl'
        else:
            raise ValueError(f"Setup '{setup_name}' dataset config must specify a supported file type")
    
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
    # ROW PRUNING LOGIC (updated for JSON support)
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
    
    def should_prune_row(self, row: Union[Dict, pd.Series], setup_name: str) -> Tuple[bool, str]:
        """
        Check if a row should be pruned based on setup pruning configuration.
        Updated to support JSON field paths.
        
        Args:
            row: Dataset row to check (dict for JSON, pd.Series for CSV/Parquet)
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
        
        # Convert pandas Series to dict if needed
        if isinstance(row, pd.Series):
            row_data = row.to_dict()
        else:
            row_data = row
        
        # Compile all patterns once for this row check
        compiled_patterns = {}
        try:
            for field_path, patterns in prune_config.items():
                compiled_patterns[field_path] = [self._compile_pattern(p) for p in patterns]
        except ValueError as e:
            raise ValueError(f"Setup '{setup_name}' pruning config error: {e}")
        
        # Check * rules first (priority) - applies to all field paths
        if "*" in compiled_patterns:
            for pattern_info in compiled_patterns["*"]:
                # For wildcard, check against all possible field paths mentioned in prompt_fields
                prompt_fields_config = setup_config.get('prompt_fields', {})
                all_field_paths = prompt_fields_config.get('question_fields', [])
                answer_field = prompt_fields_config.get('answer_field', '')
                if answer_field:
                    all_field_paths.append(answer_field)
                
                for field_path in all_field_paths:
                    value = self.resolve_field_path(row_data, field_path)
                    if value is not None:
                        if self._value_matches_pattern(str(value), pattern_info):
                            is_regex, pattern = pattern_info
                            pattern_str = pattern.pattern if is_regex else pattern
                            return True, f"Global pattern '{pattern_str}' matched field '{field_path}' with value '{value}'"
        
        # Check specific field path rules
        for field_path, pattern_list in compiled_patterns.items():
            if field_path == "*":  # Already checked above
                continue
            
            # Resolve the field path value
            value = self.resolve_field_path(row_data, field_path)
            if value is None:
                continue
            
            for pattern_info in pattern_list:
                if self._value_matches_pattern(str(value), pattern_info):
                    is_regex, pattern = pattern_info
                    pattern_str = pattern.pattern if is_regex else pattern
                    return True, f"Field path '{field_path}' pattern '{pattern_str}' matched value '{value}'"
        
        return False, ""
    
    def filter_dataset_rows(self, df: pd.DataFrame, setup_name: str) -> Tuple[pd.DataFrame, int, List[str]]:
        """
        Filter dataset rows based on pruning configuration.
        Updated to support JSON field paths.
        
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
        file_type_keys = ['parquet_file', 'csv_file', 'json_file', 'jsonl_file']
        for key in file_type_keys:
            if key in dataset_config:
                return dataset_config[key]
        
        # Fallback: extract from URL
        parsed_url = urlparse(url)
        url_filename = os.path.basename(parsed_url.path)
        
        # Handle query parameters (like ?download=true)
        if '?' in url_filename:
            url_filename = url_filename.split('?')[0]
        
        return url_filename if url_filename else f"{setup_name}_data"
    
    def download_dataset(self, setup_name: str) -> bool:
        """
        Download dataset from configured URL with support for ZIP and direct downloads.
        
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
    # DATASET LOADING WITH JSON/JSONL SUPPORT
    # =============================================================================
    
    def load_dataset(self, setup_name: str, ensure_download: bool = True) -> Optional[pd.DataFrame]:
        """
        Load a dataset by setup name, with automatic downloading if needed.
        Supports CSV, Parquet, JSON, and JSONL file formats.
        
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
            elif file_type == 'json':
                df = self._load_json_dataset(dataset_path)
                logger.info(f"Loaded JSON dataset for setup {setup_name} with {len(df)} rows")
            elif file_type == 'jsonl':
                df = self._load_jsonl_dataset(dataset_path)
                logger.info(f"Loaded JSONL dataset for setup {setup_name} with {len(df)} rows")
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
    
    def _load_json_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load a JSON file as a DataFrame.
        Handles both array of objects and single object formats.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            Exception: If file cannot be loaded or parsed
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Array of objects - each object becomes a row
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Single object - convert to single-row DataFrame
            return pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported JSON structure: expected array or object, got {type(data)}")
    
    def _load_jsonl_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load a JSONL (JSON Lines) file as a DataFrame.
        Each line should contain a valid JSON object.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            Exception: If file cannot be loaded or parsed
        """
        records = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        if not records:
            raise ValueError(f"No valid JSON records found in {file_path}")
        
        return pd.DataFrame(records)
    
    # =============================================================================
    # DATASET INFORMATION AND ANALYSIS (updated for JSON support)
    # =============================================================================
    
    def get_dataset_info(self, setup_name: str) -> Dict:
        """
        Get comprehensive information about a dataset.
        
        Includes configuration, download status, statistics, and field analysis.
        Updated to support JSON/JSONL files.
        
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
            'file_path': (dataset_config.get('csv_file') or dataset_config.get('parquet_file') 
                         or dataset_config.get('json_file') or dataset_config.get('jsonl_file')),
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
            
            # For JSON/JSONL datasets, show sample field path values
            if file_type in ['json', 'jsonl']:
                question_fields = prompt_fields_config.get('question_fields', [])
                answer_field = prompt_fields_config.get('answer_field', '')
                
                if len(df) > 0:
                    sample_row = df.iloc[0]
                    info['sample_field_values'] = {}
                    
                    # Show sample values for question fields
                    for field_path in question_fields[:3]:  # Limit to first 3 for brevity
                        value = self.resolve_field_path(sample_row, field_path)
                        info['sample_field_values'][field_path] = str(value)[:100] if value else None
                    
                    # Show sample value for answer field
                    if answer_field:
                        value = self.resolve_field_path(sample_row, answer_field)
                        info['sample_field_values'][answer_field] = str(value)[:100] if value else None
        
        elif is_downloaded:
            # Try to get basic info without fully loading (for large datasets)
            try:
                if file_type == 'csv':
                    df_sample = pd.read_csv(dataset_path, nrows=5)
                elif file_type == 'parquet':
                    df_full = pd.read_parquet(dataset_path)
                    df_sample = df_full.head(5)
                elif file_type == 'json':
                    # For JSON, just load it (usually small files)
                    df_sample = self._load_json_dataset(dataset_path)
                elif file_type == 'jsonl':
                    # For JSONL, load first few lines manually
                    records = []
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i >= 5:
                                break
                            line = line.strip()
                            if line:
                                try:
                                    records.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                    df_sample = pd.DataFrame(records) if records else None
                else:
                    df_sample = None
                
                if df_sample is not None and len(df_sample) > 0:
                    info.update({
                        'columns': list(df_sample.columns),
                        'sample_row': df_sample.iloc[0].to_dict()
                    })
            except Exception as e:
                logger.warning(f"Could not read sample from setup {setup_name}: {e}")
        
        return info
    
    # =============================================================================
    # DATASET VALIDATION (updated for JSON field paths)
    # =============================================================================
    
    def validate_dataset_fields(self, setup_name: str) -> bool:
        """
        Validate that dataset has all required fields for processing.
        Updated to support JSON field path validation.
        
        Checks that question field paths and answer field path can be resolved in the dataset.
        Provides helpful error messages for troubleshooting.
        
        Args:
            setup_name: Name of setup to validate
            
        Returns:
            bool: True if all required field paths are accessible, False otherwise
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
        
        question_field_paths = prompt_fields_config.get('question_fields', [])
        answer_field_path = prompt_fields_config.get('answer_field', '')
        
        required_field_paths = question_field_paths + ([answer_field_path] if answer_field_path else [])
        
        if len(df) == 0:
            logger.error(f"Dataset for setup {setup_name} is empty")
            return False
        
        # Test field path resolution on first few rows
        test_rows = min(3, len(df))
        missing_field_paths = []
        
        for field_path in required_field_paths:
            accessible_count = 0
            for i in range(test_rows):
                row = df.iloc[i]
                value = self.resolve_field_path(row, field_path)
                if value is not None:
                    accessible_count += 1
            
            # Field path is considered valid if it resolves in at least one test row
            if accessible_count == 0:
                missing_field_paths.append(field_path)
        
        if missing_field_paths:
            logger.error(f"Dataset for setup {setup_name} missing or inaccessible field paths: {missing_field_paths}")
            logger.info(f"Expected question field paths: {question_field_paths}")
            logger.info(f"Expected answer field path: {answer_field_path}")
            
            # For JSON/JSONL datasets, show sample structure
            file_type = self.get_dataset_file_type(setup_name)
            if file_type in ['json', 'jsonl'] and len(df) > 0:
                logger.info("Sample row structure:")
                sample_data = df.iloc[0].to_dict() if hasattr(df.iloc[0], 'to_dict') else df.iloc[0]
                logger.info(f"  {json.dumps(sample_data, indent=2, default=str)[:500]}...")
            
            return False
        
        logger.info(f"Dataset for setup {setup_name} validation passed")
        return True
    
    # =============================================================================
    # GENERIC DATA PROCESSING (updated for JSON field paths)
    # =============================================================================
    
    def get_expected_answer(self, row: Union[pd.Series, Dict], setup_name: str) -> str:
        """
        Extract expected answer from a dataset row using field path resolution.
        
        Args:
            row: Single row from dataset (pd.Series or dict)
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
        answer_field_path = prompt_fields_config.get('answer_field', '')
        
        if answer_field_path:
            answer_value = self.resolve_field_path(row, answer_field_path)
            if answer_value is not None and not (isinstance(answer_value, float) and pd.isna(answer_value)):
                return str(answer_value)
        
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