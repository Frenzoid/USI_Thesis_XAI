import pandas as pd
from typing import Dict, List, Any, Union
import re
from utils import setup_logging

logger = setup_logging("field_resolver")

class FieldPathResolver:
    """
    Utility class for resolving field paths in data structures.
    Handles both simple field access and complex nested JSON path resolution.
    """
    
    @staticmethod
    def resolve_field_path(data: Union[Dict, pd.Series, Any], path: str) -> Any:
        """
        Resolve a field path against data structure with optimized handling.
        
        Args:
            data: The data structure to navigate (dict, pd.Series, or primitive)
            path: The field path to resolve
            
        Returns:
            The value at the specified path, or None if path doesn't exist
        """
        if not path or path == "":
            return data
        
        # Convert pandas Series to dict if needed
        if isinstance(data, pd.Series):
            data = data.to_dict()
        
        # Fast path for simple field access
        if FieldPathResolver._is_simple_field(path):
            return FieldPathResolver._resolve_simple_field(data, path)
        
        # Complex path resolution
        return FieldPathResolver._resolve_complex_field_path(data, path)
    
    @staticmethod
    def _is_simple_field(path: str) -> bool:
        """Check if this is a simple field (no dots or brackets)"""
        return '.' not in path and '[' not in path
    
    @staticmethod
    def _resolve_simple_field(data: Dict, field: str) -> Any:
        """Handle simple field access"""
        if not isinstance(data, dict):
            return None
        return data.get(field)
    
    @staticmethod
    def _resolve_complex_field_path(data: Union[Dict, List, Any], path: str) -> Any:
        """Handle complex nested field path resolution"""
        try:
            parts = FieldPathResolver._parse_field_path(path)
            
            current = data
            for part in parts:
                if isinstance(part, int):
                    if not isinstance(current, (list, tuple)):
                        logger.debug(f"Expected array at path component but got {type(current)}")
                        return None
                    if part < 0 or part >= len(current):
                        logger.debug(f"Array index {part} out of bounds (length: {len(current)})")
                        return None
                    current = current[part]
                else:
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
    
    @staticmethod
    def _parse_field_path(path: str) -> List[Union[str, int]]:
        """
        Parse a field path into components using regex for better performance.
        
        Args:
            path: Field path like "context.questions[0].text"
            
        Returns:
            List of path components (strings for properties, ints for array indices)
        """
        parts = []
        
        # Split by dots first, then handle array indices
        segments = path.split('.')
        
        for segment in segments:
            if '[' in segment:
                # Handle array access: "questions[0]" or "items[2]"
                matches = re.findall(r'([^\[]+)|\[(\d+)\]', segment)
                for match in matches:
                    if match[0]:  # Property name
                        parts.append(match[0])
                    if match[1]:  # Array index
                        parts.append(int(match[1]))
            else:
                # Simple property
                if segment:  # Skip empty segments
                    parts.append(segment)
        
        return parts
    
    @staticmethod
    def extract_field_values(row: Union[Dict, pd.Series], field_paths: List[str]) -> List[str]:
        """
        Extract values from multiple field paths in a data row.
        
        Args:
            row: Data row (dict for JSON data, pd.Series for CSV/Parquet data)
            field_paths: List of field paths to extract
            
        Returns:
            List of string values extracted from the specified paths
        """
        values = []
        
        for path in field_paths:
            try:
                value = FieldPathResolver.resolve_field_path(row, path)
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    values.append("")
                else:
                    values.append(str(value))
            except Exception as e:
                logger.warning(f"Error extracting field path '{path}': {e}")
                values.append("")
        
        return values
    
    @staticmethod
    def validate_field_paths(data: Union[Dict, pd.Series], field_paths: List[str]) -> Dict[str, Any]:
        """
        Validate that field paths can be resolved against sample data.
        
        Args:
            data: Sample data row for testing
            field_paths: List of field paths to validate
            
        Returns:
            Dict containing validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'field_values': {},
            'field_path_results': {}
        }
        
        for field_path in field_paths:
            try:
                value = FieldPathResolver.resolve_field_path(data, field_path)
                validation_result['field_values'][field_path] = str(value) if value is not None else None
                validation_result['field_path_results'][field_path] = {
                    'resolved': value is not None,
                    'value_type': type(value).__name__ if value is not None else None,
                    'value_preview': str(value)[:100] if value is not None else None
                }
                
                if value is None:
                    validation_result['warnings'].append(f"Field path '{field_path}' resolved to None")
            except Exception as e:
                validation_result['valid'] = False
                error_msg = f"Error resolving field path '{field_path}': {e}"
                validation_result['errors'].append(error_msg)
                validation_result['field_path_results'][field_path] = {'error': str(e)}
        
        return validation_result