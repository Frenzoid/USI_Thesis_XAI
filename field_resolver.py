import pandas as pd
from typing import Dict, List, Any, Union
import re
import json
from utils import setup_logging

logger = setup_logging("field_resolver")

class FieldPathResolver:
    """
    Utility class for resolving field paths in data structures.
    Handles both simple field access and complex nested JSON path resolution.
    Supports JSON column parsing with 'json:column_name.field.path' syntax.
    """
    
    @staticmethod
    def _resolve_array_concatenation_path(data: Union[Dict, List, Any], path: str) -> str:
        """
        Handle array concatenation with [*] syntax.
        
        Examples:
        - 'contexts[*]' -> concatenate all array elements as strings
        - 'contexts[*].text' -> get 'text' field from each element and concatenate
        - 'nested.contexts[*].content' -> navigate to nested.contexts, then concatenate content fields
        
        Args:
            data: Parsed JSON data
            path: Field path containing [*] syntax
            
        Returns:
            Concatenated string or None if resolution fails
        """
        try:
            # Find the position of [*] in the path
            if '[*]' not in path:
                return None
            
            # Split path into before and after [*]
            parts = path.split('[*]', 1)
            before_star = parts[0]
            after_star = parts[1] if len(parts) > 1 else ""
            
            # Navigate to the array using the path before [*]
            current_data = data
            
            if before_star:
                # Remove trailing dot if present
                before_star = before_star.rstrip('.')
                if before_star:
                    current_data = FieldPathResolver._resolve_complex_field_path(current_data, before_star)
                    if current_data is None:
                        logger.debug(f"Could not resolve path before [*]: {before_star}")
                        return None
            
            # Ensure we have an array
            if not isinstance(current_data, (list, tuple)):
                logger.debug(f"Expected array for [*] operation, got {type(current_data)}")
                return None
            
            # Handle the path after [*]
            if after_star.startswith('.'):
                after_star = after_star[1:]  # Remove leading dot
            
            concatenated_values = []
            
            for i, item in enumerate(current_data):
                if item is None:
                    logger.debug(f"Array element {i} is None, skipping")
                    continue
                
                if after_star:
                    # Apply the remaining path to this array element
                    value = FieldPathResolver._resolve_complex_field_path(item, after_star)
                    if value is not None:
                        concatenated_values.append(str(value))
                    else:
                        logger.debug(f"Array element {i}: could not resolve path '{after_star}'")
                else:
                    # No path after [*], just convert element to string
                    concatenated_values.append(str(item))
            
            # Join all values with a space separator
            if concatenated_values:
                result = ' '.join(concatenated_values)
                logger.debug(f"Array concatenation result: {len(concatenated_values)} elements -> {len(result)} characters")
                return result
            else:
                logger.debug("Array concatenation resulted in no valid values")
                return None
                
        except Exception as e:
            logger.debug(f"Error in array concatenation for path '{path}': {e}")
            return None
    
    @staticmethod
    def resolve_field_path(data: Union[Dict, pd.Series, Any], path: str) -> Any:
        """
        Resolve a field path against data structure with optimized handling.
        Supports JSON column parsing with 'json:column_name.field.path' syntax.
        
        Args:
            data: The data structure to navigate (dict, pd.Series, or primitive)
            path: The field path to resolve, optionally with json: prefix
            
        Returns:
            The value at the specified path, or None if path doesn't exist
        """
        if not path or path == "":
            return data
        
        # Convert pandas Series to dict if needed
        if isinstance(data, pd.Series):
            data = data.to_dict()
        
        # Handle JSON column parsing
        if path.startswith("json:"):
            return FieldPathResolver._resolve_json_field_path(data, path[5:])
        
        # Fast path for simple field access
        if FieldPathResolver._is_simple_field(path):
            return FieldPathResolver._resolve_simple_field(data, path)
        
        # Complex path resolution
        return FieldPathResolver._resolve_complex_field_path(data, path)
    
    @staticmethod
    def _resolve_json_field_path(data: Dict, path: str) -> Any:
        """
        Handle JSON column parsing with format 'column_name.json_field.path'.
        Supports array concatenation with [*] syntax.
        
        Args:
            data: Row data dictionary
            path: Field path after removing 'json:' prefix
            
        Returns:
            Parsed value from JSON column or None if parsing fails
        """
        if '.' not in path:
            # Simple case: json:column_name (return entire parsed JSON)
            column_name = path
            json_path = ""
        else:
            # Complex case: json:column_name.field.path
            parts = path.split('.', 1)
            column_name = parts[0]
            json_path = parts[1]
        
        # Get the JSON string from the column
        if not isinstance(data, dict) or column_name not in data:
            logger.debug(f"Column '{column_name}' not found in data")
            return None
        
        json_string = data[column_name]
        if json_string is None or (isinstance(json_string, float) and pd.isna(json_string)):
            logger.debug(f"Column '{column_name}' contains null/NaN value")
            return None
        
        # Parse the JSON string
        try:
            if isinstance(json_string, str):
                parsed_json = json.loads(json_string)
            else:
                # Already parsed (might be dict/list)
                parsed_json = json_string
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Failed to parse JSON in column '{column_name}': {e}")
            return None
        
        # If no further path specified, return the parsed JSON
        if not json_path:
            return parsed_json
        
        # Check for array concatenation syntax [*]
        if '[*]' in json_path:
            return FieldPathResolver._resolve_array_concatenation_path(parsed_json, json_path)
        
        # Apply the remaining field path to the parsed JSON
        return FieldPathResolver._resolve_complex_field_path(parsed_json, json_path)
    
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
        Supports JSON column parsing with json: prefix.
        
        Args:
            row: Data row (dict for JSON data, pd.Series for CSV/Parquet data)
            field_paths: List of field paths to extract (may include json: prefixes)
            
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
        Includes support for JSON column parsing validation.
        
        Args:
            data: Sample data row for testing
            field_paths: List of field paths to validate (may include json: prefixes)
            
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
                
                # Enhanced result info for JSON fields
                is_json_field = field_path.startswith("json:")
                result_info = {
                    'resolved': value is not None,
                    'value_type': type(value).__name__ if value is not None else None,
                    'value_preview': str(value)[:100] if value is not None else None,
                    'is_json_field': is_json_field
                }
                
                if is_json_field:
                    # Additional validation for JSON fields
                    column_name = field_path[5:].split('.')[0]  # Remove 'json:' and get column name
                    if isinstance(data, pd.Series):
                        data_dict = data.to_dict()
                    else:
                        data_dict = data
                    
                    if column_name in data_dict:
                        json_string = data_dict[column_name]
                        result_info['json_column_exists'] = True
                        result_info['json_string_length'] = len(str(json_string)) if json_string else 0
                        
                        # Try to validate JSON parsing
                        try:
                            if isinstance(json_string, str):
                                json.loads(json_string)
                                result_info['json_valid'] = True
                            else:
                                result_info['json_valid'] = True  # Already parsed
                        except (json.JSONDecodeError, TypeError):
                            result_info['json_valid'] = False
                            validation_result['warnings'].append(f"JSON field '{field_path}' contains invalid JSON in column '{column_name}'")
                    else:
                        result_info['json_column_exists'] = False
                        validation_result['warnings'].append(f"JSON field '{field_path}' references non-existent column '{column_name}'")
                
                validation_result['field_path_results'][field_path] = result_info
                
                if value is None:
                    validation_result['warnings'].append(f"Field path '{field_path}' resolved to None")
                    
            except Exception as e:
                validation_result['valid'] = False
                error_msg = f"Error resolving field path '{field_path}': {e}"
                validation_result['errors'].append(error_msg)
                validation_result['field_path_results'][field_path] = {'error': str(e)}
        
        return validation_result