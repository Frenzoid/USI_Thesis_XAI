"""
Custom metrics for Causal Reasoning Selection task.

Task: Given a premise, select which hypothesis (0 or 1) represents the correct causal relationship.
Expected output: ONLY "0" or "1" - no explanation needed.

Each metric function receives a response dictionary with the following fields:
- prompt: The input prompt sent to the model
- response: The model's generated response  
- expected_output: The expected/reference response (should be "0" or "1")
- question_values: List of question field values used to populate the prompt
- success: Boolean indicating if the response generation was successful
- error: Error message if success is False, None otherwise

Each function should return a float value between 0.0 and 1.0.
"""

import re


def extract_answer(response_text):
    """
    Helper function to extract the answer (0 or 1) from the response.
    
    Search strategy:
    1. Look for "hypothesis 0" or "hypothesis 1" (case insensitive)
    2. If not found, look for standalone "0" or "1" in the text
    
    Args:
        response_text: The model's response text
        
    Returns:
        str: "0", "1", or None if no answer found
    """
    if not response_text:
        return None
    
    response_lower = response_text.lower()
    
    # Strategy 1: Look for "hypothesis X" patterns
    hypothesis_pattern = r'hypothesis\s*[:\-]?\s*([01])'
    matches = re.findall(hypothesis_pattern, response_lower)
    if matches:
        return matches[0]
    
    # Strategy 2: Look for "correct/answer is X" patterns
    correct_pattern = r'(?:correct|answer|select|choose)(?:\s+hypothesis)?(?:\s+is)?[:\s]+([01])'
    matches = re.findall(correct_pattern, response_lower)
    if matches:
        return matches[0]
    
    # Strategy 3: Look for standalone 0 or 1
    standalone_pattern = r'(?:^|[^0-9])([01])(?:[^0-9]|$)'
    matches = re.findall(standalone_pattern, response_text)
    if matches:
        return matches[0]
    
    return None


def answer_correctness(response_data):
    """
    Check if the extracted answer matches the expected output.
    
    This is the primary metric - did the model select the right hypothesis?
    Extracts "0" or "1" from the response using various strategies.
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: 1.0 if correct, 0.0 if incorrect or no answer found
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '')
    expected = response_data.get('expected_output', '').strip()
    
    extracted = extract_answer(generated)
    
    if extracted is None:
        return 0.0
    
    return 1.0 if extracted == expected else 0.0


def follows_format_instruction(response_data):
    """
    Check if the response contains ONLY the answer (0 or 1).
    
    The instruction asks to "Select the correct hypothesis (0 for Hypothesis 1, 
    1 for Hypothesis 2)" - meaning the response should be just "0" or "1".
    
    This metric penalizes models that provide explanations when only a 
    single digit was requested.
    
    Scoring:
    - 1.0: Response is exactly "0" or "1" (or with minimal whitespace)
    - 0.8: Response is ≤15 characters (e.g., "1" with brief formatting)
    - 0.5: Response is ≤100 characters (short but includes some explanation)
    - 0.2: Response is >100 characters (verbose, didn't follow format)
    - 0.0: No answer found
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Score based on format compliance (higher = more concise)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    # Check if answer is present
    extracted = extract_answer(generated)
    if extracted is None:
        return 0.0
    
    # Remove common whitespace and check if it's just the answer
    cleaned = generated.strip()
    
    # Perfect: just "0" or "1"
    if cleaned in ['0', '1']:
        return 1.0
    
    # Very good: minimal characters
    if len(cleaned) <= 15:
        return 0.8
    
    # Acceptable: short response
    if len(cleaned) <= 100:
        return 0.5
    
    # Poor: verbose explanation when only number was needed
    return 0.2


def answer_extractability(response_data):
    """
    Measure how easily the answer can be extracted from the response.
    
    Even if the model provides explanation (against instructions), the answer
    should be clear and unambiguous. This metric rewards responses where:
    - Answer appears early in the text
    - Answer is explicitly stated
    - No contradictory statements
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Score for how extractable/clear the answer is
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    extracted = extract_answer(generated)
    if extracted is None:
        return 0.0
    
    response_lower = generated.lower()
    
    # Find position of answer in text
    hypothesis_pattern = f'hypothesis\\s*[:\\-]?\\s*{extracted}'
    match = re.search(hypothesis_pattern, response_lower)
    
    if match:
        position = match.start()
    else:
        # Look for standalone number
        standalone_pattern = f'(?:^|[^0-9])({extracted})(?:[^0-9]|$)'
        match = re.search(standalone_pattern, response_lower)
        position = match.start() if match else len(generated)
    
    response_length = len(generated)
    relative_position = position / response_length if response_length > 0 else 1.0
    
    # Base score on position (earlier = better)
    if relative_position <= 0.1:
        position_score = 1.0
    elif relative_position <= 0.3:
        position_score = 0.8
    elif relative_position <= 0.5:
        position_score = 0.6
    else:
        position_score = 0.4
    
    # Check for contradictions (multiple different answers)
    all_matches = re.findall(r'hypothesis\s*[:\-]?\s*([01])', response_lower)
    if len(all_matches) > 1 and len(set(all_matches)) > 1:
        # Contradictory answers found
        return position_score * 0.3
    
    # Check for explicit statement
    explicit_patterns = [
        r'(?:correct|answer|select)\s+(?:hypothesis\s+)?(?:is\s+)?',
        r'^[01]$'  # Just the answer alone
    ]
    
    is_explicit = any(re.search(pattern, response_lower) for pattern in explicit_patterns)
    
    if is_explicit:
        return min(position_score + 0.2, 1.0)
    
    return position_score


# Metric registry for Causal Reasoning Selection task
CAUSAL_REASONING_METRICS = {
    'answer_correctness': answer_correctness,
    'follows_format_instruction': follows_format_instruction,
    'answer_extractability': answer_extractability
}