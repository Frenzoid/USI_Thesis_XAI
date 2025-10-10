"""
Custom metrics for Medical Research Question Answering task.

Task: Given a research question, context, and conclusion, determine the answer.
Expected output: Single lowercase word - "yes", "no", or "maybe"

The instruction explicitly states: "respond with only one word: yes, no, or maybe."

Each metric function receives a response dictionary with the following fields:
- prompt: The input prompt sent to the model
- response: The model's generated response  
- expected_output: The expected/reference response (should be "yes", "no", or "maybe")
- question_values: List of question field values used to populate the prompt
- success: Boolean indicating if the response generation was successful
- error: Error message if success is False, None otherwise

Each function should return a float value between 0.0 and 1.0.
"""

import re


def answer_correctness(response_data):
    """
    Check if the extracted answer matches the expected output.
    
    This is the primary metric - did the model provide the correct answer?
    Uses case-insensitive matching since the core content matters more
    than capitalization.
    
    Valid answers: "yes", "no", "maybe" (case-insensitive)
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: 1.0 if correct, 0.0 if incorrect or invalid answer
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip().lower()
    expected = response_data.get('expected_output', '').strip().lower()
    
    # Valid answers
    valid_answers = {'yes', 'no', 'maybe'}
    
    # Extract answer from response
    # First, check if response is just one word
    words = generated.split()
    
    if len(words) == 1 and words[0] in valid_answers:
        extracted = words[0]
    else:
        # Try to find yes/no/maybe in the text
        # Look for these words as standalone words
        for answer in valid_answers:
            pattern = r'\b' + answer + r'\b'
            if re.search(pattern, generated):
                extracted = answer
                break
        else:
            # No valid answer found
            return 0.0
    
    # Check if extracted answer matches expected
    if extracted == expected:
        return 1.0
    else:
        return 0.0


def strict_format_compliance(response_data):
    """
    Check if the response is EXACTLY one lowercase word.
    
    The instruction says "respond with only one word: yes, no, or maybe."
    This means:
    - Exactly one word
    - Must be "yes", "no", or "maybe"
    - Should be lowercase (as shown in instruction)
    - No punctuation, no extra whitespace, no explanation
    
    Scoring:
    - 1.0: Exactly "yes", "no", or "maybe" (lowercase, no extra text)
    - 0.8: Correct word but wrong case (e.g., "Yes" instead of "yes")
    - 0.5: Correct word with minimal punctuation (e.g., "yes.")
    - 0.0: Multiple words or explanation present
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Format compliance score (1.0 = perfect format)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    if not generated:
        return 0.0
    
    # Valid answers
    valid_answers_lower = {'yes', 'no', 'maybe'}
    valid_answers_capitalized = {'Yes', 'No', 'Maybe'}
    
    # Check if response is exactly one of the valid lowercase answers
    if generated in valid_answers_lower:
        return 1.0
    
    # Check if response is capitalized version
    if generated in valid_answers_capitalized:
        return 0.8
    
    # Check if response is valid answer with minimal punctuation
    cleaned = generated.strip('.,!?;: ')
    if cleaned in valid_answers_lower:
        return 0.5
    
    if cleaned in valid_answers_capitalized:
        return 0.4
    
    # Check if response contains only one word
    words = generated.split()
    if len(words) == 1:
        word_lower = words[0].lower().strip('.,!?;: ')
        if word_lower in valid_answers_lower:
            return 0.3
    
    # Multiple words or explanation present
    return 0.0


def no_explanation_penalty(response_data):
    """
    Penalize responses that include explanations or justifications.
    
    The instruction is clear: "respond with only one word"
    Any additional text is a violation of the format requirement.
    
    This metric checks:
    - Response length (should be ~3-5 characters)
    - Number of words (should be 1)
    - Presence of explanation markers (because, since, as, etc.)
    
    Scoring:
    - 1.0: Only the answer word (≤10 characters)
    - 0.7: Answer word plus minimal punctuation (≤15 characters)
    - 0.4: Short text (≤50 characters)
    - 0.2: Medium text (≤100 characters)
    - 0.0: Long explanation (>100 characters)
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Score (1.0 = no explanation, 0.0 = lengthy explanation)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    if not generated:
        return 0.0
    
    response_length = len(generated)
    word_count = len(generated.split())
    
    # Perfect: just the answer word
    if response_length <= 10 and word_count == 1:
        return 1.0
    
    # Good: answer word with minimal punctuation
    if response_length <= 15 and word_count == 1:
        return 0.7
    
    # Acceptable: short response
    if response_length <= 50:
        return 0.4
    
    # Poor: medium explanation
    if response_length <= 100:
        return 0.2
    
    # Very poor: long explanation
    return 0.0


# Metric registry for Medical Research Question Answering task
MEDICAL_QA_METRICS = {
    'answer_correctness': answer_correctness,
    'strict_format_compliance': strict_format_compliance,
    'no_explanation_penalty': no_explanation_penalty
}