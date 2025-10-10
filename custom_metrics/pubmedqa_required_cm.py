"""
Custom metrics for Evidence-Based Medical Research Question Answering task.

Task: Given a research question and experimental context (WITHOUT conclusions),
determine the answer based on experimental evidence only.

Expected output: Single lowercase word - "yes", "no", or "maybe"

Key difference from standard medical QA: Models don't have access to study conclusions,
must reason from experimental data alone. Observation: Models tend to overuse "maybe"
when told they lack conclusions, showing excessive caution.

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


def maybe_overuse_penalty(response_data):
    """
    Penalize inappropriate overuse of "maybe" as a conservative fallback.
    
    When models don't have access to conclusions, they tend to overuse "maybe"
    even when experimental evidence clearly supports "yes" or "no".
    
    This metric rewards:
    - Definitive answers ("yes" or "no") - shows confidence
    - "maybe" when it's actually the correct answer
    
    This metric penalizes:
    - "maybe" when experimental evidence supports a definitive answer
    
    Scoring:
    - 1.0: Response is definitive ("yes" or "no"), OR response is "maybe" and expected is "maybe"
    - 0.0: Response is "maybe" when expected answer was definitive ("yes" or "no")
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Overuse penalty score (1.0 = appropriate, 0.0 = overusing "maybe")
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip().lower()
    expected = response_data.get('expected_output', '').strip().lower()
    
    # Valid answers
    valid_answers = {'yes', 'no', 'maybe'}
    definitive_answers = {'yes', 'no'}
    
    # Extract answer from response
    words = generated.split()
    
    if len(words) == 1 and words[0] in valid_answers:
        extracted = words[0]
    else:
        # Try to find yes/no/maybe in the text
        for answer in valid_answers:
            pattern = r'\b' + answer + r'\b'
            if re.search(pattern, generated):
                extracted = answer
                break
        else:
            # No valid answer found - assume "maybe" (conservative)
            extracted = 'maybe'
    
    # Check for maybe overuse
    if extracted in definitive_answers:
        # Gave a definitive answer - good, shows confidence
        return 1.0
    elif extracted == 'maybe' and expected == 'maybe':
        # "maybe" is correct - appropriate use
        return 1.0
    elif extracted == 'maybe' and expected in definitive_answers:
        # Overusing "maybe" when evidence supports definitive answer
        return 0.0
    
    return 0.5  # Fallback


def correct_definitive_answers(response_data):
    """
    Among definitive answers, measure correctness (confidence calibration).
    
    When models give "yes" or "no" (not "maybe"), how often are they correct?
    This measures whether the model's confidence is well-calibrated.
    
    - High score: Model's definitive answers are usually correct (good calibration)
    - Low score: Model gives definitive answers but they're often wrong (overconfident)
    - Neutral (0.5): Model only gives "maybe" (not applicable)
    
    Scoring:
    - 1.0: Definitive answer and correct
    - 0.0: Definitive answer and incorrect
    - 0.5: Response was "maybe" (neutral - not applicable)
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Confidence calibration score (1.0 = correct when definitive)
    """
    if not response_data.get('success', False):
        return 0.5  # Neutral
    
    generated = response_data.get('response', '').strip().lower()
    expected = response_data.get('expected_output', '').strip().lower()
    
    # Valid answers
    valid_answers = {'yes', 'no', 'maybe'}
    definitive_answers = {'yes', 'no'}
    
    # Extract answer from response
    words = generated.split()
    
    if len(words) == 1 and words[0] in valid_answers:
        extracted = words[0]
    else:
        # Try to find yes/no/maybe in the text
        for answer in valid_answers:
            pattern = r'\b' + answer + r'\b'
            if re.search(pattern, generated):
                extracted = answer
                break
        else:
            # No valid answer found
            return 0.5  # Neutral
    
    # Check if response is definitive
    if extracted not in definitive_answers:
        # Response was "maybe" - not applicable
        return 0.5  # Neutral
    
    # Response is definitive ("yes" or "no")
    # Check if it matches expected
    if extracted == expected:
        return 1.0  # Correct definitive answer
    else:
        return 0.0  # Incorrect definitive answer


# Metric registry for Evidence-Based Medical Research Question Answering task
EVIDENCE_BASED_MEDICAL_QA_METRICS = {
    'answer_correctness': answer_correctness,
    'maybe_overuse_penalty': maybe_overuse_penalty,
    'correct_definitive_answers': correct_definitive_answers
}