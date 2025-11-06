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
    
    This metric penalizes (but not too harshly):
    - "maybe" when experimental evidence supports a definitive answer
    
    Scoring:
    - 1.0: Response is definitive ("yes" or "no"), OR response is "maybe" and expected is "maybe"
    - 0.3: Response is "maybe" when expected answer was definitive (shows overcaution)
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Overuse penalty score (1.0 = appropriate, lower = overusing "maybe")
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
        # Penalty but not too harsh (0.3 instead of 0.0)
        return 0.3
    
    return 0.5  # Fallback


def confidence_calibration(response_data):
    """
    Measure whether the model's confidence level is appropriate.
    
    This metric assesses whether the model makes definitive claims appropriately:
    - When model says "yes" or "no", is it usually correct? (confidence)
    - When model says "maybe", is that reasonable? (caution)
    
    Scoring:
    - 1.0: Definitive answer and correct (well-calibrated confidence)
    - 0.5: Definitive answer but incorrect (overconfident)
    - 0.7: Says "maybe" when answer is actually "maybe" (appropriate caution)
    - 0.5: Says "maybe" when answer was definitive (overcautious, not as bad as overconfident)
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Calibration score (1.0 = well-calibrated)
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
            # No valid answer found
            return 0.0
    
    # Response is definitive ("yes" or "no")
    if extracted in definitive_answers:
        if extracted == expected:
            return 1.0  # Correct definitive answer (well-calibrated)
        else:
            return 0.5  # Incorrect definitive answer (overconfident)
    
    # Response is "maybe"
    if extracted == 'maybe':
        if expected == 'maybe':
            return 0.7  # Correctly uncertain (appropriate caution)
        else:
            return 0.5  # Overcautious (not as bad as overconfident)
    
    return 0.0  # Fallback


# Metric registry for Evidence-Based Medical Research Question Answering task
EVIDENCE_BASED_MEDICAL_QA_METRICS = {
    'answer_correctness': answer_correctness,
    'maybe_overuse_penalty': maybe_overuse_penalty,
    'confidence_calibration': confidence_calibration
}
