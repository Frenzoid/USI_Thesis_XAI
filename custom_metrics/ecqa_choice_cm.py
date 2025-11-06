"""
Custom metrics for Commonsense Reasoning Multiple Choice task.

Task: Given a question and multiple choices, select the single best choice based on commonsense reasoning.
Expected output: Just the chosen answer word/phrase (e.g., "state", "sing songs", "disaster") - NO explanation.

Each metric function receives a response dictionary with the following fields:
- prompt: The input prompt sent to the model
- response: The model's generated response  
- expected_output: The expected/reference response (the correct choice)
- question_values: List of question field values used to populate the prompt
- success: Boolean indicating if the response generation was successful
- error: Error message if success is False, None otherwise

Each function should return a float value between 0.0 and 1.0.
"""

import re


def extract_choices_from_question_values(question_values):
    """
    Helper function to extract choices from question_values.
    
    Handles two formats:
    1. With topic: [topic, question, choice1, choice2, ...]
    2. Without topic: [question, choice1, choice2, ...]
    
    Args:
        question_values: List of question field values
        
    Returns:
        list: List of choices, or None if not found
    """
    if not question_values or len(question_values) < 2:
        return None
    
    # Check if first element is a short topic or a long question
    first_element = question_values[0]
    first_element_word_count = len(first_element.split())
    
    # If first element is short (≤5 words) and second element is longer,
    # assume first is topic
    if len(question_values) > 2:
        second_element_word_count = len(question_values[1].split())
        if first_element_word_count <= 5 and second_element_word_count > 5:
            # Format: [topic, question, choice1, choice2, ...]
            return question_values[2:]
    
    # Otherwise, assume no topic
    # Format: [question, choice1, choice2, ...]
    return question_values[1:]


def extract_answer(response_text, choices=None):
    """
    Helper function to extract the answer from the response.
    
    Search strategy:
    1. Look for "choice is" or "answer is" patterns followed by quoted text
    2. Look for "choice is" or "answer is" patterns followed by unquoted text
    3. If choices provided, look for any choice appearing in the text
    4. Look for quoted phrases
    
    Args:
        response_text: The model's response text
        choices: Optional list of possible choices to match against
        
    Returns:
        str: Extracted answer or None if no answer found
    """
    if not response_text:
        return None
    
    response_lower = response_text.lower().strip()
    
    # Strategy 1: Look for explicit "choice/answer is" patterns with quotes
    quoted_patterns = [
        r'(?:choice|answer|select|option)\s+is[:\s]+["\']([^"\']+)["\']',
        r'(?:single\s+best\s+)?choice\s+is[:\s]+["\']([^"\']+)["\']',
        r'best\s+choice[:\s]+["\']([^"\']+)["\']',
    ]
    
    for pattern in quoted_patterns:
        match = re.search(pattern, response_lower)
        if match:
            return match.group(1).strip()
    
    # Strategy 2: Look for "choice/answer is" patterns without quotes
    unquoted_patterns = [
        r'(?:single\s+best\s+)?choice\s+is[:\s]+([a-z\s]+?)(?:\.|,|\n|$)',
        r'(?:correct\s+)?answer\s+is[:\s]+([a-z\s]+?)(?:\.|,|\n|$)',
        r'best\s+choice[:\s]+([a-z\s]+?)(?:\.|,|\n|$)',
    ]
    
    for pattern in unquoted_patterns:
        match = re.search(pattern, response_lower)
        if match:
            answer = match.group(1).strip()
            # Clean up common trailing words
            answer = re.sub(r'\s+(because|as|since|that).*$', '', answer)
            if len(answer) > 0 and len(answer.split()) <= 5:
                return answer
    
    # Strategy 3: If choices provided, look for exact matches
    if choices:
        choices_lower = [c.lower().strip() for c in choices]
        for choice in choices_lower:
            # Look for the choice as a standalone phrase
            pattern = r'\b' + re.escape(choice) + r'\b'
            if re.search(pattern, response_lower):
                return choice
    
    # Strategy 4: Look for quoted phrases (might be the answer)
    quoted_text = re.findall(r'["\']([^"\']{1,30})["\']', response_text)
    if quoted_text:
        # Return the first quoted text that's not too long
        for text in quoted_text:
            if 1 <= len(text.split()) <= 5:
                return text.lower().strip()
    
    return None


def answer_correctness(response_data):
    """
    Check if the extracted answer matches the expected output.
    
    This is the primary metric - did the model select the correct choice?
    Handles various answer formats and attempts fuzzy matching.
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: 1.0 if correct, 0.0 if incorrect or no answer found
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '')
    expected = response_data.get('expected_output', '').strip().lower()
    question_values = response_data.get('question_values', [])
    
    # Extract choices from question_values
    choices = extract_choices_from_question_values(question_values)
    
    extracted = extract_answer(generated, choices)
    
    if extracted is None:
        return 0.0
    
    # Exact match
    if extracted == expected:
        return 1.0
    
    # Fuzzy match: check if one contains the other
    if expected in extracted or extracted in expected:
        return 1.0
    
    # Normalize and retry (remove extra spaces, punctuation)
    extracted_normalized = re.sub(r'[^\w\s]', '', extracted).strip()
    expected_normalized = re.sub(r'[^\w\s]', '', expected).strip()
    
    if extracted_normalized == expected_normalized:
        return 1.0
    
    return 0.0


def follows_format_instruction(response_data):
    """
    Check if the response contains ONLY the answer.
    
    The instruction says "return that choice" - meaning just the answer,
    not an explanation. This metric penalizes models that provide
    unnecessary explanations.
    
    Scoring:
    - 1.0: Response is just the answer (≤15 characters)
    - 0.8: Very brief response (≤50 characters)
    - 0.5: Short response with some explanation (≤150 characters)
    - 0.3: Medium response (≤300 characters)
    - 0.1: Long response (>300 characters)
    - 0.0: No answer found
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Score based on format compliance (higher = more concise)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    question_values = response_data.get('question_values', [])
    
    # Extract choices
    choices = extract_choices_from_question_values(question_values)
    
    # Check if answer is present
    extracted = extract_answer(generated, choices)
    if extracted is None:
        return 0.0
    
    response_length = len(generated)
    
    # Perfect: just the answer
    if response_length <= 15:
        return 1.0
    
    # Very good: minimal text
    if response_length <= 50:
        return 0.8
    
    # Acceptable: short explanation
    if response_length <= 150:
        return 0.5
    
    # Poor: medium explanation
    if response_length <= 300:
        return 0.3
    
    # Very poor: long explanation
    return 0.1


def answer_extractability(response_data):
    """
    Measure how easily and clearly the answer can be extracted.
    
    Even if the model provides explanation (against instructions), the answer
    should be stated clearly and explicitly. This metric rewards responses where:
    - Answer is explicitly introduced (e.g., "The answer is X")
    - Answer appears early in the text
    - No contradictory or multiple conflicting answers
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Score for how extractable/clear the answer is
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    question_values = response_data.get('question_values', [])
    
    choices = extract_choices_from_question_values(question_values)
    
    extracted = extract_answer(generated, choices)
    if extracted is None:
        return 0.0
    
    response_lower = generated.lower()
    
    # Check for explicit statement patterns
    explicit_patterns = [
        r'(?:the\s+)?(?:single\s+best\s+)?choice\s+is',
        r'(?:the\s+)?(?:correct\s+)?answer\s+is',
        r'best\s+choice(?:\s+based)?',
    ]
    
    is_explicit = any(re.search(pattern, response_lower) for pattern in explicit_patterns)
    
    # Find position of answer in text
    answer_position = response_lower.find(extracted.lower())
    if answer_position == -1:
        # Try to find it in a different way
        answer_position = len(generated) // 2  # Assume middle if not found
    
    response_length = len(generated)
    relative_position = answer_position / response_length if response_length > 0 else 1.0
    
    # Base score on position (earlier = better)
    if relative_position <= 0.1:
        position_score = 1.0
    elif relative_position <= 0.25:
        position_score = 0.9
    elif relative_position <= 0.5:
        position_score = 0.7
    else:
        position_score = 0.5
    
    # Bonus for explicit statement
    if is_explicit:
        position_score = min(position_score + 0.1, 1.0)
    
    # Check for contradictions (multiple different answers mentioned)
    if choices:
        choices_lower = [c.lower().strip() for c in choices]
        mentioned_choices = []
        for choice in choices_lower:
            pattern = r'\b' + re.escape(choice) + r'\b'
            if re.search(pattern, response_lower):
                mentioned_choices.append(choice)
        
        # If multiple different choices mentioned, reduce score
        if len(mentioned_choices) > 2:
            position_score *= 0.7
    
    return position_score


# Metric registry for Commonsense Reasoning Multiple Choice task
COMMONSENSE_REASONING_METRICS = {
    'answer_correctness': answer_correctness,
    'follows_format_instruction': follows_format_instruction,
    'answer_extractability': answer_extractability
}
