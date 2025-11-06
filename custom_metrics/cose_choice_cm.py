"""
Custom metrics for Commonsense Reasoning Multiple-Choice task.

Task: Given a question and multiple choices, select the single best choice.
Instruction: "Select the single best choice and return only that choice."

Expected output: A simple string containing only the chosen answer (e.g., "library", "work in school")

Key problems observed:
1. Looping/repetition failures (models get stuck repeating phrases)
2. Instruction non-compliance (adding explanations, conversational fluff)
3. Answer extraction difficulty (correct answer buried in verbose text)

Each metric function receives a response dictionary with the following fields:
- prompt: The input prompt sent to the model
- response: The model's generated response  
- expected_output: The expected/reference response
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
    Uses case-insensitive, whitespace-normalized matching to handle variations.
    
    Extraction strategy:
    1. Look for the answer in the first 200 characters (models often put it early)
    2. Try to find expected answer as a substring
    3. Look for common patterns like "Answer: X", "The best answer is X"
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    expected = response_data.get('expected_output', '').strip()
    
    if not generated or not expected:
        return 0.0
    
    # Normalize for comparison
    generated_lower = generated.lower()
    expected_lower = expected.lower()
    
    # Strategy 1: Check if response exactly matches expected (after normalization)
    if generated_lower == expected_lower:
        return 1.0
    
    # Strategy 2: Check if expected answer appears in first 200 chars
    first_part = generated_lower[:200]
    if expected_lower in first_part:
        return 1.0
    
    # Strategy 3: Check if expected answer appears anywhere in response
    if expected_lower in generated_lower:
        return 1.0
    
    # Strategy 4: Try to extract answer from common patterns
    patterns = [
        r'answer[:\s]+[\'"]?([^\'"\n]+)[\'"]?',
        r'best\s+(?:answer|choice)\s+is[:\s]+[\'"]?([^\'"\n]+)[\'"]?',
        r'the\s+answer\s+should\s+be[:\s]+[\'"]?([^\'"\n]+)[\'"]?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, generated_lower, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if expected_lower in extracted or extracted in expected_lower:
                return 1.0
    
    return 0.0


def response_conciseness(response_data):
    """
    Measure adherence to "return only that choice" instruction.
    
    The instruction explicitly says to return ONLY the choice, not explanations,
    not conversational additions.
    
    Scoring based on response length:
    - 1.0: Very concise (≤20 chars) - just the answer
    - 0.8: Minimal wrapper (≤50 chars) - e.g., "The best answer is library."
    - 0.5: Some explanation (≤200 chars)
    - 0.2: Verbose explanation (≤500 chars)
    - 0.0: Extreme verbosity or looping (>500 chars)
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Conciseness score (1.0 = perfect conciseness)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    if not generated:
        return 0.0
    
    response_length = len(generated)
    
    # Perfect: just the answer (very short)
    if response_length <= 20:
        return 1.0
    
    # Good: minimal wrapper text
    if response_length <= 50:
        return 0.8
    
    # Acceptable: brief explanation
    if response_length <= 200:
        return 0.5
    
    # Poor: verbose explanation
    if response_length <= 500:
        return 0.2
    
    # Very poor: extreme verbosity or looping
    return 0.0


def instruction_following_penalty(response_data):
    """
    Penalize specific instruction violations beyond just length.
    
    The instruction is clear: "return only that choice"
    
    This metric checks for:
    1. Conversational additions ("Let me know if...", "Have a great day!")
    2. Multiple questions or answers in the response
    3. Excessive explanations or reasoning
    4. Repetitive looping patterns
    
    Scoring:
    - 1.0: Clean answer with no violations
    - Penalties for each violation type:
      - -0.3: Conversational fluff
      - -0.2: Multiple questions/answers
      - -0.2: Excessive explanation markers
      - -0.5: Repetitive looping detected
    - Minimum score: 0.0
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Penalty score (1.0 = no violations, 0.0 = severe violations)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    if not generated:
        return 0.0
    
    score = 1.0
    generated_lower = generated.lower()
    
    # Check 1: Conversational additions
    conversational_phrases = [
        'let me know',
        'have a great day',
        'feel free',
        'i can help',
        'if you need',
        'just let me know',
        'best regards',
        'thank you',
    ]
    
    if any(phrase in generated_lower for phrase in conversational_phrases):
        score -= 0.3
    
    # Check 2: Multiple questions (indicates model generated unrelated content)
    question_count = generated.count('Question:')
    if question_count > 1:
        score -= 0.2
    
    # Check 3: Excessive explanation markers
    explanation_markers = [
        'reasoning',
        'because',
        'explanation',
        'let me explain',
        'the reason',
        'this is because',
    ]
    
    explanation_count = sum(1 for marker in explanation_markers if marker in generated_lower)
    if explanation_count >= 2:
        score -= 0.2
    
    # Check 4: Repetitive looping patterns
    # If same phrase appears 3+ times, likely a loop
    lines = generated.split('\n')
    if len(lines) > 10:
        # Check if many lines are identical or very similar
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(lines) > 20 and len(unique_lines) < len(lines) / 3:
            score -= 0.5  # Likely looping
    
    # Also check for repeated phrases within text
    # Split into sentences and check for repetition
    sentences = re.split(r'[.!?]+', generated)
    if len(sentences) > 10:
        sentence_counts = {}
        for sentence in sentences:
            normalized = sentence.strip().lower()
            if len(normalized) > 10:  # Only count substantial sentences
                sentence_counts[normalized] = sentence_counts.get(normalized, 0) + 1
        
        # If any sentence appears 3+ times, it's repetitive
        if any(count >= 3 for count in sentence_counts.values()):
            score -= 0.5
    
    return max(0.0, score)


# Metric registry for Commonsense Reasoning Multiple-Choice task
COMMONSENSE_REASONING_METRICS = {
    'answer_correctness': answer_correctness,
    'response_conciseness': response_conciseness,
    'instruction_following_penalty': instruction_following_penalty
}
