"""
Custom metrics for Commonsense Reasoning with Explanation task.

Task: Given a question and multiple choices, briefly explain in one sentence which choice
is the correct answer.

Instruction: "Briefly explain in one sentence which choice is the correct answer to this question"

Expected output: A brief explanation identifying and justifying the correct answer
(e.g., "Informational pamphlets can contain lots of types of information... available at a library.")

Key differences from "answer only" variant:
- Explanations are REQUIRED (not penalized)
- Should identify the correct choice AND explain why
- Should be brief (ideally one sentence, though 2-3 is often acceptable)
- Still shouldn't loop or generate unrelated content

Each metric function receives a response dictionary with the following fields:
- prompt: The input prompt sent to the model
- response: The model's generated response  
- expected_output: The expected explanation (contains the correct answer)
- question_values: List containing [question_text, choices_list]
- success: Boolean indicating if the response generation was successful
- error: Error message if success is False, None otherwise

Each function should return a float value between 0.0 and 1.0.
"""

import re
import ast


def answer_identification_correctness(response_data):
    """
    Check if the response identifies the correct choice.
    
    This extracts which answer choice the model is explaining and checks if it
    matches the correct answer (which can be inferred from expected_output or
    by checking which choice appears in the explanation).
    
    Strategy:
    1. Parse the choices from question_values
    2. Look for common patterns like "correct answer is X", "Answer: X"
    3. Check which choice is mentioned prominently in the response
    4. See if that choice also appears in expected_output
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: 1.0 if correct choice identified, 0.0 otherwise
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    expected = response_data.get('expected_output', '').strip()
    question_values = response_data.get('question_values', [])
    
    if not generated or not expected or len(question_values) < 2:
        return 0.0
    
    # Parse choices from question_values
    # Format is usually: ['choice1', 'choice2', ...]
    try:
        choices_str = question_values[1]
        # Try to parse as Python list
        choices = ast.literal_eval(choices_str)
        if not isinstance(choices, list):
            return 0.0
    except:
        # Fallback: try to extract choices with regex
        choices_match = re.findall(r"'([^']+)'", question_values[1])
        if not choices_match:
            return 0.0
        choices = choices_match
    
    generated_lower = generated.lower()
    expected_lower = expected.lower()
    
    # Strategy 1: Look for explicit answer declarations
    answer_patterns = [
        r'correct\s+(?:answer|choice)\s+is\s+["\']?([^"\',.!\n]+)["\']?',
        r'answer:\s*["\']?([^"\',.!\n]+)["\']?',
        r'best\s+answer\s+is\s+["\']?([^"\',.!\n]+)["\']?',
        r'the\s+answer\s+is\s+["\']?([^"\',.!\n]+)["\']?',
    ]
    
    identified_answer = None
    for pattern in answer_patterns:
        match = re.search(pattern, generated_lower, re.IGNORECASE)
        if match:
            identified_answer = match.group(1).strip()
            break
    
    # Strategy 2: If no explicit pattern, find which choice is mentioned most prominently
    # Check which choice appears in the first 200 characters
    if not identified_answer:
        first_part = generated_lower[:200]
        for choice in choices:
            if choice.lower() in first_part:
                identified_answer = choice.lower()
                break
    
    # Strategy 3: Check all choices and see which one appears
    if not identified_answer:
        for choice in choices:
            if choice.lower() in generated_lower:
                identified_answer = choice.lower()
                break
    
    if not identified_answer:
        return 0.0
    
    # Now check if this answer also appears in expected output
    # The expected output usually mentions or implies the correct answer
    for choice in choices:
        choice_lower = choice.lower()
        if choice_lower in expected_lower and choice_lower in identified_answer:
            return 1.0
        if identified_answer in choice_lower and choice_lower in expected_lower:
            return 1.0
    
    # If we can't verify from expected, at least check if identified answer is a valid choice
    for choice in choices:
        if choice.lower() in identified_answer or identified_answer in choice.lower():
            return 0.5  # Partial credit: valid choice mentioned, but can't verify correctness
    
    return 0.0


def explanation_quality(response_data):
    """
    Assess the quality of the explanation.
    
    A good explanation should:
    1. Be reasonably brief (instruction says "one sentence" but 2-3 sentences is acceptable)
    2. Contain reasoning (words like "because", "since", "as", etc.)
    3. Be relevant to the question
    4. Not be excessively verbose
    
    Scoring:
    - 1.0: Clear, concise explanation with reasoning (50-300 chars)
    - 0.8: Good explanation, slightly longer (300-500 chars)
    - 0.6: Acceptable but verbose (500-800 chars)
    - 0.3: Very verbose but has some explanation (800-1500 chars)
    - 0.0: Extremely verbose or no reasoning (>1500 chars or no reasoning words)
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Quality score (1.0 = excellent, 0.0 = poor)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    if not generated:
        return 0.0
    
    generated_lower = generated.lower()
    response_length = len(generated)
    
    # Check for reasoning words (indicates explanation, not just answer)
    reasoning_words = [
        'because', 'since', 'as', 'due to', 'reason', 'why',
        'therefore', 'thus', 'so', 'which', 'that is why',
        'this is because', 'the reason', 'can be', 'would be',
        'typically', 'often', 'usually', 'commonly', 'likely'
    ]
    
    has_reasoning = any(word in generated_lower for word in reasoning_words)
    
    # If no reasoning words at all, it's probably not a good explanation
    if not has_reasoning:
        return 0.0
    
    # Score based on length (assuming it has reasoning)
    if 50 <= response_length <= 300:
        return 1.0  # Perfect: concise with reasoning
    elif 300 < response_length <= 500:
        return 0.8  # Good: clear but slightly longer
    elif 500 < response_length <= 800:
        return 0.6  # Acceptable but verbose
    elif 800 < response_length <= 1500:
        return 0.3  # Very verbose
    else:
        return 0.0  # Extremely verbose or too short


def response_cleanliness(response_data):
    """
    Penalize looping, repetition, and unrelated content generation.
    
    The response should be a clean explanation without:
    1. Repetitive looping (same text repeated many times)
    2. Multiple unrelated questions/answers
    3. Excessive conversational fluff (though some is acceptable in explanations)
    
    Scoring starts at 1.0 with penalties for violations:
    - -0.5: Repetitive looping detected
    - -0.3: Multiple unrelated questions generated
    - -0.2: Excessive conversational fluff
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Cleanliness score (1.0 = clean, 0.0 = very messy)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    if not generated:
        return 0.0
    
    score = 1.0
    generated_lower = generated.lower()
    
    # Check 1: Repetitive looping
    # Split into sentences and check for repetition
    sentences = re.split(r'[.!?]+', generated)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if len(sentences) > 5:
        # Count identical sentences
        sentence_counts = {}
        for sentence in sentences:
            normalized = sentence.lower().strip()
            sentence_counts[normalized] = sentence_counts.get(normalized, 0) + 1
        
        # If any sentence appears 3+ times, it's looping
        max_repeats = max(sentence_counts.values()) if sentence_counts else 0
        if max_repeats >= 3:
            score -= 0.5
        elif max_repeats >= 2 and len(sentences) > 10:
            score -= 0.3
    
    # Also check for repeated phrases in lines
    lines = generated.split('\n')
    if len(lines) > 10:
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(lines) > 15 and len(unique_lines) < len(lines) / 3:
            score -= 0.5  # Likely looping
    
    # Check 2: Multiple unrelated questions
    # Count how many times "Question:" appears
    question_count = generated.count('Question:')
    if question_count > 2:
        score -= 0.3
    elif question_count > 1:
        score -= 0.15
    
    # Check 3: Excessive conversational fluff
    # Some conversational language is OK in explanations, but too much is bad
    fluff_phrases = [
        'let me know if you need anything else',
        'please let me know',
        'have a great day',
        'feel free to',
        'i can help with',
        'just let me know',
    ]
    
    fluff_count = sum(1 for phrase in fluff_phrases if phrase in generated_lower)
    if fluff_count >= 3:
        score -= 0.2
    elif fluff_count >= 2:
        score -= 0.1
    
    return max(0.0, score)


# Metric registry for Commonsense Reasoning with Explanation task
COMMONSENSE_REASONING_EXPLANATION_METRICS = {
    'answer_identification_correctness': answer_identification_correctness,
    'explanation_quality': explanation_quality,
    'response_cleanliness': response_cleanliness
}
