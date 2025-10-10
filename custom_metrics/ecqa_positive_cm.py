"""
Custom metrics for Positive Justification Generation task.

Task: Justify why the correct answer choice is correct using simple, atomic sentences.

Expected output: 1-3 simple, atomic sentences where:
- Each sentence contains one fact that cannot be broken down further
- Sentences are simple (no "because" clauses or complex subordination)
- Justification is brief and focused only on the correct answer
- Uses external commonsense knowledge

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


def simple_atomic_sentences(response_data):
    """
    Check if justifications are written as simple, atomic sentences.
    
    Atomic sentences should:
    - Contain one fact that cannot be broken down further
    - Avoid subordinate clauses (because, since, as, when, if, etc.)
    - Avoid coordinating conjunctions in the middle (and, but, so, etc.)
    - Be simple declarative statements
    
    This metric penalizes:
    - "because" clauses and other causal subordination
    - "which/that" relative clauses
    - Multiple facts joined by "and"
    - Complex sentence structures
    
    Scoring based on sentence simplicity and atomicity.
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Simplicity score (1.0 = all simple atomic sentences)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    if not generated:
        return 0.0
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', generated)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    score = 0.0
    
    # Patterns that indicate non-atomic sentences
    complexity_patterns = [
        (r'\bbecause\b', 0.4),        # "X because Y" - causal clause
        (r'\bsince\b', 0.3),          # "X since Y" - causal clause
        (r'\bas\b.*\b(is|are|was|were|provides|makes)\b', 0.3),  # "X as Y" - causal
        (r'\bwhich\b', 0.3),          # "X which Y" - relative clause
        (r'\bthat\b.*\b(is|are|was|were|provides|makes)\b', 0.25), # "X that Y" - relative clause
        (r'\bwhen\b', 0.2),           # "X when Y" - temporal clause
        (r'\bif\b', 0.2),             # "X if Y" - conditional
        (r'\bso\b', 0.2),             # "X so Y" - result clause
        (r',\s*and\b', 0.25),         # "X, and Y" - compound sentence
        (r',\s*but\b', 0.25),         # "X, but Y" - compound sentence
        (r',\s*which\b', 0.3),        # "X, which Y" - non-restrictive clause
        (r'\bmaking\b', 0.2),         # "X, making Y" - participial phrase
        (r'\bfitting\b', 0.2),        # "X, fitting Y" - participial phrase
        (r'\bimplying\b', 0.2),       # "X, implying Y" - participial phrase
    ]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Start with full score for this sentence
        sentence_score = 1.0
        
        # Check for complexity patterns
        for pattern, penalty in complexity_patterns:
            if re.search(pattern, sentence_lower):
                sentence_score -= penalty
        
        # Check sentence length (atomic sentences are typically shorter)
        word_count = len(sentence.split())
        if word_count > 25:
            sentence_score -= 0.3
        elif word_count > 20:
            sentence_score -= 0.2
        elif word_count > 15:
            sentence_score -= 0.1
        
        # Ensure non-negative
        sentence_score = max(0.0, sentence_score)
        score += sentence_score
    
    # Average across all sentences
    return score / len(sentences)


def correct_answer_focus(response_data):
    """
    Check if the justification focuses ONLY on the correct answer.
    
    The task is to justify why the correct answer is right, NOT to:
    - Discuss incorrect choices
    - Compare correct answer to incorrect ones
    - Mention why other options are wrong
    
    This metric:
    - Checks that correct answer is mentioned
    - Penalizes mention of incorrect choices
    - Rewards focused justification
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Focus score (1.0 = only discusses correct answer)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').lower()
    question_values = response_data.get('question_values', [])
    
    if not generated or len(question_values) < 2:
        return 0.0
    
    # Extract correct answer and all choices
    # Format: [question, choice1, choice2, ..., choiceN, correct_answer]
    correct_answer = question_values[-1].lower().strip()
    all_choices = [c.lower().strip() for c in question_values[1:-1]]
    
    # Incorrect choices
    incorrect_choices = [c for c in all_choices if c != correct_answer]
    
    score = 1.0
    
    # Check if correct answer is mentioned (required)
    correct_pattern = r'\b' + re.escape(correct_answer) + r'\b'
    if not re.search(correct_pattern, generated):
        return 0.0  # Must mention correct answer
    
    # Penalize mentions of incorrect choices
    for incorrect_choice in incorrect_choices:
        incorrect_pattern = r'\b' + re.escape(incorrect_choice) + r'\b'
        if re.search(incorrect_pattern, generated):
            score -= 0.25
    
    # Check for comparative language (indicates discussing other options)
    comparative_patterns = [
        r'\bother\s+(?:options|choices)\b',
        r'\bunlike\b',
        r'\bcompared\s+to\b',
        r'\brather\s+than\b',
        r'\binstead\s+of\b',
        r'\bwhile\s+(?:the\s+)?other\b',
    ]
    
    for pattern in comparative_patterns:
        if re.search(pattern, generated):
            score -= 0.15
    
    # Ensure non-negative
    return max(0.0, score)


def concise_justification(response_data):
    """
    Check if the justification is appropriately concise.
    
    The instruction asks for "a simple, atomic sentence" (singular),
    and expected outputs show 1-3 short sentences. This metric:
    - Rewards 1-3 sentences
    - Penalizes verbose explanations
    - Checks that sentences are reasonably short
    
    Scoring:
    - 1.0: 1-3 sentences, each under 20 words
    - Lower scores for more sentences or longer sentences
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Conciseness score (1.0 = appropriately brief)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    if not generated:
        return 0.0
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', generated)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    num_sentences = len(sentences)
    
    # Calculate average sentence length
    word_counts = [len(s.split()) for s in sentences]
    avg_length = sum(word_counts) / len(word_counts)
    
    # Score based on number of sentences
    if num_sentences == 1:
        sentence_count_score = 1.0
    elif num_sentences == 2:
        sentence_count_score = 0.95
    elif num_sentences == 3:
        sentence_count_score = 0.9
    elif num_sentences == 4:
        sentence_count_score = 0.7
    elif num_sentences <= 6:
        sentence_count_score = 0.5
    else:
        sentence_count_score = 0.2
    
    # Score based on average sentence length
    if avg_length <= 15:
        length_score = 1.0
    elif avg_length <= 20:
        length_score = 0.9
    elif avg_length <= 25:
        length_score = 0.7
    elif avg_length <= 30:
        length_score = 0.5
    else:
        length_score = 0.3
    
    # Combined score (weighted average)
    score = (sentence_count_score * 0.6) + (length_score * 0.4)
    
    return score


# Metric registry for Positive Justification Generation task
POSITIVE_JUSTIFICATION_METRICS = {
    'simple_atomic_sentences': simple_atomic_sentences,
    'correct_answer_focus': correct_answer_focus,
    'concise_justification': concise_justification
}