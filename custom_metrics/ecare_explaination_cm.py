"""
Custom metrics for Conceptual Explanation task.

Task: Given a cause-effect relationship, provide a BRIEF conceptual explanation.
Expected output: A concise, general conceptual statement (typically 1 sentence, 10-20 words).

Each metric function receives a response dictionary with the following fields:
- prompt: The input prompt sent to the model
- response: The model's generated response  
- expected_output: The expected/reference response
- question_values: List of question field values used to populate the prompt
- success: Boolean indicating if the response generation was successful
- error: Error message if success is False, None otherwise

Each function should return a float value between 0.0 and 1.0.
"""


def response_brevity(response_data):
    """
    Measure if the response is appropriately brief.
    
    The prompt asks for a "brief conceptual explanation". Expected outputs
    are typically 10-20 words (one sentence). This metric heavily penalizes
    verbose, multi-paragraph explanations.
    
    Scoring based on word count:
    - 1.0: 5-25 words (ideal brevity)
    - 0.8: 26-50 words (acceptable but slightly verbose)
    - 0.5: 51-100 words (too detailed)
    - 0.3: 101-200 words (verbose)
    - 0.1: 201-400 words (very verbose)
    - 0.0: >400 words (extremely verbose)
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Brevity score (1.0 = ideal length, lower = too verbose)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    expected = response_data.get('expected_output', '').strip()
    
    # Count words in the generated response
    word_count = len(generated.split())
    
    # Count words in expected output to get target range
    expected_word_count = len(expected.split())
    
    # Score based on word count
    if 5 <= word_count <= 25:
        return 1.0
    elif 26 <= word_count <= 50:
        return 0.8
    elif 51 <= word_count <= 100:
        return 0.5
    elif 101 <= word_count <= 200:
        return 0.3
    elif 201 <= word_count <= 400:
        return 0.1
    else:
        return 0.0


def conceptual_abstraction_level(response_data):
    """
    Measure if the response provides a general conceptual statement vs
    specific scenario explanation.
    
    Expected outputs are high-level generalizations (e.g., "Nits are a 
    measure of brightness", "Gender is a sense of self") rather than 
    detailed explanations of the specific scenario.
    
    This metric penalizes:
    - Numbered lists or step-by-step reasoning
    - References to the specific scenario/characters (e.g., "Tom", "the fisherman")
    - Detailed explanations with multiple points
    - References and citations
    
    This metric rewards:
    - General statements about concepts
    - Definitions or generalizations
    - Simple declarative sentences
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Abstraction score (1.0 = general concept, 0.0 = specific scenario)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    generated_lower = generated.lower()
    
    # Indicators of over-specific/detailed explanation (penalize these)
    specificity_indicators = [
        # Numbered lists or structured explanations
        (r'^\d+\.', 0.3),  # Starts with "1.", "2.", etc.
        (r'\n\s*\d+\.', 0.3),  # Contains numbered points
        (r'^\*\*', 0.2),  # Markdown headers
        (r'here\'s|here is', 0.2),
        
        # Scenario-specific references
        (r'\bhe\b|\bshe\b|\bhis\b|\bher\b', 0.15),
        (r'\btom\b|\bmario\b|\bfisherman\b|\bsoldier\b', 0.25),
        (r'\bthis case\b|\bthis scenario\b|\bthis situation\b', 0.2),
        
        # Detailed reasoning markers
        (r'first|second|third|finally', 0.2),
        (r'step\s*\d+|stage\s*\d+', 0.2),
        
        # Academic/citation markers
        (r'\(\d{4}\)', 0.3),  # Year citations like (2015)
        (r'references:|works cited:', 0.4),
        (r'et al\.', 0.3),
    ]
    
    penalty = 0.0
    for pattern, penalty_value in specificity_indicators:
        import re
        if re.search(pattern, generated_lower):
            penalty += penalty_value
    
    # Cap penalty at 1.0
    penalty = min(penalty, 1.0)
    
    # Check for general conceptual language (reward these)
    conceptual_indicators = [
        r'\bare\b.*\ba\b',  # "X are a Y" pattern
        r'\bis\b.*\ba\b',   # "X is a Y" pattern
        r'is defined as',
        r'refers to',
        r'means that',
        r'represents',
    ]
    
    bonus = 0.0
    import re
    for pattern in conceptual_indicators:
        if re.search(pattern, generated_lower):
            bonus = 0.2
            break
    
    # Base score starts at 1.0, subtract penalties, add bonus
    score = 1.0 - penalty + bonus
    
    return max(0.0, min(score, 1.0))


def single_sentence_format(response_data):
    """
    Check if the response is a single sentence or very few sentences.
    
    Expected outputs are typically one concise sentence. This metric
    penalizes responses with multiple sentences, paragraphs, or
    complex structure.
    
    Scoring:
    - 1.0: 1 sentence
    - 0.8: 2 sentences
    - 0.6: 3 sentences
    - 0.4: 4-5 sentences
    - 0.2: 6-10 sentences
    - 0.0: >10 sentences
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Format score (1.0 = single sentence, lower = multiple sentences)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    # Count sentences (simple heuristic: count periods, exclamation marks, question marks)
    # More sophisticated: exclude periods in abbreviations
    import re
    
    # Remove common abbreviations that shouldn't count as sentence endings
    text = generated
    text = re.sub(r'\be\.g\.|\bi\.e\.|\bvs\.|\bdr\.|\bmr\.|\bms\.|\betc\.',
                  '', text, flags=re.IGNORECASE)
    
    # Count sentence-ending punctuation
    sentence_endings = re.findall(r'[.!?]+', text)
    sentence_count = len(sentence_endings)
    
    # Also check for paragraph breaks (multiple newlines)
    paragraph_breaks = len(re.findall(r'\n\s*\n', generated))
    
    # Penalize paragraph breaks heavily
    if paragraph_breaks > 0:
        sentence_count += paragraph_breaks * 3
    
    # Score based on sentence count
    if sentence_count <= 1:
        return 1.0
    elif sentence_count == 2:
        return 0.8
    elif sentence_count == 3:
        return 0.6
    elif sentence_count <= 5:
        return 0.4
    elif sentence_count <= 10:
        return 0.2
    else:
        return 0.0


# Metric registry for Conceptual Explanation task
CONCEPTUAL_EXPLANATION_METRICS = {
    'response_brevity': response_brevity,
    'conceptual_abstraction_level': conceptual_abstraction_level,
    'single_sentence_format': single_sentence_format
}
