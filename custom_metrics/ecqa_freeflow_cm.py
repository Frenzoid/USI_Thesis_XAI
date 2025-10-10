"""
Custom metrics for Commonsense Reasoning Explanation Generation task.

Task: Generate a comprehensive explanation paragraph that justifies the correct answer
and refutes incorrect choices using commonsense reasoning.

Expected output: ONE well-structured paragraph that:
- Justifies why the correct answer is right
- Refutes why each incorrect choice is wrong
- Uses external commonsense knowledge
- Is minimal (no redundant information)

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


def all_choices_addressed(response_data):
    """
    Check if the explanation addresses all answer choices.
    
    A comprehensive explanation should:
    - Justify why the correct answer is right
    - Refute why EACH incorrect choice is wrong
    
    This metric counts how many of the choices are mentioned/discussed
    in the explanation.
    
    Scoring:
    - 1.0: All choices mentioned (100%)
    - Proportional score for partial coverage
    - Minimum 0.0 if no choices or only correct answer mentioned
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Proportion of choices addressed (0.0 to 1.0)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').lower()
    question_values = response_data.get('question_values', [])
    
    # Extract choices from question_values
    # Format: [question, choice1, choice2, ..., choiceN, correct_answer]
    if len(question_values) < 3:
        return 0.0
    
    # Last item is the correct answer, items 1 to -1 are the choices
    choices = question_values[1:-1]
    
    if not choices:
        return 0.0
    
    # Count how many choices are mentioned in the explanation
    mentioned_count = 0
    for choice in choices:
        choice_lower = choice.lower().strip()
        
        # Check if the choice appears in the explanation
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(choice_lower) + r'\b'
        if re.search(pattern, generated):
            mentioned_count += 1
        else:
            # Also check for quoted versions
            quoted_pattern = r'["\']' + re.escape(choice_lower) + r'["\']'
            if re.search(quoted_pattern, generated):
                mentioned_count += 1
    
    # Calculate proportion of choices addressed
    coverage = mentioned_count / len(choices)
    
    return coverage


def single_paragraph_format(response_data):
    """
    Check if the explanation is written as a single paragraph.
    
    The instruction explicitly asks for "one well-structured paragraph".
    This metric penalizes responses with:
    - Multiple paragraphs (separated by blank lines)
    - Bullet points or numbered lists
    - Excessive line breaks
    
    Scoring:
    - 1.0: Single continuous paragraph
    - 0.7: 2 paragraphs
    - 0.4: 3 paragraphs
    - 0.2: 4+ paragraphs
    - 0.0: Structured lists or heavy formatting
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Format compliance score (1.0 = single paragraph)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    if not generated:
        return 0.0
    
    # Check for bullet points or numbered lists (major violation)
    list_patterns = [
        r'^\s*[-•*]\s+',  # Bullet points
        r'^\s*\d+[\.)]\s+',  # Numbered lists
        r'\n\s*[-•*]\s+',  # Bullet points mid-text
        r'\n\s*\d+[\.)]\s+',  # Numbered lists mid-text
    ]
    
    for pattern in list_patterns:
        if re.search(pattern, generated, re.MULTILINE):
            return 0.0
    
    # Count paragraph breaks (double newlines or single newlines with significant spacing)
    # First normalize: replace multiple spaces/tabs with single space
    normalized = re.sub(r'[ \t]+', ' ', generated)
    
    # Count paragraph breaks (look for double newlines)
    paragraph_breaks = len(re.findall(r'\n\s*\n', normalized))
    
    # Also check for single newlines that aren't just line wrapping
    # (i.e., lines that start a new sentence)
    single_newlines = re.findall(r'\n', normalized)
    significant_breaks = 0
    
    for i, line in enumerate(normalized.split('\n')):
        line_stripped = line.strip()
        # Check if line starts with capital letter (new sentence)
        if i > 0 and line_stripped and line_stripped[0].isupper():
            significant_breaks += 1
    
    total_breaks = paragraph_breaks + (significant_breaks // 2)  # Divide by 2 to not double-count
    
    # Score based on number of paragraph breaks
    if total_breaks == 0:
        return 1.0
    elif total_breaks == 1:
        return 0.7
    elif total_breaks == 2:
        return 0.4
    else:
        return 0.2


def explanation_depth(response_data):
    """
    Measure the depth and quality of reasoning in the explanation.
    
    High-quality explanations should:
    - Provide actual reasoning (not just restate facts)
    - Use external commonsense knowledge
    - Explain WHY choices are correct/incorrect
    - Avoid redundant information from the question
    
    This metric looks for:
    + Reasoning indicators: "because", "as", "since", "therefore", "this is why"
    + Causal language: "leads to", "results in", "causes", "allows"
    + Comparative language: "unlike", "whereas", "while", "in contrast"
    + Knowledge indicators: "typically", "generally", "commonly", "universally"
    - Excessive direct quotes from question
    - Simple restatements without explanation
    
    Scoring based on presence of reasoning elements and absence of
    redundancy/shallow content.
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Depth score from 0.0 to 1.0
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').lower()
    question_values = response_data.get('question_values', [])
    
    if not generated:
        return 0.0
    
    # Get the question text to check for redundancy
    question_text = question_values[0].lower() if question_values else ""
    
    score = 0.0
    
    # 1. Check for reasoning indicators (max 0.35)
    reasoning_indicators = [
        'because', 'as', 'since', 'therefore', 'thus', 'hence',
        'this is why', 'the reason', 'due to', 'given that',
        'consequently', 'as a result'
    ]
    reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in generated)
    reasoning_score = min(reasoning_count * 0.1, 0.35)
    score += reasoning_score
    
    # 2. Check for causal/explanatory language (max 0.25)
    causal_language = [
        'leads to', 'results in', 'causes', 'allows', 'enables',
        'makes', 'provides', 'demonstrates', 'shows', 'indicates',
        'suggests', 'implies', 'means that'
    ]
    causal_count = sum(1 for phrase in causal_language if phrase in generated)
    causal_score = min(causal_count * 0.08, 0.25)
    score += causal_score
    
    # 3. Check for comparative/contrastive reasoning (max 0.2)
    comparative_language = [
        'unlike', 'whereas', 'while', 'in contrast', 'however',
        'on the other hand', 'conversely', 'instead', 'rather than',
        'compared to', 'different from', 'similar to'
    ]
    comparative_count = sum(1 for phrase in comparative_language if phrase in generated)
    comparative_score = min(comparative_count * 0.1, 0.2)
    score += comparative_score
    
    # 4. Check for external knowledge indicators (max 0.2)
    knowledge_indicators = [
        'typically', 'generally', 'commonly', 'usually', 'often',
        'universally', 'naturally', 'inherently', 'fundamentally',
        'by definition', 'in practice', 'historically', 'traditionally'
    ]
    knowledge_count = sum(1 for indicator in knowledge_indicators if indicator in generated)
    knowledge_score = min(knowledge_count * 0.1, 0.2)
    score += knowledge_score
    
    # 5. Penalty for excessive redundancy with question (subtract up to 0.3)
    if question_text:
        # Extract significant words from question (5+ characters)
        question_words = set(word for word in question_text.split() if len(word) >= 5)
        
        if question_words:
            # Count how many question words appear in explanation
            explanation_words = generated.split()
            redundant_words = sum(1 for word in explanation_words if word in question_words)
            redundancy_ratio = redundant_words / len(explanation_words) if explanation_words else 0
            
            # Penalize if more than 20% of explanation is question words
            if redundancy_ratio > 0.2:
                penalty = min((redundancy_ratio - 0.2) * 1.5, 0.3)
                score -= penalty
    
    # 6. Check for shallow content patterns (subtract up to 0.2)
    shallow_patterns = [
        r'is\s+(not\s+)?correct',
        r'is\s+(not\s+)?the\s+answer',
        r'(doesn\'t|does\s+not)\s+make\s+sense',
        r'is\s+(too\s+)?(vague|specific|general)',
    ]
    shallow_count = sum(1 for pattern in shallow_patterns if re.search(pattern, generated))
    if shallow_count > 3:
        score -= 0.2
    
    # Ensure score stays in [0, 1] range
    return max(0.0, min(score, 1.0))


# Metric registry for Commonsense Reasoning Explanation Generation task
EXPLANATION_GENERATION_METRICS = {
    'all_choices_addressed': all_choices_addressed,
    'single_paragraph_format': single_paragraph_format,
    'explanation_depth': explanation_depth
}