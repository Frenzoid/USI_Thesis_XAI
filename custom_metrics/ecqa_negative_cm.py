"""
Custom metrics for Negative Properties Generation task.

Task: Write negative properties explaining why each incorrect answer choice is wrong.

Expected output: Multiple simple, atomic sentences where:
- Each sentence explains why ONE incorrect choice is wrong
- Sentences are simple and atomic (one fact that cannot be broken down)
- Uses external commonsense knowledge, not question restatements
- Covers ALL incorrect choices (not the correct answer)

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


def incorrect_choices_coverage(response_data):
    """
    Check if the response addresses ALL incorrect choices and ONLY incorrect choices.
    
    The task requires explaining why each incorrect option is wrong, which means:
    - All incorrect choices should be mentioned
    - The correct answer should NOT be extensively discussed (it's not incorrect)
    
    Scoring:
    - Full credit for covering all incorrect choices
    - Penalty if correct answer is discussed extensively
    - Proportional score for partial coverage
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Coverage score (0.0 to 1.0)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').lower()
    question_values = response_data.get('question_values', [])
    
    # Extract choices from question_values
    # Format: [question, choice1, choice2, ..., choiceN, correct_answer]
    if len(question_values) < 3:
        return 0.0
    
    # All choices except the last one (which is correct answer)
    all_choices = question_values[1:-1]
    correct_answer = question_values[-1].lower().strip()
    
    # Incorrect choices are all choices except the correct one
    incorrect_choices = [c for c in all_choices if c.lower().strip() != correct_answer]
    
    if not incorrect_choices:
        return 0.0
    
    # Count how many incorrect choices are mentioned
    mentioned_incorrect = 0
    for choice in incorrect_choices:
        choice_lower = choice.lower().strip()
        # Check if the choice appears in the explanation
        pattern = r'\b' + re.escape(choice_lower) + r'\b'
        if re.search(pattern, generated):
            mentioned_incorrect += 1
    
    # Calculate coverage of incorrect choices
    coverage = mentioned_incorrect / len(incorrect_choices)
    
    # Check if correct answer is mentioned (penalty)
    correct_pattern = r'\b' + re.escape(correct_answer) + r'\b'
    correct_mentions = len(re.findall(correct_pattern, generated))
    
    # Small penalty if correct answer is mentioned multiple times
    # (once or twice is okay for context, but more suggests focusing on it)
    if correct_mentions > 2:
        penalty = min((correct_mentions - 2) * 0.1, 0.3)
        coverage = max(0.0, coverage - penalty)
    
    return coverage


def atomic_sentence_format(response_data):
    """
    Check if properties are written as simple, atomic sentences.
    
    Atomic sentences should:
    - Contain one fact that cannot be broken down further
    - Be simple (no complex clauses joined by "and", "but", "while", etc.)
    - Be clear and direct
    
    This metric penalizes:
    - Complex sentences with multiple clauses
    - Sentences connected by conjunctions (and, but, while, whereas, etc.)
    - Very long sentences (>20 words typically indicates complexity)
    - Run-on sentences
    
    Scoring based on:
    - Average sentence length (shorter is better)
    - Presence of coordinating/subordinating conjunctions
    - Sentence complexity indicators
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Atomicity score (1.0 = all simple atomic sentences)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    if not generated:
        return 0.0
    
    # Split into sentences
    # Use periods, exclamation marks, question marks as delimiters
    sentences = re.split(r'[.!?]+', generated)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    score = 0.0
    total_sentences = len(sentences)
    
    # Conjunctions that indicate non-atomic sentences
    complex_conjunctions = [
        r'\band\b', r'\bbut\b', r'\bwhile\b', r'\bwhereas\b',
        r'\balthough\b', r'\bthough\b', r'\bhowever\b',
        r'\bsince\b', r'\bbecause\b.*\band\b',  # Multiple clauses with because...and
        r'\bif\b.*\bthen\b', r'\beither\b.*\bor\b'
    ]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        word_count = len(sentence.split())
        
        # Check sentence length (ideal: 5-20 words)
        if word_count <= 20:
            length_score = 1.0
        elif word_count <= 30:
            length_score = 0.7
        else:
            length_score = 0.4
        
        # Check for complex conjunctions
        conjunction_penalty = 0.0
        for conj_pattern in complex_conjunctions:
            if re.search(conj_pattern, sentence_lower):
                conjunction_penalty += 0.3
        
        conjunction_penalty = min(conjunction_penalty, 0.6)
        
        # Calculate sentence score
        sentence_score = length_score - conjunction_penalty
        sentence_score = max(0.0, sentence_score)
        
        score += sentence_score
    
    # Average score across all sentences
    return score / total_sentences


def external_knowledge_usage(response_data):
    """
    Measure if the response uses external commonsense knowledge.
    
    Good responses should:
    - Provide facts NOT stated in the question
    - Use general world knowledge
    - Explain WHY choices are incorrect with external reasoning
    
    Bad responses:
    - Just restate words from the question
    - Repeat the question phrasing
    - Say "X is not Y" without explaining why
    
    This metric:
    - Penalizes high overlap with question text (but more tolerance than before)
    - Rewards explanatory language
    - Checks for meaningful content beyond question restatement
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: External knowledge score (0.0 to 1.0)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').lower()
    question_values = response_data.get('question_values', [])
    
    if not generated or not question_values:
        return 0.0
    
    # Get the question text
    question_text = question_values[0].lower() if question_values else ""
    
    score = 1.0
    
    # 1. Calculate overlap with question (penalty for high overlap, but more tolerant)
    if question_text:
        # Extract significant words from question (5+ characters)
        question_words = set(word.strip('.,!?;:') for word in question_text.split() 
                           if len(word) >= 5)
        
        # Extract words from generated response
        generated_words = [word.strip('.,!?;:') for word in generated.split()]
        
        if question_words and generated_words:
            # Count overlapping words
            overlap_count = sum(1 for word in generated_words if word in question_words)
            overlap_ratio = overlap_count / len(generated_words)
            
            # More lenient: penalize only if more than 50% overlap (was 30%)
            if overlap_ratio > 0.5:
                penalty = min((overlap_ratio - 0.5) * 1.0, 0.3)
                score -= penalty
    
    # 2. Check for explanatory/knowledge-indicating language (bonus)
    knowledge_indicators = [
        r'is a\b', r'are\b', r'means\b', r'refers to\b',
        r'typically\b', r'usually\b', r'generally\b', r'often\b',
        r'is defined as\b', r'known for\b', r'characterized by\b',
        r'involves\b', r'requires\b', r'consists of\b'
    ]
    
    knowledge_count = sum(1 for pattern in knowledge_indicators 
                         if re.search(pattern, generated))
    
    knowledge_bonus = min(knowledge_count * 0.08, 0.3)
    score += knowledge_bonus
    
    # 3. Check for shallow negation patterns (penalty)
    # Patterns like "X is not Y" without explanation
    shallow_patterns = [
        r'\bis not\b.*\.',  # "is not X." - just negation, no explanation
        r'\bcannot be\b.*\.',
        r'\bdoes not\b.*\.',
        r'\bis incorrect\b.*\.',
    ]
    
    shallow_count = 0
    sentences = re.split(r'[.!?]+', generated)
    
    for sentence in sentences:
        sentence = sentence.strip().lower()
        # Check if sentence is ONLY a shallow negation (no explanation)
        for pattern in shallow_patterns:
            if re.search(pattern, sentence):
                # Check if sentence is very short (< 10 words) - likely shallow
                if len(sentence.split()) < 10:
                    shallow_count += 1
                    break
    
    if shallow_count > 0:
        total_sentences = len([s for s in sentences if s.strip()])
        if total_sentences > 0:
            shallow_ratio = shallow_count / total_sentences
            # Penalize if more than 30% of sentences are shallow
            if shallow_ratio > 0.3:
                penalty = min((shallow_ratio - 0.3) * 0.8, 0.3)
                score -= penalty
    
    # 4. Check for definitions and facts (bonus)
    definition_patterns = [
        r'\b\w+\s+is\s+a\s+\w+',  # "X is a Y"
        r'\b\w+\s+are\s+\w+',     # "X are Y"
        r'\b\w+\s+means\s+',      # "X means"
        r'\b\w+\s+refers\s+to\s+',# "X refers to"
    ]
    
    definition_count = sum(1 for pattern in definition_patterns 
                          if re.search(pattern, generated))
    
    if definition_count >= 2:
        score += 0.1
    
    # Ensure score stays in [0, 1] range
    return max(0.0, min(score, 1.0))


# Metric registry for Negative Properties Generation task
NEGATIVE_PROPERTIES_METRICS = {
    'incorrect_choices_coverage': incorrect_choices_coverage,
    'atomic_sentence_format': atomic_sentence_format,
    'external_knowledge_usage': external_knowledge_usage
}
