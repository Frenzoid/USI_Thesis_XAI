"""
Custom metrics for Commonsense Reasoning Answer Justification task.

Task: Given a question and a pre-selected answer, explain why that answer is correct.
Instruction: "In one short sentence, explain why the given answer is correct"

Expected output: A brief explanation justifying why the given answer makes sense
(e.g., "Telephone directories can be found in phone booths because people often want to look up numbers before making calls")

Key characteristics:
- The answer is ALREADY PROVIDED (not being selected)
- Task is to JUSTIFY/EXPLAIN the correctness
- Should be brief ("one short sentence" though 2-3 is often acceptable)
- Should contain reasoning (why/because/since)
- Should NOT have excessive formatting, meta-commentary, or unrelated content

Each metric function receives a response dictionary with the following fields:
- prompt: The input prompt sent to the model
- response: The model's generated response  
- expected_output: The expected explanation
- question_values: List containing [question_text, given_answer]
- success: Boolean indicating if the response generation was successful
- error: Error message if success is False, None otherwise

Each function should return a float value between 0.0 and 1.0.
"""

import re


def explanation_correctness(response_data):
    """
    Check if the explanation makes logical sense and explains WHY.
    
    A correct explanation should:
    1. Contain reasoning (because/since/as/etc.)
    2. Reference concepts from the question or answer
    3. Actually explain why the answer is correct (not just restate it)
    
    Since we can't fully verify logical correctness without NLU, we check for:
    - Presence of reasoning words
    - Reasonable length (not just repeating the answer)
    - Presence of key terms from question/answer
    
    Scoring:
    - 1.0: Contains reasoning + relevant concepts + reasonable length
    - 0.7: Contains reasoning but may be weak
    - 0.3: Minimal reasoning or just restating
    - 0.0: No reasoning or empty
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Explanation quality score (1.0 = good explanation)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    question_values = response_data.get('question_values', [])
    
    if not generated:
        return 0.0
    
    generated_lower = generated.lower()
    
    # Check 1: Contains reasoning words
    reasoning_words = [
        'because', 'since', 'as', 'due to', 'reason', 'why',
        'therefore', 'thus', 'so', 'which means', 'indicates',
        'implies', 'suggests', 'shows', 'demonstrates',
        'typically', 'often', 'usually', 'commonly', 'can be',
        'would be', 'is used', 'refers to', 'means that'
    ]
    
    has_reasoning = any(word in generated_lower for word in reasoning_words)
    
    if not has_reasoning:
        return 0.0
    
    # Check 2: Not just repeating the answer
    # If response is very short (< 20 chars), likely just repeating
    if len(generated) < 20:
        return 0.3
    
    # Check 3: References concepts from question or answer
    # Extract key terms from question and answer
    score = 0.7  # Base score for having reasoning
    
    if len(question_values) >= 2:
        question_text = question_values[0].lower()
        answer_text = question_values[1].lower()
        
        # Check if explanation mentions concepts from question or answer
        # (beyond just exact repetition)
        question_words = set(re.findall(r'\b\w{4,}\b', question_text))
        answer_words = set(re.findall(r'\b\w{4,}\b', answer_text))
        explanation_words = set(re.findall(r'\b\w{4,}\b', generated_lower))
        
        # Look for conceptual overlap (not exact matches)
        if explanation_words & question_words or explanation_words & answer_words:
            score = 1.0  # Good explanation with relevant concepts
    
    return score


def brevity_compliance(response_data):
    """
    Measure adherence to "In one short sentence" instruction.
    
    The instruction explicitly says "one short sentence" but 2-3 sentences
    is often acceptable for complex explanations.
    
    Scoring based on sentence count and length:
    - 1.0: 1-2 sentences, reasonable length (50-200 chars)
    - 0.8: 2-3 sentences, still brief (200-400 chars)
    - 0.5: 4-5 sentences or medium length (400-600 chars)
    - 0.2: 6-10 sentences or long (600-1000 chars)
    - 0.0: Very verbose (>10 sentences or >1000 chars)
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Brevity score (1.0 = perfect brevity)
    """
    if not response_data.get('success', False):
        return 0.0
    
    generated = response_data.get('response', '').strip()
    
    if not generated:
        return 0.0
    
    # Count sentences by splitting on .!?
    # Filter out very short fragments
    sentences = re.split(r'[.!?]+', generated)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    sentence_count = len(sentences)
    
    response_length = len(generated)
    
    # Score based on both sentence count and length
    if sentence_count <= 2 and 50 <= response_length <= 200:
        return 1.0  # Perfect: 1-2 short sentences
    elif sentence_count <= 3 and response_length <= 400:
        return 0.8  # Good: 2-3 sentences, still brief
    elif sentence_count <= 5 and response_length <= 600:
        return 0.5  # Acceptable: slightly verbose
    elif sentence_count <= 10 and response_length <= 1000:
        return 0.2  # Poor: quite verbose
    else:
        return 0.0  # Very poor: extremely verbose


def format_cleanliness(response_data):
    """
    Penalize formatting bloat, meta-commentary, and extraneous content.
    
    The response should be a clean explanation without:
    1. Step-by-step formatting (## Step 1, ## Step 2)
    2. Meta-commentary ("The best answer is X" when answer already given)
    3. Unrelated content generation (random questions/riddles)
    4. Looping/repetition
    
    Scoring starts at 1.0 with penalties:
    - -0.2: Step-by-step formatting
    - -0.2: Meta-commentary about "best answer"
    - -0.3: Unrelated content generation
    - -0.5: Looping/repetition detected
    
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
    
    # Check 1: Step-by-step or formal formatting
    step_patterns = [
        r'##\s*step\s*\d+',
        r'step\s*\d+:',
        r'the final answer is:',
    ]
    
    if any(re.search(pattern, generated_lower) for pattern in step_patterns):
        score -= 0.2
    
    # Check 2: Meta-commentary about "best answer"
    # The answer was already given, so saying "The best answer is X" is redundant
    meta_phrases = [
        'the best answer is',
        'the correct answer is',
        'answer is:',
    ]
    
    # Count how many times these appear (some meta-commentary is OK, excessive is not)
    meta_count = sum(1 for phrase in meta_phrases if phrase in generated_lower)
    if meta_count >= 2:
        score -= 0.2
    elif meta_count >= 3:
        score -= 0.3
    
    # Check 3: Unrelated content generation
    # Look for signs of generating unrelated Q&A or riddles
    unrelated_patterns = [
        r'let me know if you want',
        r'another question:',
        r'what.+\?.*\?',  # Multiple questions (sign of generating unrelated content)
        r'i can generate',
        r'would you like me to',
    ]
    
    if any(re.search(pattern, generated_lower) for pattern in unrelated_patterns):
        score -= 0.3
    
    # Check 4: Looping/repetition
    # Split into sentences and check for repetition
    sentences = re.split(r'[.!?]+', generated)
    sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 15]
    
    if len(sentences) > 5:
        # Count repeated sentences
        sentence_counts = {}
        for sentence in sentences:
            sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
        
        max_repeats = max(sentence_counts.values()) if sentence_counts else 0
        if max_repeats >= 3:
            score -= 0.5  # Severe looping
        elif max_repeats >= 2:
            score -= 0.3  # Some repetition
    
    return max(0.0, score)


# Metric registry for Commonsense Reasoning Answer Justification task
COMMONSENSE_REASONING_JUSTIFICATION_METRICS = {
    'explanation_correctness': explanation_correctness,
    'brevity_compliance': brevity_compliance,
    'format_cleanliness': format_cleanliness
}
