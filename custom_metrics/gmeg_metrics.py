"""
Custom metrics for GMEG (Grammatical Error Correction Explanations) dataset.

Each metric function receives a response dictionary with the following fields:
- prompt: The input prompt sent to the model
- response: The model's generated response  
- expected_output: The expected/reference response
- question_values: List of question field values used to populate the prompt
- success: Boolean indicating if the response generation was successful
- error: Error message if success is False, None otherwise

Each function should return a float value between 0.0 and 1.0.
"""

def bullet_point_ratio(response_data):
    """
    Compare bullet point usage between generated and expected responses.
    
    Many GMEG explanations use bullet points to list corrections.
    This metric measures how well the model matches the expected bullet point usage.
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Ratio of generated to expected bullet points (capped at 2.0)
    """
    generated = response_data.get('response', '')
    expected = response_data.get('expected_output', '')
    
    if not response_data.get('success', False):
        return 0.0
    
    # Count bullet points in both texts
    gen_bullets = len([line for line in generated.split('\n') 
                      if line.strip().startswith(('-', '•', '*'))])
    exp_bullets = len([line for line in expected.split('\n') 
                      if line.strip().startswith(('-', '•', '*'))])
    
    # Calculate ratio (capped at 2.0 to avoid extreme values)
    if exp_bullets == 0 and gen_bullets == 0:
        return 1.0  # Perfect match when both have no bullets
    elif exp_bullets == 0:
        return 0.0  # Generated has bullets but expected doesn't
    else:
        ratio = gen_bullets / exp_bullets
        return min(ratio, 2.0)


def correction_terminology_recall(response_data):
    """
    Measure how well the model uses correction-specific terminology.
    
    Checks for words that indicate specific types of grammatical corrections.
    Higher scores indicate better use of correction vocabulary.
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Recall score for correction terminology usage
    """
    generated = response_data.get('response', '')
    expected = response_data.get('expected_output', '')
    
    if not response_data.get('success', False):
        return 0.0
    
    # Define correction terminology
    correction_terms = [
        'spelling', 'grammar', 'punctuation', 'capitalization', 'word choice',
        'corrected', 'changed', 'replaced', 'added', 'removed', 'fixed',
        'error', 'mistake', 'wrong', 'incorrect', 'subject-verb', 'agreement',
        'tense', 'plural', 'singular', 'article'
    ]
    
    gen_lower = generated.lower()
    exp_lower = expected.lower()
    
    # Count correction terms in both texts
    gen_correction_terms = sum(1 for term in correction_terms if term in gen_lower)
    exp_correction_terms = sum(1 for term in correction_terms if term in exp_lower)
    
    # Calculate recall
    if exp_correction_terms == 0:
        return 1.0 if gen_correction_terms == 0 else 0.5  # No expected terms
    else:
        recall = min(gen_correction_terms / exp_correction_terms, 1.0)
        return recall


def structural_format_match(response_data):
    """
    Check if generated and expected responses follow similar structural formats.
    
    Both should follow similar formatting patterns (bullet points vs paragraphs).
    This helps ensure the model produces appropriately structured explanations.
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: 1.0 if formats match, 0.0 if they don't
    """
    generated = response_data.get('response', '')
    expected = response_data.get('expected_output', '')
    
    if not response_data.get('success', False):
        return 0.0
    
    # Check if both use bullet points
    gen_bullets = len([line for line in generated.split('\n') 
                      if line.strip().startswith(('-', '•', '*'))])
    exp_bullets = len([line for line in expected.split('\n') 
                      if line.strip().startswith(('-', '•', '*'))])
    
    # Both have structured format (bullets) or both don't
    gen_has_structure = gen_bullets > 0
    exp_has_structure = exp_bullets > 0
    
    return 1.0 if gen_has_structure == exp_has_structure else 0.0


def original_text_mention(response_data):
    """
    Check if the model mentions or references the original text content.
    
    Good explanations often reference specific parts of the original text
    that were corrected. This metric rewards such references.
    
    Args:
        response_data: Dictionary containing response fields
        
    Returns:
        float: Score based on how well original text is referenced
    """
    generated = response_data.get('response', '')
    question_values = response_data.get('question_values', [])
    
    if not response_data.get('success', False) or len(question_values) < 2:
        return 0.0
    
    original_text = question_values[0]  # First question value is original text
    
    if not original_text.strip():
        return 1.0  # No original text to reference
    
    # Extract key words from original text (excluding common words)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                   'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    original_words = set(word.lower().strip('.,!?";') 
                        for word in original_text.split() 
                        if len(word) > 2 and word.lower() not in common_words)
    
    if not original_words:
        return 1.0  # No meaningful words to reference
    
    # Check how many original words appear in the generated response
    generated_lower = generated.lower()
    referenced_words = sum(1 for word in original_words if word in generated_lower)
    
    # Score based on percentage of original words referenced
    reference_ratio = referenced_words / len(original_words)
    return min(reference_ratio, 1.0)


# Metric registry for this dataset
GMEG_METRICS = {
    'bullet_point_ratio': bullet_point_ratio,
    'correction_terminology_recall': correction_terminology_recall, 
    'structural_format_match': structural_format_match,
    'original_text_mention': original_text_mention
}