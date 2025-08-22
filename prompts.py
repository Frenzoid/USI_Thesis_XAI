import pandas as pd
from typing import Dict, List
from utils import setup_logging

logger = setup_logging("prompts")

class PromptManager:
    """Centralized prompt management system"""
    
    def __init__(self):
        self.prompts = {
            'gmeg_v1_basic': {
                'description': 'Basic correction explanation prompt',
                'template': """
You will be given 2 texts, one original and one revised with corrections.

Identify each correction.
For each correction, provide a brief, clear description of the change
Present all changes as bullet points

For example:
Original Text: "I believe there are a lot of high buildings now in Japan."
Revised Text: "I believe there are a lot of tall buildings now in Japan."
Expected output:
- Word "high" corrected to "tall" (more appropriate adjective for describing buildings)

Original Text:
{original_text}

Revised Text:
{revised_text}
""".strip()
            },
            
            'gmeg_v2_enhanced': {
                'description': 'Enhanced correction explanation with categorization',
                'template': """
You are an expert in text correction analysis. You will be given an original text and a revised version.

Your task:
1. Identify each correction made between the original and revised text
2. Categorize each correction (spelling, grammar, word choice, punctuation, etc.)
3. Provide a clear explanation for each change
4. Present all changes as bullet points

Format: "- [Category] Change description"

Example:
Original: "The effect of the new policy will effect everyone."
Revised: "The effect of the new policy will affect everyone."
Output:
- [Word Choice] "effect" changed to "affect" (correct verb form for influence)

Original Text:
{original_text}

Revised Text:
{revised_text}
""".strip()
            },
            
            'gmeg_v3_detailed': {
                'description': 'Detailed linguistic analysis prompt',
                'template': """
As a linguistics expert, analyze the corrections made between these two texts:

Original Text: {original_text}
Revised Text: {revised_text}

Provide a comprehensive analysis including:
1. Type of error corrected (grammatical, lexical, syntactic, semantic)
2. Linguistic rule or principle applied
3. Impact on text clarity and correctness
4. Alternative corrections that could have been made

Format each analysis as:
- [Error Type] Original → Corrected: Explanation (Linguistic principle: X)

Be thorough and educational in your explanations.
""".strip()
            },
            
            'gmeg_v4_minimal': {
                'description': 'Minimal, concise correction explanation',
                'template': """
Compare these texts and list only the essential changes:

Original: {original_text}
Revised: {revised_text}

Output format:
- Changed X to Y
- Added/Removed Z

Be concise and direct.
""".strip()
            },
            
            'gmeg_v5_pedagogical': {
                'description': 'Educational teaching-focused prompt',
                'template': """
You are a writing tutor explaining corrections to a student. Compare these texts:

Original: {original_text}
Revised: {revised_text}

Explain each correction in a way that helps the student learn:
- What was wrong and why
- The correct form and why it's better
- How to avoid this mistake in the future

Use encouraging, educational language suitable for language learners.
""".strip()
            },
            
            'gmeg_v6_formal': {
                'description': 'Formal academic analysis style',
                'template': """
Conduct a formal textual analysis of the revisions made:

Text A (Original): {original_text}
Text B (Revised): {revised_text}

Analysis:
Examine the modifications systematically, categorizing each change according to linguistic taxonomy and explaining the rationale behind each editorial decision. Consider morphological, syntactic, semantic, and pragmatic dimensions of the alterations.

Present findings in academic format with precise terminology.
""".strip()
            },
            
            'gmeg_v7_casual': {
                'description': 'Casual, conversational explanation style',
                'template': """
Hey! I need to explain what changed between these two texts:

Original: {original_text}
Fixed: {revised_text}

Here's what got corrected:

(Explain in a friendly, casual way what was wrong and how it got fixed. Use everyday language that anyone can understand.)
""".strip()
            },
            
            'gmeg_v8_comparative': {
                'description': 'Side-by-side comparative analysis',
                'template': """
BEFORE vs AFTER Analysis:

BEFORE: {original_text}
AFTER:  {revised_text}

Changes Made:
[Analyze each change by directly comparing the before and after states, highlighting what specifically changed and why each change improves the text]

Focus on the transformation process and improvement achieved.
""".strip()
            },
            
            'gmeg_few_shot': {
                'description': 'Few-shot learning prompt with examples',
                'template': """
{few_shot_examples}

Now analyze the following:

Original Text:
{original_text}

Revised Text:
{revised_text}
""".strip()
            },
        }
        
        logger.info(f"PromptManager initialized with {len(self.prompts)} prompt templates")

    def get_prompt(self, prompt_key: str, **kwargs) -> str:
        """Get formatted prompt with variables"""
        if prompt_key not in self.prompts:
            available = list(self.prompts.keys())
            raise ValueError(f"Unknown prompt: {prompt_key}. Available: {available}")
        
        template = self.prompts[prompt_key]['template']
        try:
            formatted_prompt = template.format(**kwargs)
            logger.debug(f"Generated prompt for key '{prompt_key}' with length {len(formatted_prompt)}")
            return formatted_prompt
        except KeyError as e:
            logger.error(f"Missing required parameter for prompt '{prompt_key}': {e}")
            raise
        
    def list_prompts(self) -> Dict[str, str]:
        """List all available prompts with descriptions"""
        return {k: v['description'] for k, v in self.prompts.items()}

    def add_prompt(self, key: str, template: str, description: str):
        """Add a new prompt template"""
        self.prompts[key] = {
            'template': template,
            'description': description
        }
        logger.info(f"Added new prompt template: {key}")

    def generate_few_shot_examples(self, dataset: pd.DataFrame, n_examples: int = 3) -> str:
        """Generate few-shot examples from dataset"""
        logger.info(f"Generating {n_examples} few-shot examples from dataset")
        
        examples = dataset.sample(min(n_examples, len(dataset)))
        formatted_examples = []
        
        for _, row in examples.iterrows():
            # For GMEG dataset, use the correct field names
            original = row.get('original', 'N/A')
            revised = row.get('revised', 'N/A')
            explanation = row.get('please_explain_the_revisions_write_na_if_not_annotatable', 
                                row.get('explanation', 'N/A'))
            
            # Skip examples marked as 'NA' or 'not annotatable'
            if str(explanation).lower().strip() in ['na', 'n/a', 'not applicable', 'not annotatable']:
                continue
                
            example = f"""
Original: {original}
Revised: {revised}
Analysis: {explanation}
""".strip()
            formatted_examples.append(example)
        
        # If we don't have enough valid examples, pad with what we have
        if not formatted_examples:
            # Fallback to any example, even if NA
            row = examples.iloc[0]
            example = f"""
Original: {row.get('original', 'N/A')}
Revised: {row.get('revised', 'N/A')}
Analysis: {row.get('please_explain_the_revisions_write_na_if_not_annotatable', 'No analysis available')}
""".strip()
            formatted_examples.append(example)
        
        result = "\n\n".join(formatted_examples)
        logger.debug(f"Generated few-shot examples with {len(formatted_examples)} examples")
        return result
    
    def validate_prompt_parameters(self, prompt_key: str, **kwargs) -> bool:
        """Validate that all required parameters are provided for a prompt"""
        if prompt_key not in self.prompts:
            return False
        
        template = self.prompts[prompt_key]['template']
        
        # Extract parameter names from template
        import re
        param_names = set(re.findall(r'\{(\w+)\}', template))
        provided_params = set(kwargs.keys())
        
        missing_params = param_names - provided_params
        if missing_params:
            logger.warning(f"Missing parameters for prompt '{prompt_key}': {missing_params}")
            return False
        
        return True
    
    def get_prompt_info(self, prompt_key: str = None) -> Dict:
        """Get information about prompts"""
        if prompt_key:
            if prompt_key in self.prompts:
                info = dict(self.prompts[prompt_key])
                # Extract parameter names
                import re
                param_names = re.findall(r'\{(\w+)\}', info['template'])
                info['required_parameters'] = list(set(param_names))
                return info
            else:
                return {}
        else:
            # Return info for all prompts
            all_info = {}
            for key in self.prompts:
                all_info[key] = self.get_prompt_info(key)
            return all_info
