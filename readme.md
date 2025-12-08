# XAI Explanation Evaluation Framework

A command-line framework for evaluating LLMs on explainable AI tasks with support for custom metrics, multiple prompting strategies, and interactive visualizations.

## Quick Start
```bash
# Clone and setup
git clone https://github.com/Frenzoid/USI_Thesis_XAI
cd USI_Thesis_XAI
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure API keys
cp .env.template .env
# Edit .env with your API keys (OpenAI, Google GenAI, Anthropic)

# Check system status
python main.py status
python main.py list-options
```

## Core Commands

### Run Experiments
```bash
python main.py run-experiment --model MODEL --setup SETUP --prompt PROMPT --mode MODE [options]

Options:
  --model MODEL         Model name(s) or "all"
  --setup SETUP         Setup name(s) or "all"
  --prompt PROMPT       Prompt name(s) or "all"
  --mode MODE           zero-shot or few-shot
  --few-shot-row N      Row index for few-shot example
  --size N              Sample size (default: configured value)
  --temperature TEMP    Generation temperature (default: 0.1)
  --skip                Skip existing experiments
  --force               Bypass validation

# Examples
python main.py run-experiment --model gpt-4o-mini --setup my_task --prompt explain_v1 --mode zero-shot --size 100
python main.py run-experiment --model "gpt-4o-mini claude-3.5-sonnet" --setup my_task
python main.py run-experiment --mode few-shot --few-shot-row 42 --setup my_task
```

### Evaluate Results
```bash
python main.py evaluate [--experiment NAME]

# Evaluate all experiments
python main.py evaluate

# Evaluate specific experiment
python main.py evaluate --experiment my_experiment_name
```

### Generate Visualizations
```bash
python main.py plot

# Generates interactive HTML dashboard in ./outputs/plots/
```

### Utility Commands
```bash
python main.py download-datasets --setup SETUP    # Download datasets
python main.py show-prompt --prompt NAME --row N  # Preview populated prompt
python main.py validate-prompts                   # Validate prompt configs
python main.py dataset validate                   # Validate dataset fields
python main.py status                             # Check system status
python main.py list-options                       # List models, setups, prompts
python main.py cleanup --target TARGET            # Clean files (datasets/logs/results/cache/all)
```

### Finetuning
```bash
python main.py finetune run --model BASE_MODEL --tune CONFIG_NAME
python main.py finetune list                      # List finetune configs
python main.py finetune models                    # List finetuned models
```

## Configuration

All configs are in `./configs/`:

### setups.json - Dataset and Task Definitions
```json
{
  "my_task": {
    "description": "Task description",
    "dataset": {
      "download_link": "https://example.com/data.zip",
      "download_path": "DS_MY_TASK",
      "csv_file": "data/examples.csv"
    },
    "prompt_fields": {
      "question_fields": ["question", "context"],
      "answer_field": "explanation"
    },
    "prune_row": {
      "explanation": ["", "NA", "n/a"]
    },
    "custom_metrics": {
      "module_path": "custom_metrics.my_metrics",
      "metrics_registry": "MY_METRICS"
    }
  }
}
```

### prompts.json - Prompt Templates
```json
{
  "my_prompt_zeroshot": {
    "mode": "zero-shot",
    "compatible_setup": "my_task",
    "template": "Question: {}\nContext: {}\nProvide an explanation.",
    "description": "Basic zero-shot prompt"
  },
  "my_prompt_fewshot": {
    "mode": "few-shot",
    "compatible_setup": "my_task",
    "few_shot_example": "Example:\nQuestion: {}\nContext: {}\nExplanation: {}\n\n",
    "template": "Now answer:\nQuestion: {}\nContext: {}\nProvide an explanation.",
    "description": "Few-shot prompt"
  }
}
```

### models.json - Model Definitions
```json
{
  "gpt-4o-mini": {
    "type": "api",
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "max_tokens": 256
  },
  "llama-8b": {
    "type": "local",
    "model_path": "unsloth/llama-3-8b-instruct-bnb-4bit",
    "load_in_4bit": true,
    "use_unsloth": true,
    "use_chat_template": true,
    "max_tokens": 256
  }
}
```

### finetunes.json - Finetuning Configs
```json
{
  "my_finetune": {
    "setup": "my_task",
    "prompt": "my_prompt_zeroshot",
    "hyperparameters": {
      "learning_rate": 2e-4,
      "num_epochs": 3,
      "batch_size": 4,
      "lora_r": 16,
      "lora_alpha": 32
    },
    "training_config": {
      "max_samples": 1000,
      "train_test_split": 0.8
    }
  }
}
```

## Custom Metrics

Create `custom_metrics/my_metrics.py`:
```python
def has_reasoning(response_data):
    """Check for causal reasoning language."""
    if not response_data.get('success', False):
        return 0.0
    response = response_data.get('response', '').lower()
    keywords = ['because', 'since', 'therefore', 'thus']
    return 1.0 if any(k in response for k in keywords) else 0.0

MY_METRICS = {
    'has_reasoning': has_reasoning
}
```

Reference in setup config:
```json
"custom_metrics": {
  "module_path": "custom_metrics.my_metrics",
  "metrics_registry": "MY_METRICS"
}
```

## Default Evaluation Metrics

- **Token-level**: exact_match, precision, recall, f1_score, jaccard
- **N-gram**: bleu, rouge1_f, rouge2_f, rougeL_f
- **Semantic**: semantic_similarity (sentence embeddings)

## Output Structure
```
outputs/
├── responses/      # Model responses with metadata
├── evaluations/    # Computed metrics
└── plots/          # Interactive HTML dashboards
```

## Supported Models

**API Models**: OpenAI (GPT-4o, GPT-4o-mini), Google (Gemini), Anthropic (Claude)

**Local Models**: Llama, Mistral, Qwen, Phi, Gemma (via Unsloth with 4-bit quantization)

## Hardware Requirements

- **API models**: Any system with internet
- **Local models**: GPU with 6GB+ VRAM (12GB+ for 7B+ models)
```bash
python main.py status  # Check GPU availability and memory
```

## Typical Workflow
```bash
# 1. Define setup in configs/setups.json
# 2. Create prompts in configs/prompts.json
# 3. Select models in configs/models.json

# 4. Preview prompt
python main.py show-prompt --prompt my_prompt --row 10

# 5. Run experiment
python main.py run-experiment --model gpt-4o-mini --setup my_task --prompt my_prompt --mode zero-shot --size 100

# 6. Evaluate
python main.py evaluate

# 7. Visualize
python main.py plot
```
