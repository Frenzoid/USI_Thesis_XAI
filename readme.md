# XAI Explanation Evaluation System

A command-line system for evaluating Large Language Models' performance in XAI (Explainable AI) explanation tasks, comparing their outputs against human annotations.

## 🚀 Quick Start

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd xai-explanation-evaluation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure API keys:**
```bash
cp .env.template .env
# Edit .env with your API keys (OpenAI, Google GenAI, Anthropic)
```

3. **Check system status:**
```bash
python main.py status
python main.py list-options
```

### First Experiment

```bash
# Run a simple experiment
python main.py run-experiment --model gpt-4o-mini --dataset gmeg --prompt gmeg_v1_basic --size 20

# Evaluate results
python main.py evaluate --experiment-type baseline

# Generate plots
python main.py plot --experiment-type baseline
```

## 📁 Project Structure

```
xai-explanation-evaluation/
├── main.py                  # CLI entry point - command dispatcher
├── config.py                # Configuration management and constants
├── experiment_runner.py     # Orchestrates inference experiments
├── evaluator.py             # Evaluation pipeline runner
├── plotter.py               # Visualization pipeline runner
├── models.py                # Model management (local + API)
├── dataset_manager.py       # Dataset loading and preparation
├── prompt_manager.py        # Prompt template management
├── evaluation.py            # Core evaluation metrics framework
├── visualization.py         # Core visualization framework
├── utils.py                 # Utility functions and system setup
├── requirements.txt         # Python dependencies
├── configs/                 # Configuration files
│   ├── models.json          # Model definitions and parameters
│   ├── datasets.json        # Dataset sources and metadata
│   └── prompts.json         # Prompt templates
├── outputs/                 # Generated outputs
│   ├── responses/baseline/  # Model inference results
│   ├── evaluations/baseline/# Evaluation metrics
│   └── plots/baseline/      # Visualizations
├── datasets/                # Downloaded datasets
├── cached_models/            # Cached model files (renamed for consistency)
├── finetuned_models/        # Finetuned models (future)
└── logs/                    # System logs
```

## 🎯 Available Commands

### Core Experiment Commands

**`run-experiment`** - Run inference experiments
```bash
python main.py run-experiment [options]

Options:
  --model MODELS           Space-separated model names or 'all'
  --dataset DATASETS       Space-separated dataset names or 'all'
  --prompt PROMPTS         Space-separated prompt names or 'all'
  --size SIZE              Sample size (default: 50)
  --temperature TEMP       Generation temperature (default: 0.1)
  --experiment-type TYPE   Experiment type (default: baseline)
  --force                  Ignore validation errors

Examples:
  # Single experiment
  python main.py run-experiment --model llama3.2-1b --dataset gmeg --prompt gmeg_v1_basic
  
  # Multiple models
  python main.py run-experiment --model "gpt-4o-mini gemini-1.5-flash" --dataset gmeg --prompt gmeg_v1_basic
  
  # All configurations
  python main.py run-experiment --experiment-type baseline
```

**`evaluate`** - Evaluate experiment results
```bash
python main.py evaluate [options]

Options:
  --experiment NAMES       Comma-separated experiment names
  --experiment-type TYPE   Filter by experiment type

Examples:
  # Specific experiment
  python main.py evaluate --experiment baseline_gmeg_gpt-4o-mini_gmeg_v1_basic_50_0p1
  
  # All baseline experiments
  python main.py evaluate --experiment-type baseline
```

**`plot`** - Generate visualizations
```bash
python main.py plot [options]

Options:
  --experiment NAMES       Comma-separated experiment names
  --experiment-type TYPE   Filter by experiment type
  --compare               Generate comparison plots

Examples:
  # Individual plots
  python main.py plot --experiment baseline_gmeg_gpt-4o-mini_gmeg_v1_basic_50_0p1
  
  # Comparison plots
  python main.py plot --experiment "exp1,exp2,exp3" --compare
  
  # All plots for experiment type
  python main.py plot --experiment-type baseline
```

### Utility Commands

**`download-datasets`** - Download datasets
```bash
python main.py download-datasets --dataset DATASETS

Examples:
  python main.py download-datasets --dataset gmeg
  python main.py download-datasets --dataset all
```

**`cleanup`** - Clean up system files
```bash
python main.py cleanup --target TARGETS [--dry-run]

Targets: datasets, logs, results, cache, finetuned, all

Examples:
  python main.py cleanup --target logs --dry-run
  python main.py cleanup --target all
```

**`list-options`** - Show available models, datasets, prompts
```bash
python main.py list-options
```

**`list-commands`** - Show all commands with examples
```bash
python main.py list-commands
```

**`status`** - Check system status
```bash
python main.py status
```

## 🤖 Supported Models

### Local Models (via Unsloth)
- **Llama**: 3.2-1b, 3.2-3b, 3-8b
- **Mistral**: 7b
- **Others**: Qwen2, Phi-3, Gemma, CodeLlama, TinyLlama

### API Models
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Google**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus

## 📊 Available Datasets

- **GMEG**: Grammatical error correction explanations dataset

## 📝 Available Prompts

All prompts work with the GMEG dataset:

- `gmeg_v1_basic` - Basic correction explanation
- `gmeg_v2_enhanced` - Enhanced with categorization
- `gmeg_v3_detailed` - Detailed linguistic analysis
- `gmeg_v4_minimal` - Minimal, concise explanations
- `gmeg_v5_pedagogical` - Educational, teaching-focused
- `gmeg_v6_formal` - Formal academic analysis
- `gmeg_v7_casual` - Casual, conversational style
- `gmeg_v8_comparative` - Side-by-side comparison
- `gmeg_few_shot` - Few-shot learning with examples

## 📈 Evaluation Metrics

- **Token-based**: F1, Precision, Recall, Exact Match, Jaccard
- **Semantic**: Cosine similarity using sentence embeddings
- **Dataset-specific**: GMEG correction terminology, format consistency

## 🔧 Configuration

Edit JSON files in `configs/`:

- `models.json` - Add new models and parameters
- `datasets.json` - Configure dataset sources
- `prompts.json` - Create custom prompt templates

Example model configuration:
```json
{
  "my-model": {
    "type": "api",
    "provider": "openai",
    "model_name": "gpt-4",
    "max_tokens": 256,
    "description": "My custom model"
  }
}
```

## 🔍 Example Workflows

**Model Comparison Study:**
```bash
# Run experiments with multiple models
python main.py run-experiment --model "gpt-4o-mini gemini-1.5-flash llama3.2-1b" --dataset gmeg --prompt gmeg_v1_basic --size 100

# Evaluate all results
python main.py evaluate --experiment-type baseline

# Generate comparison plots
python main.py plot --experiment-type baseline --compare
```

**Prompt Engineering Analysis:**
```bash
# Test different prompts
python main.py run-experiment --model gpt-4o-mini --dataset gmeg --prompt "gmeg_v1_basic gmeg_v2_enhanced gmeg_v4_minimal" --size 50

# Compare prompt effectiveness
python main.py plot --experiment-type baseline --compare
```

## 🛠️ Dependencies

Core requirements (see `requirements.txt` for full list):
- `torch` - PyTorch for local models
- `transformers` - Hugging Face transformers
- `openai` - OpenAI API client
- `google-genai` - Google Gemini API
- `anthropic` - Anthropic Claude API
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation

Optional for local models:
- `unsloth` - Efficient local model loading
- `trl` - Training and fine-tuning tools

## 🔑 API Keys

Required environment variables in `.env`:
```bash
OPENAI_API_KEY=your-openai-key
GENAI_API_KEY=your-google-genai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

## 📄 Output Files

Experiments generate structured outputs:
- **Inference**: `outputs/responses/baseline/inference_{name}.json`
- **Evaluation**: `outputs/evaluations/baseline/evaluation_{name}.json`
- **Plots**: `outputs/plots/baseline/plot_{name}.html`

## 🎮 GPU/CPU Requirements

### **GPU Recommended For:**
- **Local Models**: Running Llama, Mistral, etc. requires significant memory
- **Minimum**: 6GB VRAM for smaller models (1B-3B parameters)  
- **Recommended**: 12GB+ VRAM for larger models (7B+ parameters)

### **CPU-Only Limitations:**
- **Local models will be very slow** (10-100x slower than GPU)
- **High RAM usage** (16GB+ recommended for larger models)
- **API models work perfectly** on any system

## 🚨 Troubleshooting
```bash
# Check your system capabilities
python main.py status

# This will show:
# - GPU availability and memory
# - RAM status  
# - Compatible model types
# - Recommendations for your system
```

The system **automatically prevents** GPU-dependent operations on incompatible hardware while allowing all other functionality.

**Common Issues:**

1. **Missing API keys**: Add keys to `.env` file
2. **Unsloth not found**: Install with `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
3. **Dataset download fails**: Check internet connection, try `python main.py download-datasets --dataset gmeg`
4. **Configuration errors**: Use `python main.py status` to check system health

**Getting Help:**
```bash
python main.py list-commands  # Show all commands
python main.py status         # Check system status
python main.py list-options   # Show available options
```