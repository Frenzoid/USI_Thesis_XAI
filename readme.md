# XAI Explanation Evaluation System

A comprehensive command-line system for evaluating Large Language Models' performance in XAI (Explainable AI) explanation tasks, with intelligent argument resolution, custom metrics, and sophisticated evaluation pipelines.

## 🌟 Key Features

- **Intelligent Argument Resolution**: Space-separated multi-input handling with automatic compatibility checking
- **Mode-based Prompting**: Zero-shot and few-shot prompting with automatic validation
- **Custom Metrics**: Dataset-specific evaluation metrics through a plugin system  
- **Metadata-based Processing**: Content-based metadata extraction for robust file handling
- **GPU/CPU Awareness**: Automatic hardware detection with appropriate model routing
- **Interactive Visualizations**: Plotly-based interactive HTML reports and comparisons
- **Comprehensive Evaluation**: Token-based metrics + semantic similarity + custom dataset metrics
- **Prompt Preview**: Live template population with real dataset data

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

### First Steps

```bash
# Preview a prompt template with real data
python main.py show-prompt --prompt gmeg_v1_basic --row 42

# Run a simple experiment (system handles compatibility automatically)
python main.py run-baseline-exp --model gpt-4o-mini --dataset gmeg --size 20

# Multiple models with intelligent argument resolution
python main.py run-baseline-exp --model "gpt-4o-mini claude-3.5-sonnet" --dataset gmeg

# Evaluate results
python main.py evaluate

# Generate interactive plots
python main.py plot
```

## 📁 Project Structure

```
xai-explanation-evaluation/
├── main.py                  # CLI with intelligent argument resolver
├── config.py                # Configuration management and validation
├── experiment_runner.py     # Experiment orchestration with mode support
├── evaluator_runner.py      # Evaluation pipeline with custom metrics
├── plotter.py               # Visualization pipeline with metadata detection
├── models.py                # Unified model management (local + API)
├── dataset_manager.py       # Generic dataset handling with field mapping
├── prompt_manager.py        # Mode-based prompt management (zero-shot/few-shot)
├── evaluation.py            # Core evaluation framework with custom metrics
├── visualization.py         # Interactive visualization framework
├── utils.py                 # System utilities and hardware detection
├── requirements.txt         # Python dependencies
├── configs/                 # JSON configuration files
│   ├── models.json          # Model definitions and parameters
│   ├── datasets.json        # Dataset sources and field mappings
│   └── prompts.json         # Prompt templates with mode support
├── custom_metrics/          # Dataset-specific custom metrics
│   └── gmeg_metrics.py      # GMEG correction analysis metrics
├── outputs/                 # Organized experiment outputs
│   ├── responses/baseline/  # Model inference results
│   ├── evaluations/baseline/# Evaluation metrics and scores
│   └── plots/baseline/      # Interactive HTML visualizations
├── datasets/                # Downloaded datasets
├── cached_models/           # Cached model files
├── finetuned_models/        # Custom finetuned models (future)
└── logs/                    # System logs with component separation
```

## 🎯 Available Commands

### Core Experiment Commands

**`run-baseline-exp`** - Run inference experiments with intelligent argument resolution
```bash
python main.py run-baseline-exp [options]

Options:
  --model MODELS           Space-separated model names or 'all'
  --dataset DATASETS       Space-separated dataset names or 'all'  
  --prompt PROMPTS         Space-separated prompt names or 'all'
  --mode MODE              Prompting mode: zero-shot or few-shot
  --few-shot-row ROW       Specific row index for few-shot example (0-based)
  --size SIZE              Sample size (default: 50)
  --temperature TEMP       Generation temperature (default: 0.1)
  --force                  Force run despite compatibility warnings

Intelligent Argument Resolution:
  • Specify any combination - system handles compatibility automatically
  • Multiple values: space-separated ("gpt-4o-mini claude-3.5-sonnet") or comma-separated
  • Unspecified arguments default to all compatible options
  • Dataset-prompt compatibility automatically validated
  • Mode filtering only uses prompts that support the specified mode

Examples:
  # Single experiment with automatic compatibility
  python main.py run-baseline-exp --model gpt-4o-mini --dataset gmeg
  
  # Multiple models with intelligent expansion
  python main.py run-baseline-exp --model "gpt-4o-mini gemini-1.5-flash" --dataset gmeg
  
  # Mode-based filtering (only compatible prompts selected)
  python main.py run-baseline-exp --mode few-shot --dataset gmeg
  
  # Few-shot with specific example row
  python main.py run-baseline-exp --mode few-shot --few-shot-row 42 --dataset gmeg
  
  # All compatible combinations
  python main.py run-baseline-exp
```

**`show-prompt`** - Display populated prompt templates with real data
```bash
python main.py show-prompt [options]

Options:
  --prompt PROMPT_NAME     Name of prompt template to show (required)
  --row ROW_NUMBER         Specific row number to use from dataset (0-based, optional)
                           If not specified, uses a random row

Examples:
  # Show prompt with specific dataset row
  python main.py show-prompt --prompt gmeg_v1_basic --row 42
  
  # Show prompt with random dataset row  
  python main.py show-prompt --prompt gmeg_v2_enhanced
  
  # Preview few-shot prompt structure
  python main.py show-prompt --prompt gmeg_few_shot --row 15
  
  # Check all prompts for a dataset
  python main.py show-prompt --prompt gmeg_v4_minimal
```

**`evaluate`** - Evaluate experiment results with custom metrics
```bash
python main.py evaluate [options]

Options:
  --experiment NAMES       Comma-separated experiment names
  --experiment-type TYPE   Filter by experiment type

Examples:
  # Evaluate specific experiment
  python main.py evaluate --experiment baseline__gmeg__gpt-4o-mini__zero-shot__gmeg_v1_basic__50__0p1
  
  # Evaluate all experiments
  python main.py evaluate
```

**`plot`** - Generate interactive visualizations
```bash
python main.py plot [options]

Options:
  --experiment NAMES       Comma-separated experiment names
  --experiment-type TYPE   Filter by experiment type
  --compare               Generate comparison plots

Examples:
  # Individual interactive plots
  python main.py plot --experiment baseline__gmeg__gpt-4o-mini__zero-shot__gmeg_v1_basic__50__0p1
  
  # Comparison dashboard
  python main.py plot --experiment "exp1,exp2,exp3" --compare
  
  # All plots with navigation index
  python main.py plot
```

### Utility Commands

**`download-datasets`** - Download and prepare datasets
```bash
python main.py download-datasets --dataset DATASETS

Examples:
  python main.py download-datasets --dataset gmeg
  python main.py download-datasets --dataset "gmeg RHAI_1"
  python main.py download-datasets --dataset all
```

**`cleanup`** - Clean up system files
```bash
python main.py cleanup --target TARGETS [--dry-run]

Targets: datasets, logs, results, cache, finetuned, all
```

**Information and Help Commands:**

**`list-options`** - Show available models, datasets, and prompts
```bash
python main.py list-options

# Shows all configured:
# - Models (with type: local/api)  
# - Datasets (with descriptions)
# - Prompts (with compatibility info)
# - Experiment types
```

**`list-commands` / `help` / `show-commands`** - Show all available commands
```bash
python main.py list-commands   # Detailed command reference
python main.py help            # Same as list-commands  
python main.py show-commands   # Same as list-commands

# All three commands show:
# - Complete command syntax
# - All available arguments
# - Usage examples
# - Intelligent argument resolution guide
```

**`status`** - Check system status and capabilities
```bash
python main.py status

# Shows:
# - GPU/CPU availability and memory
# - Configuration file validation
# - API key status
# - Directory structure
# - Hardware recommendations
```

## 🤖 Supported Models

### Local Models (via Unsloth - GPU Recommended)
- **Llama 3.2**: 1B, 3B parameters
- **Llama 3**: 8B parameters  
- **Mistral**: 7B parameters
- **Others**: Qwen2, Phi-3, Gemma, CodeLlama, TinyLlama

### API Models (Works on Any System)
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Google**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus

## 📊 Available Datasets

- **GMEG**: Grammatical error correction explanations with custom metrics
- **RHAI_1**: Reframing Human AI explanations (evaluation format)  
- **RHAI_2**: Reframing Human AI explanations (choice selection format)

*Each dataset includes field mappings, evaluation configurations, and custom metrics*

## 📝 Available Prompts

### GMEG-Compatible Prompts (Zero-shot)
- `gmeg_v1_basic` - Basic correction explanation
- `gmeg_v2_enhanced` - Enhanced with categorization  
- `gmeg_v3_detailed` - Detailed linguistic analysis
- `gmeg_v4_minimal` - Minimal, concise explanations
- `gmeg_v5_pedagogical` - Educational, teaching-focused
- `gmeg_v6_formal` - Formal academic analysis
- `gmeg_v7_casual` - Casual, conversational style
- `gmeg_v8_comparative` - Side-by-side comparison

### GMEG-Compatible Prompts (Few-shot)
- `gmeg_few_shot` - Few-shot learning with examples

### RHAI_1 and RHAI_2 Experiments Prompts (Zero-Shot)
- `exp1_explanation` - RHAI_1 compatible explanation prompt
- `exp2_choice_selection` - RHAI_2 compatible choice selection prompt

*All prompts include mode specifications and dataset compatibility validation*

## 📈 Evaluation Metrics

### Core Metrics (All Datasets)
- **Token-based**: F1, Precision, Recall, Exact Match, Jaccard
- **Semantic**: Cosine similarity using sentence embeddings (all-mpnet-base-v2)

### GMEG Custom Metrics  
- **bullet_point_ratio**: Bullet point formatting consistency
- **correction_terminology_recall**: Use of correction-specific vocabulary
- **structural_format_match**: Format matching (structured vs. paragraph)
- **original_text_mention**: Reference to original text content

*Custom metrics are loaded automatically based on dataset configuration*

## ⚙️ Configuration System

The system uses JSON configuration files with validation:

### Adding a Custom Model (`configs/models.json`)
```json
{
  "my-custom-model": {
    "type": "api",
    "provider": "openai", 
    "model_name": "gpt-4",
    "max_tokens": 256,
    "description": "My custom GPT-4 configuration"
  }
}
```

### Adding a Custom Dataset (`configs/datasets.json`)
```json
{
  "my-dataset": {
    "download_link": "https://example.com/dataset.zip",
    "download_path": "my_dataset_folder", 
    "csv_file": "data.csv",
    "question_fields": ["input_text", "context"],
    "answer_field": "expected_output",
    "description": "My custom dataset",
    "evaluation_config": {
      "na_indicators": ["N/A", "not annotatable", "", "unclear"],
      "case_sensitive": false,
      "skip_empty_responses": true
    },
    "custom_metrics": {
      "module_path": "custom_metrics.my_dataset_metrics",
      "metrics_registry": "MY_METRICS"
    }
  }
}
```

### Adding Custom Prompts (`configs/prompts.json`)
```json
{
  "my-prompt": {
    "type": "baseline",
    "mode": "zero-shot", 
    "compatible_dataset": "my-dataset",
    "template": "Analyze this text: {}\nWith context: {}\n",
    "description": "My custom analysis prompt"
  }
}
```

### Creating Custom Metrics for Datasets

**Step 1: Create a custom metrics module** (`custom_metrics/my_dataset_metrics.py`)
```python
"""
Custom metrics for my-dataset.

Each metric function receives a response dictionary with fields:
- response: Model's generated response  
- expected_output: Expected/reference response
- question_values: List of question field values
- success: Boolean indicating if generation was successful
- error: Error message if success is False

Each function should return a float between 0.0 and 1.0.
"""

def keyword_coverage(response_data):
    """Check if response covers expected keywords."""
    generated = response_data.get('response', '').lower()
    expected = response_data.get('expected_output', '').lower()
    
    if not response_data.get('success', False):
        return 0.0
    
    # Extract keywords from expected output
    expected_keywords = set(expected.split())
    if not expected_keywords:
        return 1.0
    
    # Count how many keywords appear in generated response
    covered_keywords = sum(1 for keyword in expected_keywords 
                          if keyword in generated)
    
    return covered_keywords / len(expected_keywords)

def response_length_ratio(response_data):
    """Compare response length to expected length."""
    generated = response_data.get('response', '')
    expected = response_data.get('expected_output', '')
    
    if not response_data.get('success', False):
        return 0.0
    
    gen_len = len(generated.split())
    exp_len = len(expected.split())
    
    if exp_len == 0:
        return 1.0 if gen_len == 0 else 0.0
    
    # Return ratio capped at 1.0 to avoid penalizing longer responses too much
    ratio = gen_len / exp_len
    return min(ratio, 1.0)

# Required: Registry dictionary
MY_DATASET_METRICS = {
    'keyword_coverage': keyword_coverage,
    'response_length_ratio': response_length_ratio
}
```

**Step 2: Update dataset configuration** (`configs/datasets.json`)
```json
{
  "my-dataset": {
    "download_link": "https://example.com/dataset.zip",
    "download_path": "my_dataset_folder", 
    "csv_file": "data.csv",
    "question_fields": ["input_text", "context"],
    "answer_field": "expected_output",
    "description": "My custom dataset",
    "evaluation_config": {
      "na_indicators": ["N/A", "not available", ""],
      "case_sensitive": true,
      "skip_empty_responses": false
    },
    "custom_metrics": {
      "module_path": "custom_metrics.my_dataset_metrics",
      "metrics_registry": "MY_DATASET_METRICS"
    }
  }
}
```

**Step 3: Create `custom_metrics/__init__.py`** (required for Python imports)
```python
# Empty file - just needs to exist for Python to treat directory as package
```

### Evaluation Configuration Options

Add an `evaluation_config` section to any dataset in `configs/datasets.json`:

```json
{
  "dataset_name": {
    "evaluation_config": {
      "na_indicators": ["N/A", "not annotatable", "", "unclear"],
      "case_sensitive": false,
      "skip_empty_responses": true
    }
  }
}
```

**Available Options:**

- **`na_indicators`** (list): Strings that mark unannotatable samples ( expected answers )
  - Examples: `["N/A", "not available", "", "unclear"]`
  - These samples are excluded from evaluation metrics
  - Case-insensitive matching after whitespace stripping

- **`case_sensitive`** (boolean): Whether text comparison is case-sensitive  
  - `false`: "Hello" matches "hello" (default)
  - `true`: "Hello" does not match "hello"

- **`skip_empty_responses`** (boolean): Whether to skip empty generated responses
  - `true`: Empty responses don't contribute to metrics (default)
  - `false`: Empty responses count as zero scores


## 💡 Example Workflows

### Prompt Development and Testing
```bash
# Preview how your prompt looks with real data
python main.py show-prompt --prompt gmeg_v1_basic --row 25

# Test different data points
python main.py show-prompt --prompt gmeg_v2_enhanced --row 100
python main.py show-prompt --prompt gmeg_v2_enhanced --row 200

# Compare prompt styles with same data
python main.py show-prompt --prompt gmeg_v4_minimal --row 42
python main.py show-prompt --prompt gmeg_v6_formal --row 42

# Run experiments after preview
python main.py run-baseline-exp --model gpt-4o-mini --prompt gmeg_v2_enhanced --size 50
```

### Model Comparison Study
```bash
# Preview prompts first
python main.py show-prompt --prompt gmeg_v1_basic

# Run multiple models with intelligent expansion
python main.py run-baseline-exp --model "gpt-4o-mini gemini-1.5-flash llama3.2-1b" --dataset gmeg --size 100

# Evaluate all results with custom metrics
python main.py evaluate

# Generate interactive comparison dashboard
python main.py plot --compare
```

### Few-shot vs Zero-shot Analysis
```bash
# Preview both modes
python main.py show-prompt --prompt gmeg_v1_basic        # zero-shot
python main.py show-prompt --prompt gmeg_few_shot        # few-shot

# Test both modes (system selects compatible prompts automatically)
python main.py run-baseline-exp --model gpt-4o-mini --mode zero-shot --dataset gmeg --size 50
python main.py run-baseline-exp --model gpt-4o-mini --mode few-shot --dataset gmeg --size 50

# Compare results
python main.py plot --compare
```

### Prompt Engineering Study  
```bash
# Preview different prompt styles
python main.py show-prompt --prompt gmeg_v1_basic --row 42
python main.py show-prompt --prompt gmeg_v2_enhanced --row 42  
python main.py show-prompt --prompt gmeg_v4_minimal --row 42

# Test multiple prompts (automatic compatibility validation)
python main.py run-baseline-exp --model gpt-4o-mini --dataset gmeg --prompt "gmeg_v1_basic gmeg_v2_enhanced gmeg_v4_minimal" --size 50

# Analyze prompt effectiveness
python main.py plot --compare
```

## 🔧 Dependencies

### Core Requirements
- `torch` - PyTorch for local models and embeddings
- `transformers` - Hugging Face transformers library
- `langchain-huggingface` - Embedding models
- `openai` - OpenAI API client  
- `google-genai` - Google Gemini API
- `anthropic` - Anthropic Claude API
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation
- `tqdm` - Progress bars
- `python-dotenv` - Environment variable management

### Optional (for Local Models)
- `unsloth` - Efficient local model loading with 4-bit quantization
- `trl` - Training and fine-tuning tools

### Installation
```bash
# Core functionality
pip install -r requirements.txt

# For local models (requires compatible GPU)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## 🔑 API Keys Configuration

Create a `.env` file from `.env.template`:
```bash
OPENAI_API_KEY=your-openai-key-here
GENAI_API_KEY=your-google-genai-key-here  
ANTHROPIC_API_KEY=your-anthropic-key-here
```

*Missing keys only disable the corresponding models - other functionality remains available*

## 📄 Output Files

### Structured Output Organization
```
outputs/
├── responses/baseline/
│   └── inference_baseline__gmeg__gpt-4o-mini__zero-shot__gmeg_v1_basic__50__0p1.json
├── evaluations/baseline/  
│   └── evaluation_baseline__gmeg__gpt-4o-mini__zero-shot__gmeg_v1_basic__50__0p1.json
└── plots/baseline/
    └── plot_baseline__gmeg__gpt-4o-mini__zero-shot__gmeg_v1_basic__50__0p1.html
```

### File Content
- **Inference files**: Complete experiment data with responses, metadata, and statistics
- **Evaluation files**: Comprehensive metrics including custom dataset-specific scores
- **Plot files**: Interactive HTML with navigation, hover details, and export options

## 🖥️ Hardware Requirements & Compatibility

### ✅ Works on Any System
- **API Models**: GPT, Gemini, Claude (only need internet + API keys)
- **Evaluation & Visualization**: All metrics and plotting functionality
- **Dataset Management**: Download, preparation, and analysis
- **Prompt Preview**: Template population with real data

### 🚀 GPU Recommended For Local Models
- **Minimum**: 6GB VRAM for smaller models (1B-3B parameters)
- **Recommended**: 12GB+ VRAM for larger models (7B+ parameters)  
- **CPU Fallback**: Available but 10-100x slower, requires 16GB+ RAM

### System Status Check
```bash
python main.py status

# Shows:
# ✅ GPU Available: 1 device(s)
# 📊 Total GPU Memory: 24.0 GB  
# 🚀 Can run local models: Yes
# 💾 RAM: 16.2 GB available of 32.0 GB
```

The system **automatically prevents** GPU-dependent operations on incompatible hardware while **preserving all other functionality**.

## 🔧 Troubleshooting

### Quick Diagnostics
```bash
python main.py status          # Check system capabilities
python main.py list-options    # Show available configurations  
python main.py help            # Show all available commands
```

### Common Issues

**Configuration Problems:**
```bash
# Missing or invalid configuration files
python main.py status  # Shows which configs are missing/invalid

# Invalid experiment arguments  
python main.py run-baseline-exp --model invalid-model --force  # Force mode shows details

# Test prompt templates
python main.py show-prompt --prompt gmeg_v1_basic  # Preview templates
```

**API Key Issues:**
```bash
# Check API key status
python main.py status  # Shows ✅/❌ for each API key

# Test with available models only
python main.py list-options  # Shows which models are actually available
```

**Local Model Issues:**
```bash
# Check GPU availability  
python main.py status  # Shows GPU memory and compatibility

# Install Unsloth if missing
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Use API models as fallback
python main.py run-baseline-exp --model "gpt-4o-mini gemini-1.5-flash"
```

**Dataset Issues:**
```bash
# Download datasets manually
python main.py download-datasets --dataset gmeg

# Check download status
python main.py list-options  # Shows dataset availability
```

**Prompt Template Issues:**
```bash
# Preview problematic prompts
python main.py show-prompt --prompt gmeg_v1_basic --row 0

# Test with different data
python main.py show-prompt --prompt gmeg_v2_enhanced --row 100

# Check prompt-dataset compatibility
python main.py list-options  # Shows prompt compatibility
```

### System Compatibility Matrix

| Feature | Any CPU | GPU (6GB+) | GPU (12GB+) | API Keys |
|---------|---------|------------|-------------|----------|
| API Models | ✅ | ✅ | ✅ | Required |
| Small Local Models (1-3B) | ⚠️ Slow | ✅ | ✅ | - |
| Large Local Models (7B+) | ❌ | ⚠️ Limited | ✅ | - |
| Evaluation & Plotting | ✅ | ✅ | ✅ | - |
| Custom Metrics | ✅ | ✅ | ✅ | - |
| Prompt Preview | ✅ | ✅ | ✅ | - |

*The system gracefully handles limitations and provides clear guidance for your specific setup*

## 📚 Advanced Features

### Prompt Template Development
The `show-prompt` command helps you develop and test prompts:
- **Real Data**: See how your prompts work with actual dataset examples
- **Row Selection**: Test specific edge cases or typical examples
- **Mode Comparison**: Compare zero-shot vs few-shot formatting
- **Template Validation**: Ensure field counts match dataset structure

### Intelligent Argument Resolution
The system automatically handles complex argument combinations:
- **Compatibility Validation**: Ensures prompts match datasets and modes
- **Smart Expansion**: Fills in missing arguments with compatible options
- **Conflict Resolution**: Provides clear error messages with suggestions
- **Force Mode**: Override compatibility checks when needed

### Custom Metrics Plugin System
- **Dataset-specific metrics**: Automatically loaded based on dataset configuration
- **Flexible architecture**: Easy to add new metrics for new datasets  
- **Metric validation**: Automatic type checking and error handling
- **Performance tracking**: All metrics cached and optimized

### Metadata-based Processing
- **Content-based detection**: Uses file content rather than filename parsing
- **Robust file handling**: Handles renamed or moved files gracefully
- **Cross-type compatibility**: Evaluation and plotting work with any experiment type
- **Version resilience**: Compatible with different file format versions

---

*For more information, see the inline documentation in each Python module or use the `help` command for detailed usage examples.*