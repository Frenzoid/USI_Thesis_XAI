# XAI Explanation Evaluation System

A comprehensive command-line system for evaluating Large Language Models' alignment with user study results in XAI (Explainable AI) explanation evaluation.

## 🚀 Features

- **Individual Command Structure**: Separate commands for inference, evaluation, and visualization
- **JSON-Based Configuration**: External configuration files for prompts, datasets, and models
- **Multi-Model Support**: Test both open-source models (via Unsloth) and API-based models
- **Flexible Experiment Types**: Baseline experiments (with structure for future masking/impersonation)
- **Temperature Control**: Configurable temperature parameter for generation
- **Comprehensive Evaluation**: Token-based metrics, semantic similarity, and dataset-specific evaluation
- **Advanced Visualization**: Interactive plots and comprehensive reports
- **Configuration Validation**: Automatic prompt-dataset compatibility checking
- **Descriptive Naming**: Clear experiment naming convention for easy organization

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for open-source models)
- API keys for external services (optional)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd xai-explanation-evaluation
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **For open-source model support (optional):**
```bash
# Install Unsloth for efficient model loading
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl peft accelerate bitsandbytes
```

## 🔑 API Keys Setup

Create a `.env` file in the project root:

```bash
# Copy the template
cp .env.template .env

# Edit with your API keys
nano .env
```

The `.env` file should contain:
```
OPENAI_API_KEY=your-openai-api-key
GENAI_API_KEY=your-google-genai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## ⚙️ Configuration

The system uses JSON files for configuration:

### 📝 Prompts (configs/prompts.json)
```json
{
  "gmeg_v1_basic": {
    "type": "baseline",
    "compatible_dataset": "gmeg",
    "template": "Your prompt template with {variables}",
    "description": "Prompt description"
  }
}
```

### 📊 Datasets (configs/datasets.json)
```json
{
  "gmeg": {
    "download_link": "https://github.com/grammarly/gmeg-exp/archive/refs/heads/main.zip",
    "download_path": "DS_GMEG_EXP",
    "csv_file": "gmeg-exp-main/data/3_annotated_data/full-scale_data/full_scale_annotated_full.csv",
    "question_fields": ["original", "revised"],
    "answer_field": "please_explain_the_revisions_write_na_if_not_annotatable",
    "description": "GMEG dataset for error correction explanation evaluation"
  }
}
```

### 🤖 Models (configs/models.json)
```json
{
  "llama3.2-1b": {
    "type": "local",
    "model_path": "unsloth/llama-3.2-1b-instruct-bnb-4bit",
    "max_tokens": 256,
    "finetuned": false
  },
  "gpt-4o-mini": {
    "type": "api",
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "max_tokens": 256
  }
}
```

## 🚀 Quick Start

### 1. Check System Status
```bash
python main.py status
python main.py list-options
```

### 2. Run Your First Experiment
```bash
# Single experiment
python main.py run-experiment --model llama3.2-1b --dataset gmeg --prompt gmeg_v1_basic --size 20 --temperature 0.1

# Multiple models
python main.py run-experiment --model "llama3.2-1b gpt-4o-mini" --dataset gmeg --prompt gmeg_v1_basic --size 50

# All available models and prompts
python main.py run-experiment --experiment-type baseline
```

### 3. Evaluate Results
```bash
# Evaluate specific experiment
python main.py evaluate --experiment baseline_gmeg_llama3.2-1b_gmeg_v1_basic_50_0p1

# Evaluate all baseline experiments
python main.py evaluate --experiment-type baseline
```

### 4. Generate Visualizations
```bash
# Individual plot
python main.py plot --experiment baseline_gmeg_llama3.2-1b_gmeg_v1_basic_50_0p1

# Comparison plots
python main.py plot --experiment "exp1,exp2,exp3" --compare

# All plots
python main.py plot --experiment-type baseline
```

## 📁 Project Structure

```
xai-explanation-evaluation/
├── configs/
│   ├── prompts.json          # Prompt templates
│   ├── datasets.json         # Dataset configurations
│   └── models.json           # Model configurations
├── outputs/
│   ├── responses/baseline/   # Inference results
│   ├── evaluations/baseline/ # Evaluation metrics
│   └── plots/baseline/      # Generated visualizations
├── datasets/                # Downloaded datasets
├── models_cache/            # Cached models
├── finetuned_models/        # Finetuned models (future)
├── logs/                    # System logs
├── main.py                  # CLI entry point
├── config.py                # Configuration management
├── experiment_runner.py     # Experiment orchestration
├── evaluator.py             # Evaluation runner
├── plotter.py               # Visualization runner
├── models.py                # Model management
├── dataset_manager.py       # Dataset handling
├── prompt_manager.py        # Prompt management
├── evaluation.py            # Evaluation framework
├── visualization.py         # Visualization framework
└── utils.py                 # Utility functions
```

## 🎯 Available Commands

### Inference Experiments
```bash
# Basic usage
python main.py run-experiment --model MODEL --dataset DATASET --prompt PROMPT

# Advanced options
python main.py run-experiment \
  --model "llama3.2-1b gpt-4o-mini gemini-1.5-flash" \
  --dataset gmeg \
  --prompt "gmeg_v1_basic gmeg_v2_enhanced" \
  --size 50 \
  --temperature 0.1 \
  --experiment-type baseline
```

### Evaluation
```bash
# Specific experiment
python main.py evaluate --experiment EXPERIMENT_NAME

# By experiment type
python main.py evaluate --experiment-type baseline

# All experiments
python main.py evaluate
```

### Visualization
```bash
# Individual plots
python main.py plot --experiment EXPERIMENT_NAME

# Comparison plots
python main.py plot --experiment "exp1,exp2,exp3" --compare

# All plots for experiment type
python main.py plot --experiment-type baseline
```

### Utilities
```bash
# List available options
python main.py list-options

# Check system status
python main.py status
```

## 🧪 Supported Models

### Open Source Models (via Unsloth)
- Llama 3.2 (1B, 3B)
- Llama 3 (8B)
- Mistral 7B
- Qwen2 7B
- Phi-3 (Mini, Medium)
- Gemma 7B
- CodeLlama 7B
- TinyLlama 1B

### API Models
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Google**: Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.0 Pro
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku

## 📝 Available Prompts

All prompts are designed for the GMEG dataset (grammatical error correction explanations):

- **`gmeg_v1_basic`**: Basic correction explanation prompt
- **`gmeg_v2_enhanced`**: Enhanced with categorization
- **`gmeg_v3_detailed`**: Detailed linguistic analysis
- **`gmeg_v4_minimal`**: Minimal, concise explanations
- **`gmeg_v5_pedagogical`**: Educational, teaching-focused
- **`gmeg_v6_formal`**: Formal academic analysis
- **`gmeg_v7_casual`**: Casual, conversational style
- **`gmeg_v8_comparative`**: Side-by-side comparison
- **`gmeg_few_shot`**: Few-shot learning with examples

## 📊 Evaluation Metrics

### Core Metrics
- **Exact Match**: Perfect string matching
- **Precision/Recall/F1**: Token overlap metrics
- **Jaccard Similarity**: Set-based similarity
- **Semantic Similarity**: Embedding-based cosine similarity

### GMEG-Specific Metrics
- **Bullet Point Ratio**: Format consistency measurement
- **Correction Terminology Recall**: Use of correction-specific terms
- **Structural Format Match**: Adherence to expected structure

## 📈 Visualization Outputs

The system automatically generates:
- **Individual Experiment Plots**: Bar charts with error bars
- **Metric Comparison Charts**: Multi-model comparisons
- **Radar Charts**: Multi-dimensional performance visualization
- **Interactive HTML Reports**: Comprehensive analysis with navigation

## 🔄 Experiment Types

### Baseline Experiments (✅ Implemented)
Standard LLM evaluation against human annotations:
```bash
python main.py run-experiment --experiment-type baseline
```

### Future Experiment Types (🚧 Structure Ready)
- **Masked Experiments**: Partial information experiments
- **Impersonation Experiments**: Demographic-specific evaluations

## 🎯 Example Workflows

### Complete Model Comparison Study
```bash
# 1. Run experiments with multiple models
python main.py run-experiment \
  --model "llama3.2-1b llama3.2-3b mistral-7b gpt-4o-mini gemini-1.5-flash" \
  --dataset gmeg \
  --prompt gmeg_v1_basic \
  --size 100 \
  --temperature 0.1

# 2. Evaluate all results
python main.py evaluate --experiment-type baseline

# 3. Generate comprehensive comparison plots
python main.py plot --experiment-type baseline
```

### Prompt Engineering Analysis
```bash
# Test multiple prompts with same model
python main.py run-experiment \
  --model llama3.2-1b \
  --dataset gmeg \
  --prompt "gmeg_v1_basic gmeg_v2_enhanced gmeg_v3_detailed gmeg_v4_minimal" \
  --size 50

# Evaluate and visualize prompt effects
python main.py evaluate --experiment-type baseline
python main.py plot --experiment-type baseline --compare
```

### Temperature Sensitivity Study
```bash
# Run same configuration with different temperatures
python main.py run-experiment --model llama3.2-1b --dataset gmeg --prompt gmeg_v1_basic --size 30 --temperature 0.0
python main.py run-experiment --model llama3.2-1b --dataset gmeg --prompt gmeg_v1_basic --size 30 --temperature 0.5
python main.py run-experiment --model llama3.2-1b --dataset gmeg --prompt gmeg_v1_basic --size 30 --temperature 1.0

# Compare results
python main.py evaluate --experiment-type baseline
python main.py plot --experiment "baseline_gmeg_llama3.2-1b_gmeg_v1_basic_30_0p0,baseline_gmeg_llama3.2-1b_gmeg_v1_basic_30_0p5,baseline_gmeg_llama3.2-1b_gmeg_v1_basic_30_1p0" --compare
```

## 🎨 File Naming Convention

Experiments use descriptive naming: `{experiment_type}_{dataset}_{model}_{prompt}_{size}_{temperature}`

Examples:
- `baseline_gmeg_llama3.2-1b_gmeg_v1_basic_50_0p1`
- `baseline_gmeg_gpt-4o-mini_gmeg_v2_enhanced_100_0p0`

## 🔧 Customization

### Adding New Prompts
Edit `configs/prompts.json`:
```json
{
  "my_custom_prompt": {
    "type": "baseline",
    "compatible_dataset": "gmeg",
    "template": "Your template with {original_text} and {revised_text}",
    "description": "Your custom prompt description"
  }
}
```

### Adding New Models
Edit `configs/models.json`:
```json
{
  "my_model": {
    "type": "local",  // or "api"
    "model_path": "path/to/model",
    "max_tokens": 256,
    "finetuned": false
  }
}
```

### Finetuned Models Support
Set `"finetuned": true` in model configuration. The system will look for models in `finetuned_models/` directory with `_finetuned` suffix.

## 🛠 Troubleshooting

### System Issues
```bash
python main.py status  # Check system health
```

### Configuration Problems
- Verify JSON syntax in config files
- Check prompt-dataset compatibility
- Ensure model availability

### Common Errors
- **Missing API keys**: Add to `.env` file
- **Model not found**: Check if Unsloth is installed for local models
- **Dataset download failed**: Check internet connection
- **Compatibility error**: Use `--force` to override validation

## 📊 Output Files

### Inference Results
`outputs/responses/baseline/inference_{experiment_name}.json`
- Model responses with metadata
- Processing times and error tracking
- Complete experiment configuration

### Evaluation Results
`outputs/evaluations/baseline/evaluation_{experiment_name}.json`
- Comprehensive metric calculations
- Statistical summaries
- Dataset-specific evaluations

### Visualization Files
`outputs/plots/baseline/plot_{experiment_name}.html`
- Interactive HTML visualizations
- Comparison charts and analysis
- Comprehensive experiment reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Unsloth team for efficient model loading
- Hugging Face for transformers and datasets
- OpenAI, Google, and Anthropic for API access
- The research community for datasets and evaluation methodologies