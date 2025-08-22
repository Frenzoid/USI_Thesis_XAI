# XAI Explanation Evaluation System

A comprehensive command-line system for evaluating Large Language Models' alignment with user study results in XAI (Explainable AI) explanation evaluation.

## 🚀 Features

- **Multi-Dataset Support**: Work with any dataset by configuring field mappings in JSON
- **Automatic Dataset Download**: System downloads and manages datasets automatically
- **Flexible Field Mapping**: Configure question/answer fields for any dataset structure
- **Prompt-Dataset Compatibility**: Automatic validation of prompt and dataset compatibility
- **Multi-Model Support**: Test both open-source models (via Unsloth) and API-based models
- **Generic & Specific Prompts**: Use general prompts across datasets or dataset-specific prompts
- **Comprehensive Evaluation**: Token-based metrics, semantic similarity, and domain-specific evaluation
- **Automatic Visualization**: Generate interactive plots and comprehensive reports
- **Reproducible Experiments**: JSON-based configuration with automatic result caching
- **Extensive Logging**: Detailed logging of all operations for debugging and monitoring

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
# Install Unsloth for efficient fine-tuning
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl peft accelerate bitsandbytes
```

## 🔑 API Keys Setup

The system automatically loads environment variables from a `.env` file (recommended) or system environment variables.

### Option 1: Using .env file (Recommended)
```bash
# Copy the template
cp .env.template .env

# Edit the .env file with your API keys
nano .env  # or use your preferred editor
```

The `.env` file should contain:
```
OPENAI_API_KEY=your-actual-openai-api-key
GENAI_API_KEY=your-actual-google-genai-api-key
ANTHROPIC_API_KEY=your-actual-anthropic-api-key
```

### Option 2: Using system environment variables
```bash
# OpenAI (for GPT models)
export OPENAI_API_KEY="your-openai-api-key"

# Google GenAI (for Gemini models)
export GENAI_API_KEY="your-google-genai-api-key"

# Anthropic (for Claude models)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

**Note**: The system will automatically detect and load the `.env` file when the application starts. You'll see a confirmation message when the file is loaded successfully.

## 📊 Quick Start

### 1. Check System Status
```bash
python main.py status
```

### 2. Create Example Configuration Files
```bash
python main.py create-examples
```

### 3. Download Datasets
```bash
python main.py download --datasets-config datasets.json
```

### 4. Run Experiments
```bash
# Run all experiments
python main.py run --config experiments.json

# Run specific models only
python main.py run --config experiments.json --models llama3.2-1b gpt-4o-mini

# Force rerun existing experiments
python main.py run --config experiments.json --force
```

### 5. Generate Visualizations
```bash
python main.py visualize
```

## 📁 Project Structure

```
xai-explanation-evaluation/
├── main.py                 # CLI entry point
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── models.py              # Model management
├── prompts.py             # Prompt templates
├── dataset_manager.py     # Dataset handling
├── evaluation.py          # Evaluation framework
├── visualization.py       # Visualization generation
├── experiments.py         # Experiment orchestration
├── requirements.txt       # Python dependencies
├── datasets.json         # Dataset configuration
├── experiments.json      # Experiment configuration
├── datasets/             # Downloaded datasets
├── results/              # Experiment results
├── plots/                # Generated visualizations
├── models/               # Cached models
└── logs/                 # System logs
```

## 📊 Multi-Dataset Support

The system now supports multiple datasets with flexible field mapping:

### **Currently Supported Datasets:**
- **GMEG-EXP**: Grammar and fluency error correction explanations
- **XAI-FUNGI**: Fungal species classification explanations  
- **HateBRXplain**: Hate speech detection explanations
- **ExplanationHardness**: Explanation complexity analysis
- **ReframingHumanAI**: Human-AI explanation reframing

### **Dataset Configuration Structure:**
Each dataset is configured in `datasets.json` with:
```json
{
  "name": "GMEG-EXP",
  "field_mapping": {
    "question_fields": ["original", "revised"],
    "answer_field": "please_explain_the_revisions_write_na_if_not_annotatable",
    "question_template": "Original Text: {original}\n\nRevised Text: {revised}"
  },
  "compatible_prompts": ["gmeg_v1_basic", "gmeg_v2_enhanced", "general_explanation"],
  "task_type": "text_correction_explanation"
}
```

### **Key Features:**
- **Automatic Download**: Datasets are downloaded automatically when needed
- **Flexible Field Mapping**: Configure any column names as questions/answers
- **Prompt Compatibility**: System validates prompt-dataset compatibility
- **Generic Prompts**: Use `general_*` prompts across any dataset
- **Dataset-Specific Prompts**: Use specialized prompts for specific tasks

### **Adding New Datasets:**
1. Add configuration to `datasets.json`
2. Specify field mappings and download links
3. Create compatible prompts (optional)
4. Run experiments with any model

## 🎯 Available Commands

### List Available Resources
```bash
# List available models
python main.py list-models

# List available prompts
python main.py list-prompts

# List available datasets (shows download status)
python main.py list-datasets

# Check dataset-prompt compatibility
python main.py check-compatibility --dataset gmeg_exp --prompt gmeg_v1_basic

# Show full compatibility matrix
python main.py check-compatibility
```

### Download Datasets
```bash
python main.py download --datasets-config datasets.json
```

### Run Experiments
```bash
# Basic usage
python main.py run --config experiments.json

# Advanced usage
python main.py run --config experiments.json \
  --models llama3.2-1b mistral-7b \
  --prompts gmeg_v1_basic gmeg_v2_enhanced \
  --force \
  --no-viz
```

### Generate Visualizations
```bash
python main.py visualize
```

### System Status
```bash
python main.py status
```

## ⚙️ Configuration

### Datasets Configuration (`datasets.json`)
```json
[
  {
    "name": "GMEG-EXP",
    "link": "https://github.com/your-repo/gmeg-exp/archive/main.zip",
    "storage_folder": "DS_GMEG_EXP",
    "description": "GMEG dataset for error correction explanation evaluation"
  }
]
```

### Experiments Configuration (`experiments.json`)
```json
{
  "experiments": [
    {
      "experiment_name": "baseline_llama_basic",
      "model_name": "llama3.2-1b",
      "model_type": "open_source",
      "prompt_key": "gmeg_v1_basic",
      "sample_size": 50,
      "dataset_type": "gmeg",
      "dataset_name": "gmeg"
    }
  ]
}
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

### **Generic Prompts (work with any dataset):**
- `general_explanation`: General explanation prompt for any task
- `general_basic`: Basic general analysis prompt  
- `general_detailed`: Detailed general analysis prompt

### **GMEG-Specific Prompts:**
- `gmeg_v1_basic`: Basic correction explanation prompt
- `gmeg_v2_enhanced`: Enhanced with categorization
- `gmeg_v3_detailed`: Detailed linguistic analysis
- `gmeg_v4_minimal`: Minimal, concise explanations
- `gmeg_v5_pedagogical`: Educational, teaching-focused
- `gmeg_v6_formal`: Formal academic analysis
- `gmeg_v7_casual`: Casual, conversational style
- `gmeg_v8_comparative`: Side-by-side comparison
- `gmeg_few_shot`: Few-shot learning with examples

### **Dataset-Specific Prompts:**
- `fungi_v1_basic`: Fungal classification explanations
- `hate_speech_v1`: Hate speech detection explanations
- `hardness_v1`: Explanation complexity analysis
- `reframing_v1`: Explanation reframing analysis

### **Prompt-Dataset Compatibility:**
- Generic prompts work with any dataset
- Dataset-specific prompts only work with their target dataset
- Use `python main.py check-compatibility` to verify compatibility

## 📊 Evaluation Metrics

### Core Metrics
- **Exact Match**: Perfect string matching
- **Precision/Recall/F1**: Token overlap metrics
- **Jaccard Similarity**: Set-based similarity
- **Semantic Similarity**: Embedding-based cosine similarity

### Domain-Specific Metrics (GMEG)
- **Bullet Point Ratio**: Format consistency
- **Correction Terminology Recall**: Use of correction-specific terms
- **Structural Format Match**: Adherence to expected structure

## 📈 Visualization Outputs

The system automatically generates:
- **Metric Comparison Charts**: Bar charts comparing models across metrics
- **Radar Charts**: Multi-dimensional model performance comparison
- **Prompt Engineering Analysis**: Effects of different prompts
- **Processing Time Analysis**: Performance benchmarks
- **Comprehensive Reports**: Interactive HTML reports with all visualizations

## 🔧 Customization

### Adding Custom Prompts
```python
# In prompts.py
prompt_manager.add_prompt(
    key="custom_prompt",
    template="Your custom prompt template with {variables}",
    description="Description of your custom prompt"
)
```

### Adding Custom Metrics
```python
# In evaluation.py
def custom_metric(generated: str, expected: str) -> float:
    # Your custom evaluation logic
    return score

evaluator.register_custom_metric("custom_metric", custom_metric)
```

### Adding New Datasets
1. Update `datasets.json` with dataset information
2. Implement dataset loading in `datasets.py`
3. Add prompt preparation logic if needed

## 🐛 Troubleshooting

### Common Issues

1. **Environment Variables Not Loading**
   - Make sure you've copied `.env.template` to `.env`
   - Check that the `.env` file is in the same directory as `main.py`
   - Verify that `python-dotenv` is installed: `pip install python-dotenv`
   - You should see "✅ Loaded environment variables from .env" when starting the application

2. **Import Conflicts with Package Names**
   - **Problem**: `ImportError: cannot import name 'Dataset' from 'datasets'` 
   - **Cause**: Local file names conflict with popular Python packages
   - **Solution**: The project uses `dataset_manager.py` instead of `datasets.py` to avoid conflicts with the HuggingFace `datasets` package
   - **Quick Fix**: If you see this error, run: `python fix_imports.py`
   - **Manual Fix**: 
     ```bash
     # If you still have datasets.py, rename it:
     mv datasets.py dataset_manager.py
     # Update any imports in your code from 'datasets' to 'dataset_manager'
     ```
   - **Prevention**: Avoid naming local modules the same as popular packages (`datasets`, `transformers`, `torch`, `numpy`, etc.)

3. **CUDA Out of Memory**
   - Reduce sample sizes in experiments
   - Use smaller models (llama3.2-1b instead of mistral-7b)
   - Clear GPU memory between experiments

4. **API Rate Limits**
   - Add delays between API calls
   - Use smaller sample sizes
   - Check API quotas and billing

5. **Missing Dependencies**
   - Ensure all requirements are installed
   - Check Python version compatibility
   - Install optional dependencies for specific features

### Logging

Logs are automatically generated in the `logs/` directory:
- `main_YYYYMMDD.log`: Main application logs
- `models_YYYYMMDD.log`: Model loading and inference logs
- `evaluation_YYYYMMDD.log`: Evaluation process logs
- `visualization_YYYYMMDD.log`: Visualization generation logs

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the logs for detailed error information
- Ensure your environment meets all requirements

## 🙏 Acknowledgments

- Unsloth team for efficient model loading
- Hugging Face for transformers and datasets
- OpenAI, Google, and Anthropic for API access
- The research community for datasets and evaluation methodologies