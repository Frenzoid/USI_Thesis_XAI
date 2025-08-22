#!/usr/bin/env python3
"""
Setup script for XAI Explanation Evaluation System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA available: {gpu_name} ({memory_gb:.1f} GB)")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower for open-source models)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check CUDA")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "datasets",
        "results", 
        "models",
        "plots",
        "plots/individual_experiments",
        "plots/comparative_analysis", 
        "plots/comprehensive_reports",
        "plots/model_comparisons",
        "plots/prompt_comparisons",
        "checkpoints",
        "logs"
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    return True

def check_dependencies():
    """Check if dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        "pandas",
        "numpy", 
        "torch",
        "transformers",
        "plotly",
        "scikit-learn",
        "sentence-transformers",
        "tqdm",
        "dotenv"  # python-dotenv
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "dotenv":
                import dotenv
            else:
                __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_import_conflicts():
    """Check for import conflicts with popular packages"""
    print("🔍 Checking for import conflicts...")
    
    # Check for common conflicting file names
    conflicting_files = {
        'datasets.py': 'Should be dataset_manager.py to avoid HuggingFace datasets conflict',
        'transformers.py': 'Conflicts with HuggingFace transformers package',
        'torch.py': 'Conflicts with PyTorch package',
        'numpy.py': 'Conflicts with NumPy package'
    }
    
    conflicts_found = []
    
    for filename, issue in conflicting_files.items():
        if os.path.exists(filename):
            conflicts_found.append((filename, issue))
    
    if conflicts_found:
        print("❌ Import conflicts detected:")
        for filename, issue in conflicts_found:
            print(f"   • {filename}: {issue}")
        
        print("\n💡 Run 'python fix_imports.py' to fix these issues")
        return False
    else:
        print("   ✅ No import conflicts found")
        return True
    """Check optional dependencies"""
    print("\n🔧 Checking optional dependencies...")
    
    optional_packages = {
        "unsloth": "For open-source model fine-tuning",
        "openai": "For OpenAI API models",
        "google.generativeai": "For Google GenAI models",
        "anthropic": "For Anthropic Claude models"
    }
    
    for package, description in optional_packages.items():
        try:
            if package == "google.generativeai":
                import google.generativeai
            else:
                __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ⚠️  {package} - {description}")

def create_env_template():
    """Create .env template file"""
    env_template = """# XAI Explanation Evaluation System - Environment Variables

# OpenAI API (for GPT models)
OPENAI_API_KEY=your-openai-api-key-here

# Google GenAI API (for Gemini models)  
GENAI_API_KEY=your-google-genai-api-key-here

# Anthropic API (for Claude models)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Custom configuration
# CUDA_VISIBLE_DEVICES=0
# HF_HOME=./models
"""
    
    if not os.path.exists(".env"):
        with open(".env.template", "w") as f:
            f.write(env_template)
        print("📝 Created .env.template file")
        print("   Copy to .env and add your API keys")
    else:
        print("📝 .env file already exists")

def test_system():
    """Test basic system functionality"""
    print("\n🧪 Testing system functionality...")
    
    try:
        # Test CLI help
        result = subprocess.run([sys.executable, "main.py", "--help"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("   ✅ CLI interface working")
        else:
            print("   ❌ CLI interface error")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("   ⚠️  CLI test timed out")
        return False
    except Exception as e:
        print(f"   ❌ CLI test failed: {e}")
        return False
    
    try:
        # Test system status
        result = subprocess.run([sys.executable, "main.py", "status"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("   ✅ System status check working")
        else:
            print("   ❌ System status check error")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("   ⚠️  System status test timed out")
        return False
    except Exception as e:
        print(f"   ❌ System status test failed: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 XAI Explanation Evaluation System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check for import conflicts
    if not check_import_conflicts():
        print("\n❌ Setup blocked - import conflicts detected")
        print("Please fix import conflicts first, then re-run setup")
        return 1
    
    # Create directories
    if not create_directories():
        return 1
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup incomplete - missing required dependencies")
        print("Please install requirements: pip install -r requirements.txt")
        return 1
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Check CUDA
    check_cuda()
    
    # Create environment template
    create_env_template()
    
    # Test system
    if not test_system():
        print("\n⚠️  System tests failed - please check the installation")
        return 1
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. 🔑 Set up API keys:")
    print("   cp .env.template .env")
    print("   # Edit .env file with your actual API keys")
    print("2. 📄 Create example configs: python main.py create-examples")
    print("3. 📥 Download datasets: python main.py download --datasets-config datasets.json")
    print("4. 🧪 Run experiments: python main.py run --config experiments.json")
    print("5. 📊 Check system status: python main.py status")
    print("\n📚 For more information, see README.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())