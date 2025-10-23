# Math-Agent

# MathAI - LLM-Powered Mathematical Problem Solver

A sophisticated mathematical problem solver that combines multiple AI models to understand natural language math problems, validate generated code, and provide step-by-step solutions with explanations.

## 🎯 Features

- **Natural Language Understanding**: Processes complex word problems with contextual details
- **Question Filtering**: Extracts essential mathematical information from verbose questions
- **LLM Code Validation**: Uses AI to validate variable names and function calls for security
- **Safe Code Execution**: AST-level validation with sandboxed Python execution
- **Step-by-Step Explanations**: Provides detailed solution breakdowns
- **Unit Detection**: Automatically identifies and includes units in answers
- **Math Syntax Sanitization**: Converts human-readable math (^, implicit multiplication) to Python

## 🛠️ System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended)
  - Tesla T4 or better
  - RTX 3060 or better
  - A100/V100 for optimal performance
- **RAM**: Minimum 16GB system RAM (32GB recommended)
- **Storage**: 10GB free space for models

### Software
- **Python**: 3.8 or higher (3.10+ recommended)
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **Operating System**: Linux (Ubuntu 20.04+), Windows 10+, or macOS

## 📦 Installation

### Step 1: Clone or Download
```bash
# Create project directory
mkdir mathai-solver
cd mathai-solver

# Copy MathAI.ipynb and this README to the directory
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install PyTorch with CUDA
Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and select your configuration. Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## 🚀 Usage

### Running in Jupyter Notebook

1. **Launch Jupyter**:
   ```bash
   jupyter notebook MathAI.ipynb
   ```

2. **Execute cells sequentially**:
   - Cell 1: Install dependencies (if needed)
   - Cell 2: Import libraries
   - Cell 3: Load models (takes 2-3 minutes)
   - Cells 4-11: Define utility classes and functions
   - Cell 12: Initialize solver and start interactive session

### Running in Google Colab

1. Upload `MathAI.ipynb` to Google Colab
2. **Set GPU runtime**:
   - Runtime → Change runtime type → GPU (T4)
3. Run all cells sequentially
4. Models download automatically (one-time, ~6GB)

### Example Usage

```python
# After running all setup cells
question = "A shopkeeper bought 8 pencils at ₹5 each and 3 notebooks at ₹20 each. After a ₹10 discount, how much did he pay?"

result = solver.solve(question)

# Output includes:
# - Cleaned question
# - Solution plan
# - Step-by-step breakdown
# - Python code
# - Final answer with units
```

## 🏗️ Architecture

### Component Overview

```
┌─────────────────────────────────────────┐
│         User Question Input             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Question Filter (Qwen 2.5 1.5B)      │
│   - Removes contextual fluff            │
│   - Extracts core math problem          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Math Solver (Qwen 2.5 Math 1.5B)      │
│   - Generates solution plan             │
│   - Creates Python code                 │
│   - Provides step explanations          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      LLM Code Validator                 │
│   - Validates variable names            │
│   - Checks function calls               │
│   - Ensures security                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   AST Validator & Sanitizer            │
│   - Parses code syntax                  │
│   - Blocks dangerous operations         │
│   - Converts math notation              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Safe Sandboxed Execution             │
│   - Limited builtins                    │
│   - Math library only                   │
│   - Returns final result                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Formatted Output with Units           │
└─────────────────────────────────────────┘
```

### Models Used

1. **Qwen2.5-Math-1.5B-Instruct**: Specialized in mathematical reasoning and code generation
2. **Qwen2.5-1.5B-Instruct**: General-purpose model for question filtering and validation

Both models use 4-bit quantization for memory efficiency (~1.5GB per model instead of 6GB).

## 🔒 Security Features

### Multi-Layer Validation

1. **LLM-Based Validation**:
   - Validates variable names contextually
   - Checks function call appropriateness
   - Learns from question context

2. **AST-Level Blocking**:
   - Blocks imports, exec, eval
   - Prevents file operations
   - Disallows loops, classes, functions

3. **Sandboxed Execution**:
   - Limited builtins (abs, min, max, round, etc.)
   - Math library only
   - No access to system resources

## 📊 Performance

### Model Loading
- **First load**: 2-3 minutes (downloads models)
- **Subsequent loads**: 30-45 seconds
- **Total model size**: ~6GB on disk, ~3GB in VRAM

### Inference Speed
- **Question filtering**: 1-2 seconds
- **Solution generation**: 2-4 seconds
- **Code validation**: 1-2 seconds
- **Total per question**: 5-10 seconds

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size or use CPU fallback
torch.cuda.empty_cache()
```

### Model Download Issues
```python
# Use Hugging Face mirror
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### Import Errors
```bash
# Reinstall with --upgrade flag
pip install --upgrade transformers accelerate
```

### Code Validation Failures
- The LLM validator may reject unusual variable names
- Try rewording the question more clearly
- Check logs for specific rejected names

## 📝 Supported Math Problems

- Arithmetic operations (addition, subtraction, multiplication, division)
- Percentage calculations
- Simple interest and compound interest
- Geometry (area, perimeter, volume)
- Speed, distance, and time problems
- Profit and loss calculations
- Average and statistics
- Unit conversions

## 🎓 Example Problems

```python
# Example 1: Basic arithmetic with context
question = "Sarah bought 5 apples at $2 each and 3 oranges at $1.50 each. How much did she spend?"

# Example 2: Compound interest
question = "Calculate compound interest on $1000 at 5% per annum for 3 years."

# Example 3: Geometry
question = "Find the area of a circle with radius 7 cm."

# Example 4: Speed-distance-time
question = "A car travels 180 km in 3 hours. What is its average speed?"
```

## 🤝 Contributing

This is a notebook-based project. To contribute:
1. Fork or copy the notebook
2. Make improvements
3. Test with various math problems
4. Share your enhanced version

## 📄 License

This project uses open-source models from Hugging Face. Check individual model licenses:
- [Qwen2.5-Math License](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
- [Qwen2.5 License](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

## 🙏 Acknowledgments

- **Qwen Team** for the mathematical reasoning models
- **Hugging Face** for the transformers library
- **bitsandbytes** for efficient quantization

## 📧 Support

For issues or questions:
- Check the troubleshooting section
- Review model documentation on Hugging Face
- Verify your CUDA installation and GPU compatibility

---

**Note**: First run downloads ~6GB of model files. Ensure stable internet connection and sufficient storage space.
