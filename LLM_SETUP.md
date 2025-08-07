# LLM Explainer Setup Guide

This guide explains how to set up and use the LLM explainer with actual Llama models.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Option 1: Using Ollama (Recommended for local development)

### Step 1: Install Ollama

Download and install Ollama from [https://ollama.ai/](https://ollama.ai/)

For macOS/Linux:
```bash
curl https://ollama.ai/install.sh | sh
```

### Step 2: Pull a Llama model

```bash
# Pull Llama 2 7B (recommended for most use cases)
ollama pull llama2

# Or pull other models:
ollama pull llama2:13b      # Larger, more capable
ollama pull codellama       # Code-focused
ollama pull mistral         # Alternative model
```

### Step 3: Start Ollama service

```bash
ollama serve
```

The service will run on `http://localhost:11434` by default.

### Step 4: Test the setup

```bash
# Test Ollama directly
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Explain network security",
  "stream": false
}'

# Test with the project
python test_llm_explainer.py
```

### Step 5: Run explanations with Ollama

```bash
python pipeline/run_experiments.py \
    --dataset nsl-kdd \
    --train_path data/nsl_kdd/KDDTrain+.txt \
    --test_path data/nsl_kdd/KDDTest+.txt \
    --explain llm \
    --llm_provider ollama \
    --llm_model llama2
```

## Option 2: Using Hugging Face Transformers

### Step 1: Install additional dependencies

The required packages are already in requirements.txt:
- transformers
- accelerate
- bitsandbytes (for 8-bit quantization)

### Step 2: Set up Hugging Face access (if using gated models)

For Llama 2 models, you need to:
1. Request access at [https://huggingface.co/meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
2. Create a HF token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Login: `huggingface-cli login`

### Step 3: Run with Hugging Face

```bash
python pipeline/run_experiments.py \
    --dataset nsl-kdd \
    --train_path data/nsl_kdd/KDDTrain+.txt \
    --test_path data/nsl_kdd/KDDTest+.txt \
    --explain llm \
    --llm_provider huggingface \
    --llm_model meta-llama/Llama-2-7b-chat-hf
```

### Available Hugging Face Models

- `meta-llama/Llama-2-7b-chat-hf` - Llama 2 7B Chat (recommended)
- `meta-llama/Llama-2-13b-chat-hf` - Llama 2 13B Chat (larger)
- `microsoft/DialoGPT-medium` - Smaller alternative for testing
- `huggingface/CodeBERTa-small-v1` - Code-focused alternative

## Usage Examples

### Basic usage with rules (default)
```bash
python demo.py \
    --dataset nsl-kdd \
    --train_path data/nsl_kdd/KDDTrain+.txt \
    --test_path data/nsl_kdd/KDDTest+.txt \
    --explain rules
```

### LLM explanations with Ollama
```bash
python demo.py \
    --dataset nsl-kdd \
    --train_path data/nsl_kdd/KDDTrain+.txt \
    --test_path data/nsl_kdd/KDDTest+.txt \
    --explain llm \
    --llm_provider ollama \
    --llm_model llama2
```

### Hybrid explanations (rules + LLM)
```bash
python demo.py \
    --dataset nsl-kdd \
    --train_path data/nsl_kdd/KDDTrain+.txt \
    --test_path data/nsl_kdd/KDDTest+.txt \
    --explain hybrid \
    --llm_provider ollama \
    --llm_model llama2
```

### Different input modes (ablation study)
```bash
# Label only
python demo.py ... --explain llm --ablation_llm_inputs label

# Label + features (no SHAP values)
python demo.py ... --explain llm --ablation_llm_inputs label+features

# Label + SHAP values (no raw values)
python demo.py ... --explain llm --ablation_llm_inputs label+shap

# Full information (default)
python demo.py ... --explain llm --ablation_llm_inputs full
```

## Troubleshooting

### Ollama Issues
- **Connection refused**: Make sure `ollama serve` is running
- **Model not found**: Run `ollama pull <model_name>` first
- **Slow responses**: Try a smaller model like `llama2:7b` instead of `llama2:13b`

### Hugging Face Issues
- **CUDA out of memory**: Reduce batch size or use CPU (`CUDA_VISIBLE_DEVICES=""`)
- **Model access denied**: Make sure you have access to gated models
- **Slow loading**: Models are downloaded on first use; this is normal

### General Issues
- **Import errors**: Make sure all requirements are installed
- **Out of memory**: Use smaller models or reduce batch sizes
- **Slow generation**: This is normal for large language models

## Performance Notes

- **Ollama**: Generally faster for inference, easier setup
- **Hugging Face**: More control, works offline after download
- **GPU**: Significantly faster if available (especially for HF)
- **Model size**: 7B models are usually sufficient for explanations

## Output

Explanations will be saved to the `outputs/` directory:
- `{dataset}_{model}_{explain}_explanations.json` - Generated explanations
- `{dataset}_{model}_{explain}_metrics.json` - Evaluation metrics
