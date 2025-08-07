# NLG + XAI for Intrusion Detection (NSL-KDD & UNSW-NB15)

End-to-end pipeline to:
1) preprocess NSL-KDD and UNSW-NB15,
2) train Autoencoder (PyTorch-based unsupervised) and XGBoost (supervised),
3) compute SHAP explanations,
4) generate rule-based and LLM-augmented natural-language explanations,
5) run ablations and evaluate with BLEU and optional LLM-as-judge.

## New Two-Stage Workflow

This codebase now separates model training from evaluation to allow efficient multiple experiments:

1. **Training Stage**: Train models once and save them to disk
2. **Evaluation Stage**: Load trained models and run multiple experiments/ablations

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Option 1: Unified Demo (Train + Evaluate)

```bash
# Example: NSL-KDD with XGBoost + rule-based explanations
python demo.py --dataset nsl-kdd --model xgb --explain rules \
  --train_path data/nsl_kdd/KDDTrain+.txt \
  --test_path data/nsl_kdd/KDDTest+.txt

# Example: NB15 with Autoencoder + LLM explanations (using Ollama)
python demo.py --dataset nb15 --model ae --explain llm \
  --train_path data/unsw_nb15/nb15_train.csv \
  --test_path data/unsw_nb15/nb15_test.csv \
  --llm_provider ollama --llm_model llama2
```

### Option 2: Separate Training and Evaluation

#### Step 1: Train Models

```bash
# Train XGBoost on NSL-KDD
python train_model.py --dataset nsl-kdd --model xgb \
  --train_path data/nsl_kdd/KDDTrain+.txt \
  --model_name nsl_kdd_xgb

# Train Autoencoder on NB15  
python train_model.py --dataset nb15 --model ae \
  --train_path data/unsw_nb15/nb15_train.csv \
  --model_name nb15_autoencoder
```

#### Step 2: Run Evaluations (Multiple Times)

```bash
# Evaluate with rule-based explanations
python eval_model.py --model_path trained_models/nsl_kdd_xgb.pkl \
  --test_path data/nsl_kdd/KDDTest+.txt \
  --explain rules

# Evaluate with LLM explanations
python eval_model.py --model_path trained_models/nsl_kdd_xgb.pkl \
  --test_path data/nsl_kdd/KDDTest+.txt \
  --explain llm --llm_provider ollama --llm_model llama2

# Evaluate with hybrid explanations + LLM-as-judge
python eval_model.py --model_path trained_models/nsl_kdd_xgb.pkl \
  --test_path data/nsl_kdd/KDDTest+.txt \
  --explain hybrid --judge --llm_provider huggingface
```

### Option 3: Automated Ablation Studies

```bash
# Run comprehensive ablation study
python run_ablations.py --model_path trained_models/nsl_kdd_xgb.pkl \
  --test_path data/nsl_kdd/KDDTest+.txt \
  --explain_methods rules llm hybrid \
  --llm_providers stub ollama \
  --llm_inputs label full \
  --enable_judge
```

## Features

- **Dataset Support**: NSL-KDD and UNSW-NB15 with dataset-aware feature glossaries
- **Models**: XGBoost (supervised) and PyTorch Autoencoder (unsupervised anomaly detection)
- **Explanations**: Rule-based, LLM-generated, and hybrid approaches
- **LLM Support**: Stub, Ollama, and HuggingFace providers with customizable models
- **Evaluation**: SHAP fidelity, BLEU scores, and LLM-as-judge evaluation
- **Efficient Workflow**: Train once, evaluate multiple times without retraining

## Directory Structure

- `train_model.py` - Train and save models
- `eval_model.py` - Load models and run evaluations  
- `run_ablations.py` - Automated ablation studies
- `demo.py` - Unified training + evaluation wrapper
- `trained_models/` - Saved model files and metadata
- `outputs/` - Evaluation results and explanations
- `explain/` - Explanation generation modules
- `models/` - Model implementations
- `data_loaders/` - Data loading and preprocessing
- `eval/` - Evaluation metrics and LLM-as-judge

## Configuration

Configuration is managed via `configs/defaults.yaml`. See `LLM_SETUP.md` for LLM setup instructions.
