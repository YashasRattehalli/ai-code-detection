# AI Code Detector

A machine learning system for detecting AI-generated code.

## Overview

This project provides a complete pipeline for training models to detect AI-generated code and using those models for inference. It uses a combination of code embeddings from Microsoft's UnixCoder model and custom code metrics to achieve high accuracy in distinguishing between human-written and AI-generated code samples.

## Project Structure

The project has been organized into an object-oriented, modular architecture:

```
ai_code_detector/
├── __init__.py
├── config.py                 # Configuration settings
├── train_pipeline.py         # Training pipeline implementation
├── inference_pipeline.py     # Inference pipeline implementation
├── core/
│   ├── __init__.py
│   └── code_detector.py      # Base functionality shared by both pipelines
├── models/
│   ├── __init__.py
│   ├── feature_extractor.py  # Code feature extraction
│   ├── unixcoder.py          # Code embedding generation
│   └── xgboost_classifier.py # XGBoost model implementation
└── api/                      # API implementation (if applicable)
```

## Features

- **Code Embedding Generation**: Uses Microsoft's UnixCoder model to create semantic embeddings of code.
- **Feature Extraction**: Extracts structural and quality metrics from code.
- **Cross-Validation**: Performs k-fold cross-validation for model evaluation.
- **Hyperparameter Tuning**: Uses Optuna for finding optimal model parameters.
- **Language Detection**: Automatically detects programming language from code or file extension.
- **Multi-Format Output**: Supports JSON and human-readable output formats.

## Usage

### Training a Model

To train a model with default settings:

```bash
python -m ai_code_detector.train_pipeline
```

With custom settings:

```bash
python -m ai_code_detector.train_pipeline \
  --dataset path/to/dataset.csv \
  --model-type xgboost \
  --tune-params \
  --n-folds 5
```

### Using the Model for Inference

To analyze one or more files:

```bash
python -m ai_code_detector.inference_pipeline path/to/file.py
```

With custom settings:

```bash
python -m ai_code_detector.inference_pipeline path/to/file.py \
  --model-type xgboost \
  --threshold 0.7 \
  --json
```

## Dependencies

- Python 3.8+
- PyTorch
- XGBoost
- Transformers (Hugging Face)
- NumPy
- Pandas
- scikit-learn
- Optuna (for hyperparameter tuning)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/code-detector.git
cd code-detector

# Install dependencies
pip install -r requirements.txt
``` 