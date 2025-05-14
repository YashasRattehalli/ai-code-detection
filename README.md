# AI Code Detector

This project implements a binary classifier that detects AI-generated code using a combination of code embeddings from Microsoft's UnixCoder model and traditional code features.

## Overview

The model uses a two-step approach:
1. Create code embeddings using Microsoft's UnixCoder model
2. Train an XGBoost binary classifier using these embeddings along with static code features

## Features

- Uses UnixCoder embeddings to capture semantic meaning of code
- Combines embeddings with traditional code metrics for better accuracy
- Supports multiple programming languages (Python, Java, C++)
- Provides a simple CLI for analyzing individual files
- Config-driven architecture for easy customization
- Modular design for adding new models in the future
- Optimized for performance and memory efficiency

## Project Structure

```
.
├── data/                      # Data directory
│   ├── cache/                 # Cache for embeddings and models
│   └── dataset_head.csv       # Example dataset
├── models/                    # Saved model files
├── src/                       # Source code
│   ├── models/                # Model implementations
│   │   ├── __init__.py
│   │   └── xgboost_model.py   # XGBoost classifier implementation
│   ├── config.py              # Configuration settings
│   ├── detect_ai_code.py      # CLI utility for code detection
│   ├── inference_pipeline.py  # Inference pipeline
│   └── train_pipeline.py      # Training pipeline
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## Requirements

- Python 3.8+
- Dependencies listed in pyproject.toml

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/ai-code-detector.git
cd ai-code-detector
pip install -e .
```

## Usage

### Training a Model

Train the model using the provided training pipeline:

```bash
python src/train_pipeline.py --dataset path/to/dataset.csv [--model-type xgboost] [--load-embeddings] [--tune-params]
```

Options:
- `--dataset`: Path to the training dataset (required)
- `--model-type`: Type of model to train (default: xgboost)
- `--load-embeddings`: Use pre-computed embeddings if available
- `--tune-params`: Perform hyperparameter tuning

### Detecting AI-Generated Code

Detect AI-generated code using the CLI:

```bash
python src/detect_ai_code.py -f path/to/file1.py path/to/file2.py [--model xgboost] [--json]
```

Or detect from a string:

```bash
python src/detect_ai_code.py -c "def hello(): print('Hello, world!')" -l python
```

Or read from standard input:

```bash
cat file.py | python src/detect_ai_code.py --stdin -l python
```

Options:
- `-f, --files`: Files to analyze
- `-c, --code`: Code string to analyze
- `--stdin`: Read code from stdin
- `-l, --language`: Programming language of the code
- `-m, --model`: Model type to use (default: xgboost)
- `-j, --json`: Output results in JSON format
- `-t, --threshold`: Probability threshold for classification (default: 0.5)

## Extending the Project

### Adding a New Model

To add a new model:

1. Create a new file in the `src/models/` directory
2. Implement your model class with the required interface
3. Update the `MODEL_CONFIGS` dictionary in `src/config.py`
4. The model should work with the existing pipelines

### Customizing Model Behavior

Modify the configuration in `src/config.py` to customize:

- Model parameters
- Encoder settings
- Training parameters
- File paths
- Feature columns

## License

[MIT License](LICENSE) 