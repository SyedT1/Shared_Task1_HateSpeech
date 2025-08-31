# Shared_Task1_HateSpeech

## Project Overview
This repository contains implementations for a hate speech detection shared task with three subtasks (1A, 1B, and 1C). The project explores various approaches including traditional machine learning, deep learning, large language models (LLMs), and adversarial training techniques.

## Repository Structure

### Subtask 1A
The subtask1A folder contains the following evaluation approaches:

#### Approaches Implemented:
- **Deep Learning Models** - Neural network architectures for hate speech classification
- **Ensembling with Attacks** - Ensemble methods combined with adversarial attacks
- **Large Language Models (LLMs)**
  - DistilBERT-multilingual (Macro F1: 0.6803)
  - XLM-RoBERTa-base (Macro F1: 0.705)
  - BanglaBERT-large (csebuetnlp) (Macro F1: 0.7051)
  - BanglaBERT (csebuetnlp) (Macro F1: 0.7074)
  - MuRIL-large-cased (Macro F1: 0.7102)
  - XLM-RoBERTa-large (Macro F1: 0.7281)
- **LLMs with Adversarial Attacks** - Robustness testing using adversarial examples
- **LLMs with K-Fold Cross Validation** - K-fold CV for better model evaluation
  - DistilBERT-multilingual with K-Fold (Macro F1: 0.6963)
  - XLM-RoBERTa-base with K-Fold (Macro F1: 0.7174)
  - BanglaBERT-large with K-Fold (Macro F1: 0.7313)
  - BanglaBERT with K-Fold (Macro F1: 0.7329)
- **LLMs with ML Model Ensembling** - Combining LLMs with traditional ML approaches
- **Traditional ML Models** - Classical machine learning algorithms

### Subtask 1B
The subtask1B folder mirrors the structure of 1A with similar approaches:

#### Approaches Implemented:
- **Deep Learning Models**
- **Ensembling with Attacks**
- **Large Language Models (LLMs)**
- **LLMs with Adversarial Attacks**
- **LLMs with K-Fold Cross Validation**
- **LLMs with ML Model Ensembling**
- **Traditional ML Models**

### Subtask 1C
The subtask1C folder contains more advanced implementations with adversarial training:

#### Approaches Implemented:
- **Deep Learning Models**
- **Ensembling with Attacks**
- **Large Language Models (LLMs)**
- **LLMs with Adversarial Attacks and K-Fold CV**
  - BanglaBERT (cse-buet-nlp) with multiple attack variants:
    - FGM (Fast Gradient Method) - Macro F1: 0.7375
    - GAT (Gradient Adversarial Training) - Macro F1: 0.7379
    - Simple FreeLB - Macro F1: 0.7391
    - FreeLB (Free Large-Batch adversarial training) - Macro F1: 0.7452
- **LLMs with K-Fold Cross Validation**
- **LLMs with ML Model Ensembling and K-Fold**
- **Traditional ML Models**

## Key Features

### Adversarial Training Methods
The project implements several adversarial training techniques to improve model robustness:
- **FGM (Fast Gradient Method)**: A simple and efficient adversarial training approach
- **GAT (Gradient Adversarial Training)**: Advanced gradient-based adversarial training
- **FreeLB**: Free Large-Batch adversarial training for improved generalization
- **Simple FreeLB**: A simplified version of FreeLB for easier implementation

### Model Performance
Best performing models based on Macro F1 scores:
- Subtask 1C: BanglaBERT with FreeLB (Macro F1: 0.7452)
- Subtask 1A: BanglaBERT with K-Fold CV (Macro F1: 0.7329)

## Technologies Used
- **Transformers**: DistilBERT, XLM-RoBERTa (base & large), BanglaBERT, MuRIL-large-cased
- **Deep Learning Frameworks**: PyTorch/TensorFlow (inferred from .ipynb files)
- **Adversarial Training**: FreeLB, FGM, GAT implementations
- **Evaluation**: K-Fold Cross Validation
- **Ensemble Methods**: Combining multiple models for improved performance

## File Organization
Each subtask folder contains:
- `Evaluation Phase/` - Main directory with all experimental approaches
- Individual approach folders with version naming format: `v{macro_f1_score}_{model_name}`
  - Example: `v0.7452_freelb` indicates Macro F1 score of 0.7452 using FreeLB model
- Jupyter notebooks (.ipynb) with complete implementation pipelines
- README files in evaluation phase directories

## Recent Updates
- Added results for BanglaBERT-large, DistilBERT, and XLM-RoBERTa-base models
- Implemented FreeLB adversarial training
- Completed adversarial attacks with BanglaBERT (cse-buet-nlp)
- Added comprehensive evaluation across all three subtasks

## Usage
Navigate to the specific subtask folder and approach directory to access the implementation notebooks. Each notebook contains the complete pipeline for training and evaluation of the respective model.

## Contributing
Feel free to explore different approaches and contribute improvements to the existing implementations.