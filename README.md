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
  - DistilBERT-multilingual (v0.6803)
  - XLM-RoBERTa-base (v0.705)
  - BanglaBERT-large (csebuetnlp) (v0.7051)
- **LLMs with Adversarial Attacks** - Robustness testing using adversarial examples
- **LLMs with K-Fold Cross Validation** - K-fold CV for better model evaluation
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
    - FGM (Fast Gradient Method) - v1_0.7375
    - GAT (Gradient Adversarial Training) - v_0.7379
    - Simple FreeLB - v_0.7391
    - FreeLB (Free Large-Batch adversarial training) - v_0.7452
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
Best performing models based on validation scores:
- Subtask 1C: BanglaBERT with FreeLB (v_0.7452)
- Subtask 1A: BanglaBERT-large (v0.7051)

## Technologies Used
- **Transformers**: DistilBERT, XLM-RoBERTa, BanglaBERT
- **Deep Learning Frameworks**: PyTorch/TensorFlow (inferred from .ipynb files)
- **Adversarial Training**: FreeLB, FGM, GAT implementations
- **Evaluation**: K-Fold Cross Validation
- **Ensemble Methods**: Combining multiple models for improved performance

## File Organization
Each subtask folder contains:
- `Evaluation Phase/` - Main directory with all experimental approaches
- Individual approach folders containing Jupyter notebooks (.ipynb) with implementations
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