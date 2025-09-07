# Shared_Task1_HateSpeech

## Project Overview
This repository contains implementations for a hate speech detection shared task with three subtasks (1A, 1B, and 1C). The project explores various approaches including traditional machine learning, deep learning, large language models (LLMs), and adversarial training techniques.

## Repository Structure

### Subtask 1A
-------------
The subtask1A folder contains comprehensive evaluation approaches:

#### Approaches Implemented:

##### **Deep Learning Models**
- BiLSTM - F1 Score: 56.25%
- LSTM with Attention - F1 Score: 55.18%

##### **Large Language Models (LLMs)**
- XLM-RoBERTa-large - F1 Score: 72.81%
- MuRIL-large-cased - F1 Score: 71.02%
- BanglaBERT (csebuetnlp) - F1 Score: 70.74%
- BanglaBERT-large (csebuetnlp) - F1 Score: 70.51%
- XLM-RoBERTa-base - F1 Score: 70.50%
- DistilBERT-multilingual - F1 Score: 68.03%

##### **LLMs with K-Fold Cross Validation**
- MuRIL-large-cased with K-Fold - F1 Score: 73.61%
- XLM-RoBERTa-large with K-Fold - F1 Score: 73.45%
- BanglaBERT with K-Fold - F1 Score: 73.29%
- BanglaBERT-large with K-Fold - F1 Score: 73.13%
- XLM-RoBERTa-base with K-Fold - F1 Score: 71.74%
- DistilBERT-multilingual with K-Fold - F1 Score: 69.63%

##### **K-Fold with Normalizer**
- BanglaBERT with Normalizer - F1 Score: 74.32%

##### **LLMs with K-Fold CV and Adversarial Attacks**
- BanglaBERT with FGM (Fast Gradient Method) - F1 Score: 73.61%
- BanglaBERT with AWP (Adversarial Weight Perturbation) - F1 Score: 72.61%

##### **K-Fold + Adversarial Attacks + Normalizer**
- BanglaBERT with FGM + Normalizer - F1 Score: 74.88% ⭐ (Best for Subtask 1A)

##### **Additional Approaches**
- Ensembling with Attacks
- LLMs with ML Model Ensembling
- Traditional ML Models

### Subtask 1B
-------------
The subtask1B folder contains similar approaches with the following results:

#### Approaches Implemented:

##### **Large Language Models (LLMs)**
- BanglaBERT - F1 Score: 72.09%
- MuRIL-large-cased - F1 Score: 71.93%
- XLM-RoBERTa-large - F1 Score: 71.38%

##### **LLMs with K-Fold Cross Validation**
- MuRIL-large-cased with K-Fold - F1 Score: 74.96% ⭐ (Best for Subtask 1B)
- BanglaBERT with K-Fold - F1 Score: 73.69%
- XLM-RoBERTa-large with K-Fold (in progress)

##### **K-Fold with Normalizer**
- BanglaBERT with Normalizer - F1 Score: 74.72%
- MuRIL-case-bert with Normalizer - F1 Score: 74.48%

##### **Additional Approaches**
- Deep Learning Models
- Ensembling with Attacks
- LLMs with Adversarial Attacks
- LLMs with ML Model Ensembling
- Traditional ML Models

### Subtask 1C
-------------
The subtask1C folder focuses on advanced adversarial training implementations:

#### Approaches Implemented:

##### **LLMs with Adversarial Attacks and K-Fold CV**
All using BanglaBERT (cse-buet-nlp) with different adversarial techniques:
- BanglaBERT with FreeLB - F1 Score: 74.52% ⭐ (Best for Subtask 1C)
- BanglaBERT with Simple FreeLB - F1 Score: 73.91%
- BanglaBERT with GAT - F1 Score: 73.79%
- BanglaBERT with FGM - F1 Score: 73.75%

##### **Additional Approaches**
- Deep Learning Models
- Ensembling with Attacks
- Standard LLMs
- LLMs with K-Fold Cross Validation
- LLMs with ML Model Ensembling and K-Fold
- Traditional ML Models

## Key Features

### Adversarial Training Methods
The project implements several adversarial training techniques to improve model robustness:
- **FGM (Fast Gradient Method)**: A simple and efficient adversarial training approach
- **AWP (Adversarial Weight Perturbation)**: Adversarial training that perturbs model weights for improved robustness
- **GAT (Geometry-Aware Adversarial Training)**: Advanced geometry-aware adversarial training
- **FreeLB**: Free Large-Batch adversarial training for improved generalization
- **Simple FreeLB**: A simplified version of FreeLB for easier implementation

### Training Enhancements
- **K-Fold Cross Validation**: Improves model evaluation and reduces overfitting
- **Normalizer Techniques**: Text normalization for better performance on Bangla text
- **Ensemble Methods**: Combining multiple models for improved predictions

## Model Performance Summary

### Best Performing Models by Subtask:
| Subtask | Model | F1 Score | Technique |
|---------|-------|----------|-----------|
| **1A** | BanglaBERT | 74.88% | K-Fold + FGM + Normalizer |
| **1B** | MuRIL-large-cased | 74.96% | K-Fold Cross Validation |
| **1C** | BanglaBERT | 74.52% | FreeLB Adversarial Training |

### Key Insights:
- **K-Fold Cross Validation** consistently improves performance by 1-3% across all models
- **Normalizer techniques** provide additional 0.5-1% improvement for Bangla text
- **Adversarial training** (especially FreeLB and FGM) enhances model robustness
- **Transformer models** significantly outperform traditional deep learning approaches (15-20% improvement)
- **MuRIL and BanglaBERT** are the most effective models for Bangla hate speech detection

## Technologies Used
- **Transformers**: DistilBERT, XLM-RoBERTa (base & large), BanglaBERT, MuRIL-large-cased
- **Deep Learning Frameworks**: PyTorch/TensorFlow
- **Adversarial Training**: FreeLB, FGM, AWP, GAT implementations
- **Evaluation**: K-Fold Cross Validation, Macro F1 scoring
- **Text Processing**: Bangla text normalizers
- **Ensemble Methods**: Model averaging and voting classifiers

## File Organization
Each subtask folder contains:
- `Developmental Phase/` - Initial experiments and model development
- `Evaluation Phase/` - Final implementations and evaluations
- Model directories follow naming convention: `v{f1_score}_{model_name}`
  - Example: `v0.7488_banglabert-fgm` = 74.88% F1 score using BanglaBERT with FGM
- Jupyter notebooks (.ipynb) with complete implementation pipelines

## Recent Updates
- Achieved 74.88% F1 score on Subtask 1A using BanglaBERT with FGM + Normalizer
- Added comprehensive K-Fold implementations across all subtasks
- Implemented multiple adversarial training techniques (FGM, AWP, FreeLB, GAT)
- Added normalizer techniques for improved Bangla text processing
- Completed evaluation of MuRIL-large-cased achieving best performance on Subtask 1B

## Usage
Navigate to the specific subtask folder and approach directory to access the implementation notebooks. Each notebook contains the complete pipeline for training and evaluation of the respective model.

## Performance Trends
1. **Base → K-Fold**: Average improvement of ~2-3%
2. **K-Fold → K-Fold + Normalizer**: Additional ~0.5-1% improvement
3. **K-Fold → K-Fold + Adversarial**: Variable improvement (0.5-1.5%)
4. **Combined techniques** (K-Fold + Adversarial + Normalizer): Best overall performance

## Contributing
Feel free to explore different approaches and contribute improvements to the existing implementations.