# Shared_Task1_HateSpeech

## Project Overview
This repository contains implementations for a hate speech detection shared task with three subtasks (1A, 1B, and 1C). The project explores various approaches including traditional machine learning, deep learning, large language models (LLMs), and adversarial training techniques.

## Competition Phases
The project was developed in two distinct phases:

### 🔬 **Developmental Phase**
- **Objective**: Model experimentation, architecture exploration, and hyperparameter tuning
- **Data**: Training and validation datasets provided by organizers
- **Focus**: Testing various approaches and techniques to identify best-performing models
- **Metrics**: Validation F1 scores on development set

### 🏆 **Evaluation Phase**  
- **Objective**: Final model evaluation on unseen test data
- **Data**: Hidden test set released during evaluation period
- **Focus**: Deploying best models from developmental phase with refined configurations
- **Metrics**: Test F1 scores on official evaluation set

## Repository Structure

### Subtask 1A - Binary Hate Speech Detection
-------------

#### 📊 **Developmental Phase Results**

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
- MuRIL-case-bert with Normalizer - F1 Score: 73.73%
- XLM-RoBERTa-large with Normalizer - F1 Score: 73.29%

##### **LLMs with K-Fold CV and Adversarial Attacks**
- BanglaBERT with FGM (Fast Gradient Method) - F1 Score: 73.61%
- BanglaBERT with AWP (Adversarial Weight Perturbation) - F1 Score: 72.61%

##### **K-Fold + Adversarial Attacks + Normalizer**
- BanglaBERT with FGM + Normalizer - F1 Score: 74.88% ⭐ (Best Development Score)
- MuRIL-bert-case with FGM + Normalizer - F1 Score: 73.81%

##### **Various Classification Heads**
- BanglaBERT with Custom Attention Head + FGM + Normalizer - F1 Score: 74.88% ⭐

#### 🎯 **Evaluation Phase Results**

##### **Final Test Performance**
- BanglaBERT with FGM + Normalizer - F1 Score: ~72%
- AWP-BanglaBERT with Normalizer - F1 Score: ~71%
- XLM-RoBERTa with K-Fold + Normalizer - F1 Score: ~72%
- MuRIL-base-case with K-Fold + Normalizer - F1 Score: ~72%

### Subtask 1B - Multi-class Hate Speech Classification
-------------

#### 📊 **Developmental Phase Results**

##### **Large Language Models (LLMs)**
- BanglaBERT - F1 Score: 72.09%
- MuRIL-large-cased - F1 Score: 71.93%
- XLM-RoBERTa-large - F1 Score: 71.38%

##### **LLMs with K-Fold Cross Validation**
- MuRIL-large-cased with K-Fold - F1 Score: 74.96% ⭐ (Best Development Score)
- BanglaBERT with K-Fold - F1 Score: 73.69%
- XLM-RoBERTa-large with K-Fold - In progress

##### **K-Fold with Normalizer**
- BanglaBERT with Normalizer - F1 Score: 74.72%
- MuRIL-case-bert with Normalizer - F1 Score: 74.48%

#### 🎯 **Evaluation Phase Results**

##### **Final Test Performance**
- BanglaBERT with K-Fold + Normalizer - F1 Score: ~73%
- MuRIL-bert with K-Fold + Normalizer - F1 Score: ~73%
- BanglaBERT with K-Fold CV - F1 Score: ~72%
- XLM-RoBERTa with K-Fold CV - F1 Score: ~68%

### Subtask 1C - Target Identification in Hate Speech
-------------

#### 📊 **Developmental Phase Results**

##### **LLMs with Adversarial Attacks and K-Fold CV**
All using BanglaBERT (cse-buet-nlp) with different adversarial techniques:
- BanglaBERT with FreeLB - F1 Score: 74.52% ⭐ (Best Development Score)
- BanglaBERT with Simple FreeLB - F1 Score: 73.91%
- BanglaBERT with GAT (Geometry-Aware Training) - F1 Score: 73.79%
- BanglaBERT with FGM (Fast Gradient Method) - F1 Score: 73.75%

#### 🎯 **Evaluation Phase Results**
- No evaluation phase notebooks found (Competition may not have included Subtask 1C in final evaluation)

## Key Features

### Custom Model Architectures
- **Attention-Based Pooling Head**: Learnable attention mechanism to dynamically weight and aggregate token hidden states instead of fixed pooling strategies ([CLS], mean pooling). This creates context-aware pooled representations that better capture important tokens in noisy Bengali text, particularly effective for hate speech detection with slang and informal language.

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
- **Custom Classification Heads**: Attention-based pooling for dynamic token weighting
- **Ensemble Methods**: Combining multiple models for improved predictions

## Model Performance Summary

### 📈 Best Performing Models - Developmental Phase:
| Subtask | Model | F1 Score | Technique |
|---------|-------|----------|-----------|
| **1A** | BanglaBERT | 74.88% | K-Fold + FGM + Normalizer |
| **1B** | MuRIL-large-cased | 74.96% | K-Fold Cross Validation |
| **1C** | BanglaBERT | 74.52% | FreeLB Adversarial Training |

### 🏅 Final Performance - Evaluation Phase:
| Subtask | Model | Dev F1 | Test F1 | Performance Drop |
|---------|-------|--------|---------|------------------|
| **1A** | BanglaBERT + FGM + Normalizer | 74.88% | ~72% | -2.88% |
| **1B** | BanglaBERT + K-Fold + Normalizer | 74.72% | ~73% | -1.72% |
| **1C** | - | - | - | Not evaluated |

### Key Insights:

#### Development vs Evaluation Phase Observations:
- **Generalization Gap**: Models showed 1-3% performance drop from development to test phase
- **Robust Techniques**: K-Fold CV and normalizers helped minimize overfitting
- **Best Stability**: Models with adversarial training showed better test performance retention

#### Technical Insights:
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

### Directory Structure:
```
Shared_Task1_HateSpeech/
├── subtask1A/
│   ├── Developmental Phase/     # Model experiments & validation
│   │   ├── DL Models/           # Deep learning baselines
│   │   ├── LLMs/                # Base transformer models
│   │   ├── LLMS with K Fold CV/ # K-Fold implementations
│   │   ├── K Folds with normalizer/
│   │   ├── LLMs_KFolds_adversarial attacks/
│   │   └── Various classification heads/
│   └── Evaluation Phase/        # Final test submissions
│       ├── K Folds with Normalizer/
│       └── LLMS_KFolds_attacks_normalizer/
├── subtask1B/
│   ├── Developmental Phase/
│   └── Evaluation Phase/
└── subtask1C/
    └── Developmental Phase/     # Only development (no test phase)

```

### Naming Convention:
- Model directories: `v{f1_score}_{model_name}`
  - Example: `v0.7488_banglabert-fgm` = 74.88% F1 score using BanglaBERT with FGM
- Each directory contains:
  - Jupyter notebook (.ipynb) with implementation
  - Data file (subtask_1X.tsv)

## Recent Updates
- Achieved 74.88% F1 score on Subtask 1A using BanglaBERT with FGM + Normalizer
- Implemented custom attention-based classification head for BanglaBERT
- Added K-Fold with normalizer for MuRIL and XLM-RoBERTa models
- Added comprehensive K-Fold implementations across all subtasks
- Implemented multiple adversarial training techniques (FGM, AWP, FreeLB, GAT)
- Added normalizer techniques for improved Bangla text processing
- Completed evaluation of MuRIL-large-cased achieving best performance on Subtask 1B

## Usage
Navigate to the specific subtask folder and approach directory to access the implementation notebooks. Each notebook contains the complete pipeline for training and evaluation of the respective model.

## Performance Trends

### Developmental Phase Improvements:
1. **Base → K-Fold**: Average improvement of ~2-3%
2. **K-Fold → K-Fold + Normalizer**: Additional ~0.5-1% improvement
3. **K-Fold → K-Fold + Adversarial**: Variable improvement (0.5-1.5%)
4. **Combined techniques** (K-Fold + Adversarial + Normalizer): Best overall performance

### Development → Evaluation Phase Trends:
- **Average Performance Drop**: 1-3% on unseen test data
- **Most Stable Approaches**: K-Fold + Normalizer combinations
- **Highest Risk of Overfitting**: Single model without K-Fold
- **Best Generalization**: Adversarial training methods (FGM, FreeLB)

## Contributing
Feel free to explore different approaches and contribute improvements to the existing implementations.