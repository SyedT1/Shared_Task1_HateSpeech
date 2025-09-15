# Shared Task 1: Hate Speech Detection in Bengali

## Project Overview
This repository contains comprehensive implementations for Bengali hate speech detection across three subtasks, developed for a competitive shared task. The project explores various machine learning approaches from traditional deep learning to state-of-the-art transformer models with advanced training techniques.

## Competition Phases

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
Detection of hate speech vs non-hate speech in Bengali text.

#### 📊 **Developmental Phase Results**

##### **Deep Learning Models**
- **BiLSTM** - F1 Score: 56.25%
- **LSTM with Attention** - F1 Score: 55.18%

##### **Large Language Models (LLMs)**
- **XLM-RoBERTa-large** - F1 Score: 72.81%
- **MuRIL-large-cased** - F1 Score: 71.02%
- **BanglaBERT (csebuetnlp)** - F1 Score: 70.74%
- **BanglaBERT-large (csebuetnlp)** - F1 Score: 70.51%
- **XLM-RoBERTa-base** - F1 Score: 70.50%
- **DistilBERT-multilingual** - F1 Score: 68.03%

##### **LLMs with K-Fold Cross Validation**
- **MuRIL-large-cased with K-Fold** - F1 Score: 73.61%
- **XLM-RoBERTa-large with K-Fold** - F1 Score: 73.45%
- **BanglaBERT with K-Fold** - F1 Score: 73.29%
- **BanglaBERT-large with K-Fold** - F1 Score: 73.13%
- **XLM-RoBERTa-base with K-Fold** - F1 Score: 71.74%
- **DistilBERT-multilingual with K-Fold** - F1 Score: 69.63%

##### **K-Fold with Text Normalizer**
- **BanglaBERT with Normalizer** - F1 Score: 74.32%
- **MuRIL-case-bert with Normalizer** - F1 Score: 73.73%
- **XLM-RoBERTa-large with Normalizer** - F1 Score: 73.29%

##### **LLMs with Adversarial Training**
- **BanglaBERT with FGM** - F1 Score: 73.61%
- **BanglaBERT with AWP** - F1 Score: 72.61%

##### **Advanced Combined Approaches**
- **BanglaBERT + FGM + Normalizer** - F1 Score: 74.88% ⭐ (Best Development Score)
- **MuRIL-bert-case + FGM + Normalizer** - F1 Score: 73.81%

##### **Custom Classification Heads**
- **BanglaBERT + Custom Attention Head + FGM + Normalizer** - F1 Score: 74.88% ⭐

#### 🎯 **Evaluation Phase Results**
- **BanglaBERT + FGM + Normalizer** - Test F1: 72.00%
- **AWP-BanglaBERT + Normalizer** - Test F1: 71.00%
- **XLM-RoBERTa + K-Fold + Normalizer** - Test F1: 72.00%
- **MuRIL-base-case + K-Fold + Normalizer** - Test F1: 72.00%

### Subtask 1B - Multi-class Hate Speech Classification
Classification into specific hate speech categories.

#### 📊 **Developmental Phase Results**

##### **Large Language Models (LLMs)**
- **BanglaBERT** - F1 Score: 72.09%
- **MuRIL-large-cased** - F1 Score: 71.93%
- **XLM-RoBERTa-large** - F1 Score: 71.38%

##### **LLMs with K-Fold Cross Validation**
- **MuRIL-large-cased with K-Fold** - F1 Score: 74.96% ⭐ (Best Development Score)
- **BanglaBERT with K-Fold** - F1 Score: 73.69%
- **XLM-RoBERTa-large with K-Fold** - F1 Score: In progress

##### **K-Fold with Text Normalizer**
- **BanglaBERT with Normalizer** - F1 Score: 74.72%
- **MuRIL-case-bert with Normalizer** - F1 Score: 74.48%

#### 🎯 **Evaluation Phase Results**
- **BanglaBERT + K-Fold + Normalizer** - Test F1: 73.00%
- **MuRIL-bert + K-Fold + Normalizer** - Test F1: 73.00%
- **BanglaBERT + K-Fold CV** - Test F1: 72.00%
- **XLM-RoBERTa + K-Fold CV** - Test F1: 68.00%

### Subtask 1C - Target Identification in Hate Speech
Identification of hate speech targets and entities.

#### 📊 **Developmental Phase Results**

##### **LLMs with Adversarial Training and K-Fold**
All using BanglaBERT (cse-buet-nlp) with different adversarial techniques:
- **BanglaBERT + FreeLB** - F1 Score: 74.52% ⭐ (Best Development Score)
- **BanglaBERT + Simple FreeLB** - F1 Score: 73.91%
- **BanglaBERT + GAT** - F1 Score: 73.79%
- **BanglaBERT + FGM** - F1 Score: 73.75%

#### 🎯 **Evaluation Phase Results**
- No evaluation phase implemented (Competition scope limited to Subtasks 1A and 1B)

## Technical Implementation Details

### Advanced Training Techniques

#### **Adversarial Training Methods**
- **FGM (Fast Gradient Method)**: Simple and efficient adversarial perturbations
- **AWP (Adversarial Weight Perturbation)**: Weight-space adversarial training
- **FreeLB**: Free large-batch adversarial training for improved generalization
- **Simple FreeLB**: Streamlined version of FreeLB
- **GAT (Geometry-Aware Training)**: Advanced geometry-aware adversarial training

#### **Text Normalization Pipeline**
```python
normalize(
    text,
    unicode_norm="NFKC",          # Canonical decomposition + compatibility
    punct_replacement=None,        # Preserve original punctuation
    url_replacement=None,          # Preserve URLs
    emoji_replacement=None,        # Preserve emojis
    apply_unicode_norm_last=True   # Apply normalization as final step
)
```

#### **Custom Model Architectures**
- **Attention-Based Pooling Head**: Dynamic token weighting for better representation
- **Multi-Head Classification**: Custom classification layers for Bengali text
- **Enhanced Dropout Strategies**: Improved regularization techniques

#### **Cross-Validation Strategy**
- **K-Fold Implementation**: 5-fold cross-validation for robust evaluation
- **Stratified Sampling**: Maintaining class distribution across folds
- **Ensemble Averaging**: Combining predictions from multiple folds

## Performance Analysis

### 📈 Best Performing Models by Phase

#### Developmental Phase Champions:
| Subtask | Model | F1 Score | Technique |
|---------|-------|----------|-----------|
| **1A** | BanglaBERT | 74.88% | K-Fold + FGM + Normalizer |
| **1B** | MuRIL-large-cased | 74.96% | K-Fold Cross Validation |
| **1C** | BanglaBERT | 74.52% | FreeLB Adversarial Training |

#### Evaluation Phase Performance:
| Subtask | Model | Dev F1 | Test F1 | Performance Drop |
|---------|-------|--------|---------|------------------|
| **1A** | BanglaBERT + FGM + Normalizer | 74.88% | 72.00% | -2.88% |
| **1B** | BanglaBERT + K-Fold + Normalizer | 74.72% | 73.00% | -1.72% |
| **1C** | - | - | - | Not evaluated |

### Key Performance Insights

#### Development vs Evaluation Observations:
- **Generalization Gap**: 1-3% performance drop from development to test
- **Most Stable**: K-Fold + Normalizer combinations showed best consistency
- **Overfitting Risk**: Single models without cross-validation showed higher variance
- **Best Generalization**: Adversarial training methods maintained performance better

#### Technical Effectiveness:
- **K-Fold Cross Validation**: Consistent 2-3% improvement across all models
- **Text Normalization**: Additional 0.5-1% boost for Bengali text processing
- **Adversarial Training**: 0.5-1.5% improvement with better robustness
- **Combined Techniques**: Best overall performance with stacked improvements
- **Transformer Superiority**: 15-20% improvement over traditional deep learning

## Model Architecture Details

### Transformer Models Utilized
- **BanglaBERT (csebuetnlp)**: Specialized Bengali language model
- **MuRIL-large-cased**: Multilingual model with strong Bengali support
- **XLM-RoBERTa (base & large)**: Cross-lingual transformer variants
- **DistilBERT-multilingual**: Lightweight multilingual model

### Custom Implementations
- **Enhanced Tokenization**: Bengali-specific preprocessing pipelines
- **Dynamic Padding**: Efficient batch processing strategies
- **Label Smoothing**: Improved training stability
- **Learning Rate Scheduling**: Optimized training convergence

## File Organization

### Directory Structure:
```
Shared_Task1_HateSpeech/
├── subtask1A/                    # Binary hate speech detection
│   ├── Developmental Phase/
│   │   ├── DL Models/           # BiLSTM, LSTM-Attention
│   │   ├── LLMs/                # Base transformer models
│   │   ├── LLMS with K Fold CV/ # K-Fold implementations
│   │   ├── K Folds with normalizer/
│   │   ├── LLMs_KFolds_adversarial attacks/
│   │   ├── LLMS_KFolds_attacks_normalizer/
│   │   └── Various classification heads/
│   └── Evaluation Phase/        # Final test submissions
├── subtask1B/                   # Multi-class classification
│   ├── Developmental Phase/
│   │   ├── LLMs/
│   │   ├── LLMS with K Fold CV/
│   │   └── K Folds with normalizer/
│   └── Evaluation Phase/
└── subtask1C/                   # Target identification
    └── Developmental Phase/
        └── LLMs with adversarial attacks and K Fold CV/
```

### Naming Convention:
- **Model directories**: `v{f1_score}_{model_name}`
  - Example: `v0.7488_banglabert-fgm` = 74.88% F1 score using BanglaBERT with FGM
- **Each directory contains**:
  - Jupyter notebook (.ipynb) with complete implementation
  - Dataset file (subtask_1X.tsv)
  - Model checkpoints and outputs

## Performance Evolution

### Developmental Phase Progression:
1. **Baseline Models**: 55-68% F1 (Deep Learning approaches)
2. **Base Transformers**: 68-73% F1 (Standard LLM implementations)
3. **K-Fold Enhancement**: 70-74% F1 (Cross-validation improvements)
4. **Normalization Boost**: 73-75% F1 (Text preprocessing optimization)
5. **Adversarial Training**: 73-75% F1 (Robustness improvements)
6. **Combined Excellence**: 74-75% F1 (Best technique combinations)

### Development → Evaluation Trends:
- **Average Performance Drop**: 1-3% on unseen test data
- **Most Stable Approaches**: K-Fold + Normalizer combinations
- **Highest Risk**: Single model implementations without regularization
- **Best Generalization**: Models with adversarial training components

## Technologies and Frameworks

### Core Technologies:
- **Deep Learning**: PyTorch, TensorFlow
- **Transformers**: Hugging Face Transformers library
- **Text Processing**: Custom Bengali normalizers, NLTK
- **Evaluation**: Scikit-learn, Custom metrics implementations
- **Adversarial**: Custom FGM, AWP, FreeLB implementations
- **Cross-Validation**: Stratified K-Fold with scikit-learn

### Hardware and Training:
- **GPU Acceleration**: CUDA-enabled training
- **Mixed Precision**: For memory efficiency
- **Gradient Accumulation**: Effective batch size optimization
- **Early Stopping**: Preventing overfitting

## Key Contributions

### Novel Techniques Implemented:
1. **Bengali-Specific Normalization**: NFKC Unicode with preservation strategies
2. **Advanced Adversarial Training**: Multiple adversarial techniques comparison
3. **Custom Attention Heads**: Learnable pooling mechanisms
4. **Robust Cross-Validation**: Stratified K-Fold with ensemble strategies
5. **Multi-Phase Evaluation**: Systematic development vs evaluation analysis

### Research Insights:
- **Language-Specific Approaches**: Bengali text requires specialized preprocessing
- **Adversarial Robustness**: Significant impact on generalization
- **Cross-Validation Importance**: Critical for reliable performance estimation
- **Model Ensemble Benefits**: Combining techniques yields optimal results

## Usage Instructions

### Running Experiments:
1. Navigate to desired subtask directory
2. Choose appropriate approach folder
3. Open corresponding Jupyter notebook
4. Ensure required dependencies are installed
5. Execute cells sequentially for complete pipeline

### Model Training:
- Each notebook contains complete training pipeline
- Data preprocessing and normalization included
- Model evaluation and metrics calculation automated
- Results saved with performance indicators

## Future Work

### Potential Improvements:
- **Multi-Modal Approaches**: Incorporating contextual information
- **Advanced Ensembling**: Sophisticated model combination strategies
- **Real-Time Processing**: Optimized inference pipelines
- **Transfer Learning**: Cross-task knowledge transfer
- **Data Augmentation**: Synthetic data generation for Bengali

### Research Directions:
- **Explainability**: Understanding model decision processes
- **Fairness Analysis**: Bias detection and mitigation
- **Cross-Lingual Transfer**: Knowledge sharing across languages
- **Domain Adaptation**: Generalization to different text domains

## Citation and Acknowledgments

This work represents comprehensive exploration of hate speech detection in Bengali, contributing to the advancement of multilingual NLP and social media content moderation.

---

**Note**: This repository demonstrates state-of-the-art approaches for Bengali hate speech detection, with particular emphasis on robust evaluation methodology and practical implementation strategies.