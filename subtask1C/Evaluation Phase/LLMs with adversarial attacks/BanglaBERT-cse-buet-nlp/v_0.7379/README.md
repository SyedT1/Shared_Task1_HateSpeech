# Geometry-Aware Adversarial Training (GAT) for Hate Speech Detection

This implementation applies **Geometry-Aware Adversarial Training (GAT)** based on the paper "Improving Robustness of Language Models from a Geometry-aware Perspective" (ACL 2022) to a multi-task hate speech detection system for Bengali text.

## Overview

The system performs multi-task classification for hate speech detection with three tasks:
1. **Hate Type Classification** (6 classes: None, Religious Hate, Sexism, Political Hate, Profane, Abusive)
2. **Severity Classification** (3 classes: Little to None, Mild, Severe)
3. **Target Classification** (5 classes: None, Individual, Organization, Community, Society)

## Key Innovation: GAT Implementation

### What is GAT?
GAT improves model robustness by:
- **FADA (Friendly Adversarial Data Augmentation)**: Generates adversarial examples near the decision boundary without crossing it
- **Efficient Training**: Starts adversarial training from friendly examples, reducing required search steps
- **Better Accuracy**: Maintains clean accuracy while improving robustness

### GAT Components:
1. **FGM (Fast Gradient Method)**: Basic adversarial perturbation method
2. **FADA**: Creates "friendly" adversarial examples (ε=0.2)
3. **GATTrainer**: Custom trainer implementing the GAT training loop

## Prerequisites

### Installation
```bash
# Download dataset files
wget https://raw.githubusercontent.com/AridHasan/blp25_task1/refs/heads/main/data/subtask_1C/blp25_hatespeech_subtask_1C_dev.tsv
wget https://raw.githubusercontent.com/AridHasan/blp25_task1/refs/heads/main/data/subtask_1C/blp25_hatespeech_subtask_1C_dev_test.tsv
wget https://raw.githubusercontent.com/AridHasan/blp25_task1/refs/heads/main/data/subtask_1C/blp25_hatespeech_subtask_1C_train.tsv

# Install required packages
pip install transformers
pip install datasets
pip install evaluate
pip install torch
pip install scikit-learn
```

## Step-by-Step Process

### Step 1: Data Loading and Preprocessing
```python
# Load TSV files
train_df = pd.read_csv('blp25_hatespeech_subtask_1C_train.tsv', sep='\t')
validation_df = pd.read_csv('blp25_hatespeech_subtask_1C_dev.tsv', sep='\t')
test_df = pd.read_csv('blp25_hatespeech_subtask_1C_dev_test.tsv', sep='\t')

# Map text labels to numerical values
hate_type_map = {'None': 0, 'Religious Hate': 1, 'Sexism': 2, ...}
severity_map = {'Little to None': 0, 'Mild': 1, 'Severe': 2}
to_whom_map = {'None': 0, 'Individual': 1, 'Organization': 2, ...}
```

### Step 2: Model Architecture
```python
class MultiTaskModel(torch.nn.Module):
    """
    Multi-task model with three classification heads:
    - Hate type classification (6 classes)
    - Severity classification (3 classes)  
    - Target classification (5 classes)
    
    Base model: BanglaBERT (csebuetnlp/banglabert)
    """
```

### Step 3: GAT Components Implementation

#### 3.1 FGM (Fast Gradient Method)
```python
class FGM():
    """
    Applies adversarial perturbations to embeddings
    - Stores backup of original parameters
    - Applies gradient-based perturbation
    - Can restore original parameters
    """
```

#### 3.2 FADA (Friendly Adversarial Data Augmentation)
```python
class FADA():
    """
    Generates friendly adversarial examples:
    - Uses smaller epsilon (0.3) to stay near decision boundary
    - Examples don't cross decision boundary
    - Maintains classification accuracy
    """
```

#### 3.3 GATTrainer
```python
class GATTrainer(Trainer):
    """
    Implements GAT training loop:
    1. Train on clean data
    2. Generate & train on friendly adversarial examples (ε=0.2)
    3. Apply stronger adversarial training (ε=0.5)
    
    Returns average of all three losses
    """
```

### Step 4: Training Configuration
```python
training_args = TrainingArguments(
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=8,  # Reduced for memory
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,   # Effective batch size = 16
    fp16=True,                       # Mixed precision for memory efficiency
    ...
)
```

### Step 5: 5-Fold Cross-Validation
```python
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # 1. Split data for current fold
    train_dataset = combined_dataset.select(train_idx.tolist())
    val_dataset = combined_dataset.select(val_idx.tolist())
    
    # 2. Initialize fresh model
    model = MultiTaskModel(model_name)
    
    # 3. Initialize GAT trainer
    trainer = GATTrainer(
        model=model,
        use_gat=True,
        fada_epsilon=0.2,  # Friendly adversarial epsilon
        adv_epsilon=0.5,   # Standard adversarial epsilon
        ...
    )
    
    # 4. Train and evaluate
    train_result = trainer.train()
    eval_result = trainer.evaluate()
```

### Step 6: Ensemble Prediction
```python
# Collect predictions from all folds
fold_probs = []  # Store probabilities from each fold

# Average predictions across folds
ensemble_probs = np.mean(fold_probs, axis=0)

# Generate final predictions
final_hate_preds = np.argmax(hate_probs, axis=1)
final_sev_preds = np.argmax(sev_probs, axis=1)
final_to_preds = np.argmax(to_probs, axis=1)
```

### Step 7: Output Generation
```python
# Save predictions in competition format
with open("subtask_1C.tsv", "w") as writer:
    writer.write("id\thate_type\thate_severity\tto_whom\tmodel\n")
    for index in range(len(predictions)):
        writer.write(f"{id}\t{hate_type}\t{severity}\t{target}\t{model_name}\n")
```

## Training Flow Diagram

```
Input Text
    ↓
Tokenization (BanglaBERT Tokenizer)
    ↓
Multi-Task Model
    ↓
GAT Training Loop:
    ├── 1. Clean Data Forward/Backward
    ├── 2. FADA (ε=0.2) Forward/Backward
    └── 3. Adversarial (ε=0.5) Forward/Backward
    ↓
Average Losses
    ↓
Update Model Parameters
    ↓
Repeat for Each Batch/Epoch
    ↓
5-Fold Cross-Validation
    ↓
Ensemble Predictions
    ↓
Final Output
```

## Memory Optimization Strategies

1. **Reduced Batch Size**: 16 → 8 per device
2. **Gradient Accumulation**: 2 steps (maintains effective batch size)
3. **Mixed Precision (FP16)**: Reduces memory usage by ~50%
4. **Optimized Epsilon Values**: Smaller perturbations for memory efficiency

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 2e-5 | Standard for BERT fine-tuning |
| Batch Size | 8 | Per device (16 effective with accumulation) |
| FADA Epsilon | 0.2 | Friendly adversarial perturbation |
| Adversarial Epsilon | 0.5 | Standard adversarial perturbation |
| Max Sequence Length | 512 | Maximum token length |
| Number of Folds | 5 | Cross-validation folds |
| Epochs | 1 | Training epochs per fold |

## Expected Results

After training with GAT, you should observe:
- **Improved robustness** against adversarial attacks
- **Maintained or improved clean accuracy** compared to standard training
- **Better generalization** through ensemble of 5 folds
- **Three task outputs**: Hate type, severity, and target classifications

## Performance Metrics

The model evaluates three metrics:
```python
- hate_accuracy: Accuracy for hate type classification
- severity_accuracy: Accuracy for severity classification  
- to_whom_accuracy: Accuracy for target classification
```

## Advantages of GAT

1. **Efficiency**: Fewer adversarial search steps needed
2. **Accuracy Preservation**: Friendly examples don't hurt clean accuracy
3. **Robustness**: Improved defense against adversarial attacks
4. **Geometry-Aware**: Considers decision boundary geometry

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{zhu2022improving,
  title={Improving Robustness of Language Models from a Geometry-aware Perspective},
  author={Zhu, Bin and Gu, Zhaoquan and Wang, Le and Chen, Jinyin and Xuan, Qi},
  booktitle={Findings of ACL 2022},
  pages={3115--3125},
  year={2022}
}
```

## Notes

- The model uses BanglaBERT as the base model, optimized for Bengali text
- GAT is particularly effective for improving robustness while maintaining accuracy
- Cross-validation ensures robust evaluation and reduces overfitting
- Ensemble predictions combine knowledge from all folds for better performance