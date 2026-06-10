# DiffPQ-GenRec: End-to-End Learnable Semantic IDs for Sequential Recommendation

This project aims to improve upon existing generative sequential recommendation models by making tokenization differentiable—allowing gradients to flow back and optimize semantic IDs during training.


---

## Pipeline Overview

```
Raw Dataset → Tokenizer → Model (forward/generate) → Evaluator
```
---

## Stage 1: Dataset

### Purpose
The dataset layer handles raw data loading, preprocessing, and splitting into train/val/test sets using a **leave-one-out** strategy.

### Output Contract
Returns a dictionary with three HuggingFace `Dataset` objects:

```python
{
    'train': HF Dataset([
        {'user': 'A2OKNI5Z', 'item_seq': ['B001A3E5A4', 'B002BVQY1C', 'B003K2WJVQ']},
        {'user': 'A3K2L9X1', 'item_seq': ['C001M2P5X7', 'C002N3Q6Y8']},
        ...
    ]),
    'val': HF Dataset([...]),
    'test': HF Dataset([...]),
}
```

**Add more dataset?**: Just need to follow the output contract

---

## Stage 2: Tokenizer

### Purpose
Converts raw item sequences into model-ready integer token sequences. 

### Output Contract
Returns a dictionary with three tokenized HuggingFace `Dataset` objects:

```python
{
    'train': HF Dataset([
        {
            'input_ids': [[1, 5, 20, 0, 0, ...], ...],      # Sequence of item IDs (NOT semantic tokens)
            'mask': [[1, 1, 1, 0, 0, ...], ...],  # Binary: real items vs padding
            'labels': [[10, 15, 0, -100, -100, ...], ...],  # Target item IDs (-100 for padding)
            'seq_lens': [3, 2, ...],                        # Actual sequence length before padding
            ... More depend on the model
        },
        ...
    ]),
    'val': HF Dataset([...]),   
    'test': HF Dataset([...]),
}
```


**Add more tokenizer?**: inherit `AbstractTokenizer` and override:
- `tokenize_function()`: Convert one user's item sequence into model inputs
- Inherit `tokenize()` which maps your function over all splits
- Follow the output contract

---

## Stage 3: Model

### Purpose
Predicts the next item given a user's historical sequence. Models can use any architecture, but **must** implement two methods.

### Required Methods

#### 1. `forward(batch, return_loss=True)`
**Training path**: Compute loss for gradient updates.

```python
def forward(self, batch: dict, return_loss=True):
    # model logic 
    outputs.loss = cross_entropy_loss(...)
    return outputs
```

#### 2. `generate(batch, n_return_sequences=50)`
**Inference path**: Return top-K predicted item IDs.

```python
def generate(self, batch: dict, n_return_sequences=50):

    preds = item_logits.topk(n_return_sequences).indices
    return preds  # Shape: (B, K)
```

**Add more model?**: inherit `AbstractModel` and implement `forward()` and `generate()`.

---

## Stage 4: Evaluator

### Purpose
Computes ranking metrics (Recall@K, NDCG@K) to assess recommendation quality.

### Input Contract

```python
preds = model.generate(batch, n_return_sequences=50)  # (batch_size, 50)
labels = batch['labels']                               # (batch_size,)

results = evaluator.calculate_metrics(preds, labels)
```

Current code support: 
1. Recall@K 
2. NDCG@K

---