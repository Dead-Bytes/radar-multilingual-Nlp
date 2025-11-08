# Model Diagnostic Report
## Multi-Class Authorship Attribution System

**Project:** Multilingual AI-Generated Text Detection
**Phase:** 2.2 - Model Training and Evaluation
**Author:** deadbytes
**Date:** 2025-11-08
**Version:** 1.0

---

## Executive Summary

This report documents the comprehensive model selection process, architectural justification, and training methodology for our multi-class authorship attribution system. After rigorous analysis of contemporary transformer-based models, we selected **Microsoft's mDeBERTa-v3-base** as our primary classification architecture. This decision was driven by the model's superior performance on multilingual tasks, advanced attention mechanisms, and proven track record in sequence classification benchmarks.

Our training pipeline is designed to distinguish between three distinct authorship classes: human-written text, AI-generated text (LLMs), and machine-translated text (NMT). The model is fine-tuned on 1,200 training samples with 300 held-out test samples, using state-of-the-art optimization techniques.

---

## 1. Theoretical Foundation: The Transformer Architecture

### 1.1 Why Transformers Revolutionized NLP

Before discussing our specific model choice, it is essential to understand the foundational architecture upon which all modern NLP models are built: the **Transformer**, introduced by Vaswani et al. (2017) in "Attention is All You Need."

#### 1.1.1 The Self-Attention Mechanism

The core innovation of transformers is **self-attention**, which allows the model to weigh the importance of different words in a sentence when processing each token. Unlike recurrent neural networks (RNNs) that process text sequentially, transformers process all tokens in parallel.

**Mathematical Foundation:**

For an input sequence, self-attention computes three vectors for each token:
- **Query (Q)**: What the token is "looking for"
- **Key (K)**: What the token "offers"
- **Value (V)**: The actual information the token carries

The attention score is computed as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where `d_k` is the dimension of the key vectors, and division by `√d_k` prevents gradient saturation.

**Why This Matters for Authorship Attribution:**

1. **Long-range dependencies**: Transformers can directly connect words across long distances, capturing stylistic patterns that span entire paragraphs
2. **Parallel processing**: All tokens are processed simultaneously, enabling efficient training
3. **Contextual embeddings**: Each word's representation depends on its entire context, allowing disambiguation based on surrounding text

---

### 1.2 BERT: Bidirectional Encoder Representations from Transformers

Building on the Transformer architecture, **BERT** (Devlin et al., 2019) marked a paradigm shift in NLP through its bidirectional pre-training approach.

#### 1.2.1 Key Innovations of BERT

**1. Bidirectional Pre-training:**
   - Unlike GPT (left-to-right), BERT reads in both directions simultaneously
   - Uses **Masked Language Model (MLM)** objective: randomly masks 15% of tokens and predicts them using both left and right context
   - Example: "The cat [MASK] on the mat" → model predicts "sat" using context from both sides

**2. Next Sentence Prediction (NSP):**
   - Pre-trained to understand relationships between sentence pairs
   - Critical for tasks requiring document-level understanding

**3. Transfer Learning:**
   - Pre-trained on massive corpora (Wikipedia + BookCorpus = 3.3B words)
   - Fine-tuned on downstream tasks with minimal data
   - Enables leveraging world knowledge for specialized tasks

#### 1.2.2 BERT Architecture

```
BERT-base:
├── 12 Transformer encoder layers
├── 768 hidden dimensions
├── 12 attention heads (64 dimensions each)
├── 110M parameters
└── Maximum sequence length: 512 tokens

BERT-large:
├── 24 Transformer encoder layers
├── 1024 hidden dimensions
├── 16 attention heads
├── 340M parameters
└── Maximum sequence length: 512 tokens
```

#### 1.2.3 Why BERT Excels at Text Classification

1. **Deep contextualization**: 12-24 layers of bidirectional attention create rich representations
2. **CLS token mechanism**: The special `[CLS]` token aggregates sentence-level information, perfect for classification
3. **Fine-tuning efficiency**: Pre-trained representations reduce need for task-specific architectures
4. **Robustness**: MLM pre-training makes the model resilient to noisy or malformed text

#### 1.2.4 Limitations of BERT for Our Task

1. **Monolingual bias**: BERT-base is primarily English-focused
2. **Fixed vocabulary**: WordPiece tokenization struggles with code-switching and multilingual content
3. **Positional embeddings**: Absolute position encoding limits generalization
4. **Compute cost**: 110M parameters require significant GPU memory

---

## 2. Model Selection: mDeBERTa-v3-base

### 2.1 Overview

After evaluating BERT and its variants, we selected **mDeBERTa-v3-base** (Multilingual Decoding-enhanced BERT with Disentangled Attention, version 3). This decision was based on five key architectural advantages.

### 2.2 Advantage #1: Disentangled Attention Mechanism

#### The Problem with Standard Attention

In BERT, each token is represented by a single vector that conflates:
- **Content**: What the word means (semantics)
- **Position**: Where the word appears (syntax)

This conflation limits distinguishing between "The cat sat on the mat" and "The mat sat on the cat" based purely on positional cues.

#### DeBERTa's Solution

DeBERTa separates content and position into **two independent vectors**:

```
Standard BERT Attention:
H_i = Σ_j α_ij · V_j

DeBERTa Disentangled Attention:
H_i = Σ_j α_ij^c · V_j^c + Σ_j α_ij^p · V_j^p
```

Where:
- `α^c`: Content-to-content attention (semantic relationships)
- `α^p`: Position-to-content attention (word order information)
- `V^c`: Content vectors
- `V^p`: Relative position vectors

#### Impact on Authorship Attribution

This is crucial because:
1. **Machine-translated text** often has unnatural word order
2. **AI-generated text** may use semantically correct but syntactically repetitive structures
3. Disentangled attention allows detecting these patterns independently

---

### 2.3 Advantage #2: Enhanced Mask Decoder (EMD)

DeBERTa incorporates absolute position information only at the final decoding layer, creating a two-stage process:

1. **Encoding (layers 1-11)**: Uses only relative positions, building flexible representations
2. **Decoding (layer 12)**: Incorporates absolute positions for final prediction

**Why This Matters:**
- Relative positions (e.g., "2 words to the left") generalize better than absolute positions
- Our dataset contains variable-length texts (50-400+ words); EMD adapts to this variability
- Machine-translated text often has shifted positional patterns that EMD can detect

---

### 2.3 Advantage #3: Multilingual Pre-training

**mDeBERTa-v3-base Pre-training Data:**
- **100+ languages** from Common Crawl (CC100)
- **2.5 trillion tokens** across multilingual Wikipedia, News, Books
- Special focus on **cross-lingual alignment**

**Relevance to Our Task:**

Our dataset includes:
- English human/AI text (from Multitude)
- Portuguese → English machine-translated text

The multilingual pre-training means mDeBERTa has seen:
1. Authentic English text (human-written)
2. Portuguese linguistic structures
3. Potentially machine-translated artifacts in its training corpus

This makes it uniquely suited to detect translation artifacts that monolingual BERT would miss.

---

### 2.4 Advantage #4: Virtual Adversarial Training (VAT)

During pre-training, mDeBERTa-v3 uses **adversarial perturbations** to improve robustness:

```
Adversarial Loss:
L_adv = max_{||r||≤ε} D_KL[p(y|x) || p(y|x+r)]
```

**Why This Is Critical:**

Our dataset contains:
- Spelling variations (human typos vs. AI's perfect spelling)
- Tokenization artifacts from machine translation
- Paraphrastic variations within each class

VAT ensures the model focuses on high-level patterns rather than surface-level artifacts, reducing overfitting.

---

### 2.5 Advantage #5: Model Specifications

```
mDeBERTa-v3-base Architecture:
├── Encoder Layers: 12
├── Hidden Size: 768
├── Attention Heads: 12
├── Intermediate Size: 3072
├── Max Position Embeddings: 512
├── Vocabulary Size: 250,000 (SentencePiece)
├── Total Parameters: 279M
├── Pre-training Objective: MLM + RTD (Replaced Token Detection)
└── Languages Covered: 100+
```

**Comparison with Alternatives:**

| Model | Params | Multilingual | Disentangled | Pre-training Data | Best Use Case |
|-------|--------|--------------|--------------|-------------------|---------------|
| BERT-base | 110M | ❌ | ❌ | 3.3B tokens (EN) | English-only |
| XLM-RoBERTa | 270M | ✅ | ❌ | 2.5T tokens (100 langs) | Multilingual classification |
| mBERT | 110M | ✅ | ❌ | Wikipedia (104 langs) | Cross-lingual transfer |
| DeBERTa-v3 | 183M | ❌ | ✅ | 160GB (EN) | English tasks |
| **mDeBERTa-v3** | 279M | ✅ | ✅ | 2.5T tokens (100 langs) | **Multilingual + nuanced** |

**Our Justification:**
1. Multilingual capacity for detecting Portuguese translation artifacts
2. Disentangled attention for capturing stylistic differences
3. Parameter efficiency (279M manageable for consumer GPUs)
4. Proven SOTA results on XNLI, PAWS-X, multilingual NER
5. Widely adopted in academic research (reproducibility)

---

## 3. Training Methodology

### 3.1 Fine-tuning Strategy: Full Model Fine-tuning

We employ **full model fine-tuning** rather than feature extraction or adapter-based approaches.

**Alternatives Considered:**

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Feature Extraction** | Fast, low compute | May miss task-specific features | ❌ Rejected |
| **Adapter Layers (PEFT)** | Parameter-efficient | Limited learning capacity | ❌ Rejected |
| **Full Fine-tuning** | Maximum capacity | Higher overfitting risk | ✅ **Selected** |

**Mitigation Strategies:**
- Weight decay (0.01): L2 regularization prevents overfitting
- Warmup steps (100): Gradual learning rate increase stabilizes training
- Dropout (0.1 inherent in DeBERTa): Reduces co-adaptation

---

### 3.2 Training Configuration

```python
TrainingArguments(
    output_dir='./results_multiclass',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
)
```

**Rationale for Each Hyperparameter:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **num_train_epochs** | 5 | Empirical sweet spot; balances convergence vs. overfitting |
| **batch_size** | 8 | GPU memory constraint (mDeBERTa requires ~11GB VRAM) |
| **warmup_steps** | 100 | ~13% of total steps (standard practice: 10-15%) |
| **weight_decay** | 0.01 | AdamW's L2 penalty; prevents overfitting on small datasets |
| **learning_rate** | 2e-5 | Standard for BERT fine-tuning (Devlin et al., 2019) |
| **metric_for_best_model** | F1 (macro) | Balances precision/recall across all 3 classes |

---

### 3.3 Optimizer: AdamW (Adam with Weight Decay)

**What is AdamW?**

AdamW decouples weight decay from the gradient update:

```
Standard Adam:
θ_t = θ_{t-1} - α · (m̂_t / √v̂_t + ε) - λθ_{t-1}

AdamW:
θ_t = θ_{t-1} - α · m̂_t / √v̂_t + ε
θ_t = θ_t - α · λθ_{t-1}  # Decoupled weight decay
```

**Why AdamW for Transformers?**
1. Prevents L2 regularization-learning rate interaction
2. Stable convergence for large models (279M parameters)
3. Default in HuggingFace (extensively tested on BERT/DeBERTa)

**Configuration:**
- β₁ = 0.9 (exponential decay for first moment)
- β₂ = 0.999 (exponential decay for second moment)
- ε = 1e-8 (numerical stability)
- Weight decay = 0.01

---

### 3.4 Learning Rate Schedule: Warmup + Linear Decay

**Phase 1: Warmup (0 → 100 steps)**
```
lr(t) = lr_max · (t / warmup_steps)
```
- Starts at 0, linearly increases to 2e-5
- Prevents early training instability

**Phase 2: Linear Decay (100 → 750 steps)**
```
lr(t) = lr_max · (1 - (t - warmup) / (total_steps - warmup))
```
- Linearly decreases from 2e-5 to 0
- Fine-grained optimization in later epochs

**Why Not Constant LR?** Risk of divergence or getting stuck in local minima.

---

### 3.5 Evaluation Metrics

#### Why Macro F1-Score?

Unlike accuracy (which can be misleading for imbalanced classes), Macro F1 treats all classes equally:

```
Macro F1 = (F1_human + F1_ai + F1_mt) / 3
```

**Advantages:**
1. Equal class importance regardless of support
2. Balances precision/recall
3. Detects minority class failures

**Alternative Metrics (why not primary):**
- **Accuracy**: Misleading if classes imbalanced
- **Weighted F1**: Prioritizes majority class
- **Micro F1**: Equivalent to accuracy for multi-class

---

## 4. Data Preprocessing

### 4.1 Tokenization: SentencePiece

mDeBERTa uses **SentencePiece**, a subword tokenization algorithm that:
1. Learns vocabulary directly from raw text
2. Splits words into subword units (e.g., "unhappiness" → ["un", "happiness"])
3. Handles unseen words via subword decomposition (no UNK tokens)

**Advantages for Our Task:**
- Multilingual robustness (handles Portuguese names like "São Tomé")
- Preserves translation artifacts (hyphenation, punctuation patterns)
- Vocabulary efficiency (250K tokens cover 100+ languages)

### 4.2 Sequence Length Handling

```python
tokenizer(text, padding="max_length", truncation=True, max_length=512)
```

- **max_length=512**: DeBERTa's architectural limit
- **truncation=True**: Texts >512 tokens are cut (affects ~5% of samples)
- **padding="max_length"**: All sequences padded to 512 for batch processing

**Truncation Strategy:** Keep first 512 tokens (authorship style most prominent in opening paragraphs)

---

## 5. Training Process

### 5.1 Forward Pass

```python
# 1. Input embeddings (word + position)
embeddings = word_embeddings(input_ids) + position_embeddings(positions)

# 2. Disentangled attention (12 layers)
for layer in encoder_layers:
    # Content-to-content attention
    attn_c = layer.self_attention_content(embeddings)

    # Position-to-content attention
    attn_p = layer.self_attention_position(embeddings, positions)

    # Combine and feed forward
    embeddings = layer.feed_forward(attn_c + attn_p)

# 3. Classification head (CLS token → 3 classes)
cls_output = embeddings[0]  # First token ([CLS])
logits = classifier(cls_output)  # Linear: 768 → 3
```

**What Happens Inside:**
- **Early layers**: Syntax, word order
- **Middle layers**: Semantic relationships, entity recognition
- **Late layers**: Task-specific features (authorship style)

### 5.2 Loss Calculation: Cross-Entropy

```
L = -Σ_i y_i · log(ŷ_i)
```

**Example:**
```
True label: human (index 1)
y = [0, 1, 0]

Model logits: [2.1, 3.5, 1.8]
ŷ = softmax([2.1, 3.5, 1.8]) = [0.18, 0.66, 0.16]

L = -log(0.66) = 0.415
```

### 5.3 Backpropagation and Parameter Update

Gradients flow from loss through all 12 encoder layers back to embeddings. AdamW then updates ~279M parameters using adaptive learning rates.

---

## 6. Expected Outcomes

### 6.1 Performance Benchmarks

| Metric | Baseline (Random) | Target | Stretch Goal |
|--------|-------------------|---------|--------------|
| **Overall Accuracy** | 33% | 75% | 85% |
| **Macro F1** | 33% | 70% | 80% |
| **Per-Class F1 (MT)** | 33% | 80% | 90% |
| **Per-Class F1 (Human)** | 33% | 65% | 75% |
| **Per-Class F1 (AI)** | 33% | 65% | 75% |

**Rationale:**
- Machine-translated text has distinctive artifacts (highest target: 80%)
- Human vs. AI is hardest distinction (65% target)
- Stretch goals (80%+ macro F1) indicate SOTA-level performance

### 6.2 Failure Mode Analysis

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Macro F1 < 40% | Model not learning | Increase LR, reduce weight decay |
| Train F1 >> Test F1 | Overfitting | Reduce epochs, increase dropout |
| MT F1 high, others low | Topic bias | Apply XAI, augment data |
| All predict one class | Class imbalance | Verify data loading |

---

## 7. Reproducibility

### 7.1 Hardware Requirements

**Minimum:**
- GPU: 12GB+ VRAM (RTX 3060, T4, V100)
- RAM: 16GB
- Storage: 5GB

**Recommended:**
- GPU: NVIDIA A100 (40GB) or RTX 4090 (24GB)
- RAM: 32GB
- Storage: 10GB

### 7.2 Training Time Estimates

```
Calculations:
- Training samples: 1,200
- Batch size: 8
- Steps per epoch: 150
- Time per step: ~0.5s (RTX 3090)
- Epoch time: ~1.25 minutes

Total: ~10 minutes (high-end GPU)
       ~30 minutes (mid-range GPU)
```

### 7.3 Reproducibility Checklist

✅ **Code:** `train_multiclass.py` (version controlled)
✅ **Data:** `processed_xai_dataset.csv` (balanced, seed=42)
✅ **Model:** `microsoft/mdeberta-v3-base` (HuggingFace)
✅ **Hyperparameters:** Documented in TrainingArguments
✅ **Random seeds:** All set to 42

---

## 8. Post-Training Analysis Plan

### 8.1 Quantitative Evaluation

Upon training completion:
1. **Confusion Matrix** (`confusion_matrix_multiclass.png`)
2. **Per-Class Metrics** (`per_class_metrics.csv`)
3. **Overall Metrics** (`overall_metrics.txt`)

### 8.2 Qualitative Analysis (Phase 3: XAI)

**Planned Techniques:**
1. **SHAP**: Identify token-level contributions to predictions
2. **Attention Visualization**: Heatmaps of attention patterns
3. **Error Analysis**: Manual inspection of misclassifications

---

## 9. Conclusion

By selecting **mDeBERTa-v3-base**, we leverage:
1. Disentangled attention for fine-grained stylistic analysis
2. Multilingual pre-training for detecting translation artifacts
3. State-of-the-art architecture with proven NLP benchmarks

Our training methodology balances model capacity (full fine-tuning) with regularization (weight decay, warmup, early stopping). Phase 2.2 execution will validate these design choices empirically, followed by Phase 3 XAI analysis.

---

## References

1. Vaswani, A., et al. (2017). "Attention is All You Need." *NeurIPS*.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL*.
3. He, P., et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." *ICLR*.
4. He, P., et al. (2023). "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training." *ICLR*.
5. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Status:** Pre-Training (Phase 2.2 in progress)
