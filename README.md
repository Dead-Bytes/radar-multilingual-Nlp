# Multi-Class Authorship Attribution for AI-Generated Text Detection
## Addressing the "Translator as a Generator" Problem

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/microsoft/mdeberta-v3-base)

---

## 📋 Project Overview

This project implements a novel **multi-class authorship attribution system** to distinguish between:
1. **Human-written text** (authentic authorship)
2. **AI-generated text** (Large Language Models)
3. **Machine-translated text** (Neural Machine Translation)

Unlike traditional binary classifiers (Human vs. AI), our approach addresses the critical **"Translator as a Generator"** problem identified in multilingual AI text detection research.

**Research Paper:** "Understanding and Improving Limitations of Multilingual AI Text Detection"

---

## 👥 Contributors

| Name | Roll Number | Role |
|------|-------------|------|
| **Abhinav Pangaria** | 2201005 | Lead Researcher |
| **Kaushal Chaudhary** | 2201058 | Model Development |
| **DivyRaj Saini** | 2201070 | Data Engineering |
| **Sahil Burman** | 2201172 | Evaluation & Analysis |

---

## 📊 Project Presentation

**Google Slides:** [Multi-Class Authorship Attribution Presentation](https://docs.google.com/presentation/d/1z29LxuIM7_6yYXkKGe-03Bd2EysU4Cr4VrasxV9575I/edit?slide=id.g398fa804197_2_102#slide=id.g398fa804197_2_102)

---

## 📖 Documentation Structure

### Core Documentation (Read in Order)

1. **[PROJECT_REPORT.md](PROJECT_REPORT.md)** ⭐ **START HERE**
   - Problem statement and research motivation
   - Methodology overview (Phase 1 & 2)
   - Dataset creation process
   - Next steps and research roadmap

2. **[DATA_SUMMARIZATION.md](DATA_SUMMARIZATION.md)**
   - Comprehensive dataset analysis with visualizations
   - Class distribution and text length patterns
   - Vocabulary richness (Type-Token Ratio) analysis
   - N-gram analysis and t-SNE embeddings
   - Feature engineering recommendations

3. **[MODEL_DIAGNOSTIC_REPORT.md](MODEL_DIAGNOSTIC_REPORT.md)**
   - Deep dive into Transformer & BERT architecture
   - Justification for mDeBERTa-v3-base selection
   - Training methodology (AdamW optimizer, learning rate schedules)
   - Hyperparameter rationale and evaluation metrics
   - Reproducibility guidelines

4. **[RESULTS.md](RESULTS.md)** 
   - Model performance metrics (Accuracy, Macro F1)
   - Confusion matrix analysis
   - Per-class precision, recall, F1 scores
   - Error analysis and failure modes

5. **[FUTURE_RESEARCH.md](FUTURE_RESEARCH.md)**
   - Cross-domain generalization directions
   - Finer-grained LLM detection (GPT-4 vs Claude vs Llama)
   - Temporal analysis of evolving AI writing styles
   - Real-world deployment considerations
   - Adversarial robustness testing

### Technical Documentation

- **[CLAUDE.md](CLAUDE.md)** - Developer guide for Claude Code (AI assistant context)
- **[requirements.txt](requirements.txt)** - Python dependencies

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd radar-multilingual-Nlp

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Phase 1: Dataset Preparation

```bash
# Create multi-class dataset (1,500 samples: 500 human, 500 AI, 500 MT)
python3 prepare_data.py

# Output: processed_xai_dataset.csv
```

### Phase 2: Model Training

```bash
# Train mDeBERTa-v3-base for multi-class classification
python3 train_multiclass.py

# Outputs:
# - confusion_matrix_multiclass.png
# - per_class_metrics.csv
# - overall_metrics.txt
# - final_model_multiclass/
```

### Phase 3: Evaluation & Analysis

```bash
# Run dataset analysis (generates visualizations)
jupyter notebook dataset_analysis.ipynb
```

---

## 📁 Project Structure

```
radar-multilingual-Nlp/
├── prepare_data.py              # Dataset creation script
├── train_multiclass.py          # Multi-class model training
├── dataset_analysis.ipynb       # EDA and visualization
├── processed_xai_dataset.csv    # Balanced 3-class dataset
│
├── Documentation/
│   ├── PROJECT_REPORT.md        # Main research report
│   ├── DATA_SUMMARIZATION.md    # Dataset analysis
│   ├── MODEL_DIAGNOSTIC_REPORT.md  # Model architecture & training
│   ├── RESULTS.md               # Performance metrics (TBD)
│   └── FUTURE_RESEARCH.md       # Research directions
│
├── Legacy Scripts/ (from prior binary classification research)
│   ├── tests.py                 # Multi-detector evaluation
│   ├── paraphrase.py            # Adversarial testing
│   ├── ai_generate.py           # Synthetic text generation
│   ├── finetune_radar.py        # RADAR model fine-tuning
│   └── pipeline_*.ipynb         # Language-specific pipelines
│
└── Results/ (generated after training)
    ├── confusion_matrix_multiclass.png
    ├── per_class_metrics.csv
    ├── overall_metrics.txt
    └── final_model_multiclass/
```

---

## 🔬 Key Features

### Dataset
- **1,500 balanced samples** (500 per class)
- **Sources:**
  - Human & AI text: Multitude corpus (English)
  - Machine-translated: Portuguese → English via NMT
- **Random seed:** 42 (full reproducibility)

### Model
- **Architecture:** mDeBERTa-v3-base (Microsoft)
  - 279M parameters
  - Disentangled attention mechanism
  - Multilingual pre-training (100+ languages)
- **Training:**
  - 5 epochs with AdamW optimizer
  - Warmup + linear decay learning rate schedule
  - Macro F1 as primary metric

### Evaluation
- Per-class precision, recall, F1-score
- Confusion matrix visualization
- Statistical significance testing

---

## 🧪 Advanced Usage

### Testing with Custom Models

```bash
# Evaluate custom fine-tuned model
python3 tests.py --tr True --custom path/to/model --dataset path/to/dataset --output path/to/output_file
```

### Adversarial Testing (Paraphrasing)

```bash
# Backtranslation attack
python3 paraphrase.py --mode backtranslation --custom path/to/model --dataset path/to/dataset --output path/to/output_file

# Available modes: backtranslation, transformer, translation
```

### Synthetic Text Generation

```bash
# Generate AI text in multiple languages
python3 ai_generate.py --language French --model llama --device cuda --samples 10 --output_ai ai_test.txt

# Supported languages: French, German, Italian, Spanish
# Supported models: llama, vicuna-7b
```

---

## 📊 Expected Results

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Overall Accuracy** | 75% | 85% |
| **Macro F1** | 70% | 80% |
| **MT F1** | 80% | 90% |
| **Human F1** | 65% | 75% |
| **AI F1** | 65% | 75% |

**Rationale:** Machine-translated text has distinctive artifacts (highest target), while Human vs. AI is the most challenging distinction.

---

## 🛠️ Technical Requirements

### Minimum Hardware
- **GPU:** 12GB+ VRAM (NVIDIA RTX 3060, Tesla T4, V100)
- **RAM:** 16GB
- **Storage:** 5GB

### Recommended Hardware
- **GPU:** NVIDIA A100 (40GB) or RTX 4090 (24GB)
- **RAM:** 32GB
- **Storage:** 10GB

### Software
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.57+
- CUDA 11.8+ (for GPU acceleration)

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{multiclass-authorship-attribution-2025,
  title={Multi-Class Authorship Attribution for AI-Generated Text Detection},
  author={Pangaria, Abhinav and Chaudhary, Kaushal and Saini, DivyRaj and Burman, Sahil},
  year={2025},
  note={Understanding and Improving Limitations of Multilingual AI Text Detection}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Acknowledgments

- **Microsoft Research** for the mDeBERTa-v3-base model
- **HuggingFace** for the Transformers library
- **Multitude Corpus** for providing the base dataset
- Research paper: "Understanding and Improving Limitations of Multilingual AI Text Detection"

---

## 📧 Contact

For questions or collaboration inquiries, please contact the project contributors via their institutional email addresses.

