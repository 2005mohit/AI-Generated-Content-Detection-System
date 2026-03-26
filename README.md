# AI Content Authenticity Detection System

**A Capstone Project by K.R. Mangalam University MCA Students**

A **dual-modality AI detection system** that identifies whether text or images are AI-generated or created by humans. This project demonstrates the application of deep learning and ensemble techniques in content authentication.

---

## Project Overview

### Objectives
1. **Develop an intelligent system** to detect AI-generated text and images
2. **Implement ensemble learning** combining multiple detection approaches
3. **Compare model performance** across different architectures
4. **Evaluate real-world applicability** beyond benchmark datasets
5. **Deploy as a user-friendly application** for practical use

### Key Highlights
- **Text Detection**: RoBERTa ensemble model - **90% test accuracy**
- **Image Detection**: EfficientNet B0 CNN - **70% real-world accuracy**
- **Dual-Modality**: Handles both text and image inputs
- **Ensemble Approach**: Combines deep learning + heuristic analysis
- **Interactive UI**: Streamlit-based web application
- **Team**: 5 members, K.R. Mangalam University MCA (2025-2026)

### Technical Stack
| Component | Technology |
|-----------|-----------|
| **Text Model** | RoBERTa (HuggingFace) |
| **Image Model** | EfficientNet B0 (PyTorch) |
| **Frontend** | Streamlit |
| **Backend** | Python 3.10+ |
| **ML Framework** | PyTorch, Transformers |
| **Data Processing** | Pandas, NumPy, scikit-learn |

---

## Problem Statement & Motivation

With the rapid advancement of **Generative AI models** (ChatGPT, Claude, DALL-E, Midjourney), there is an urgent need for reliable detection systems to:

- **Combat academic dishonesty** in educational institutions
- **Prevent misinformation** and fake content spread
- **Maintain content authenticity** across digital platforms
- **Support fact-checking** and journalism
- **Protect intellectual property** from unauthorized AI replication

This project addresses these challenges by developing a **scalable, accurate, and user-friendly detection system** suitable for deployment in educational and professional environments.

---

## Project Architecture

### System Design

```
┌─────────────────────────────────────────┐
│        USER INTERFACE LAYER             │
│      (Streamlit Web Application)        │
│  ┌─────────────────────────────────┐   │
│  │ Text Tab | Image Tab | Results  │   │
│  └─────────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               │
      ┌────────┴─────────┐
      │                  │
      ▼                  ▼
┌──────────────┐  ┌──────────────┐
│TEXT PIPELINE │  │IMAGE PIPELINE│
│              │  │              │
│1. Preprocess │  │1. Resize     │
│2. Tokenize   │  │2. Normalize  │
│3. RoBERTa    │  │3. CNN        │
│4. Heuristic  │  │4. Threshold  │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
┌──────────────────────────────────┐
│    ENSEMBLE INFERENCE            │
│  • Probability Calculation       │
│  • Confidence Scoring            │
│  • Component Analysis            │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│    RESULTS & VISUALIZATION       │
│  • AI/Real Label                 │
│  • Confidence Score              │
│  • Component Breakdown           │
└──────────────────────────────────┘
```

---

## Repository Structure

```
AI-Content-Authenticity-Detection/
│
├── app.py                          # Main Streamlit application
│
├── Pipeline/
│   ├── text_pipeline.py            # Text detection (RoBERTa + Heuristics)
│   ├── image_pipeline.py           # Image detection (EfficientNet B0)
│   └── __init__.py
│
├── model/
│   ├── text_model/                 # RoBERTa fine-tuned weights
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   │
│   └── image_model/
│       └── image_model.pth         # EfficientNet B0 trained weights
│
├── notebooks/
│   ├── preprocessing.ipynb         # EDA & data cleaning
│   ├── training.ipynb              # Text model training
│   └── image-detection-model.ipynb # Image model training
│
├── requirements.txt                # Dependencies
├── README.md                        # Project documentation
└── .gitignore
```

---

## Dataset Analysis

### Text Dataset Composition
| Source | Platform | Samples | Purpose |
|--------|----------|---------|---------|
| HC3 | HuggingFace | ~50,000 | ChatGPT, Claude, Davinci-3 responses |
| MEGA | Kaggle | ~30,000 | Diverse AI & human-written text |
| **Total** | **Combined** | **~80,000** | 70% train / 15% val / 15% test |

**Data Characteristics**:
- Multiple AI models represented
- Diverse writing styles and domains
- Balanced class distribution after preprocessing
- Length range: 50-2000 tokens

### Image Dataset Composition
| Name | Source | Real Images | AI-Generated | Total | Distribution |
|------|--------|------------|-------------|-------|---------------|
| Defactify COCOAI | HuggingFace | 24,000 | 24,000 | 48,000 | 43.75% train / 9.375% val / 46.875% test |

**Image Characteristics**:
- Resolution: Variable (auto-resized to 224×224)
- AI Models: Stable Diffusion, DALL-E, Midjourney
- Domains: Landscapes, objects, people, abstract
- Quality: Mix of high-quality and compressed images

---

## Model Development & Architecture

### 1. Text Detection Model

#### Model Selection: RoBERTa
- **Reason**: Pre-trained on large corpus, fine-tuned for AI detection
- **Base**: `chatgpt-detector-roberta` (HuggingFace)
- **Architecture**: 12-layer transformer with 768 hidden dimensions

| Parameter | Value |
|-----------|-------|
| Architecture | RoBERTa (Robustly Optimized BERT) |
| Layers | 12 |
| Hidden Units | 768 |
| Attention Heads | 12 |
| Total Parameters | ~110M |
| Max Sequence Length | 512 tokens |
| Output | Binary classification (AI/Human) |
| Decision Threshold | 0.60 |

#### Ensemble Strategy (Text)
**Combining Deep Learning + Rule-Based Detection**:

```
Final Score = (0.4 × RoBERTa_Score) + (0.6 × Heuristic_Score)
```

**Component 1: RoBERTa Classifier (40% weight)**
- Captures semantic patterns
- Learns from fine-tuning on AI-generated content
- Probability-based output

**Component 2: Heuristic Detector (60% weight)**
- Analyzes linguistic features
- Rule-based scoring
- Interpretable results

**Heuristic Features Analyzed**:
| Feature | Detection Logic |
|---------|---|
| **Burstiness** | StdDev of words per sentence (AI-generated text is more uniform) |
| **Vocabulary Diversity** | Unique words / Total words ratio |
| **Sentence Length** | Average words per sentence (AI tends toward similar lengths) |
| **Text Length** | Total character count (too short = uncertain) |
| **Repetition Patterns** | Frequency of repeated phrases |

#### Text Preprocessing Pipeline
```
Raw Text
    ↓
Remove URLs & HTML tags
    ↓
Strip markdown formatting
    ↓
Normalize whitespace
    ↓
Remove special characters
    ↓
Tokenization (max 512 tokens)
    ↓
Ready for inference
```

### 2. Image Detection Model

#### Model Selection: EfficientNet B0
- **Reason**: Lightweight yet powerful, ideal for deployment
- **Base**: ImageNet pre-trained (torchvision)
- **Trade-off**: Speed vs. Accuracy

| Parameter | Value |
|-----------|-------|
| Base Model | EfficientNet B0 |
| Backbone | ImageNet Pre-trained |
| Input Size | 224×224×3 (RGB) |
| Custom Classifier Head | GlobalAvgPool → Dense(256) → ReLU → Dropout(0.3) → Dense(2) |
| Total Parameters | ~4M |
| Output | Binary classification (AI/Real) |
| Decision Threshold | 0.50 |

#### Training Methodology: 2-Phase Transfer Learning

**Phase 1: Feature Learning (20 epochs)**
- Freeze backbone layers
- Train only custom head
- Optimizer: SGD (LR=0.001)
- Purpose: Learn task-specific features on frozen representations
- Result: Prevents overfitting to small dataset

**Phase 2: Fine-tuning (15 epochs)**
- Unfreeze all backbone layers
- Train entire network
- Optimizer: Adam (LR=1e-4)
- Lower learning rate to preserve pre-trained knowledge
- Purpose: Adapt pre-trained features to AI-generated images

#### Image Preprocessing & Augmentation
```
Raw Image
    ↓
Resize to 256×256
    ↓
Center crop to 224×224
    ↓
Normalize (ImageNet stats)
    ├─ Mean: [0.485, 0.456, 0.406]
    └─ Std: [0.229, 0.224, 0.225]
    ↓
Augmentation (Training only):
├─ Random rotation (±15°)
├─ Color jitter (brightness/contrast)
└─ Horizontal flip (50% probability)
    ↓
Tensor conversion
```

---

## Experimental Results & Performance Analysis

### Text Model Results

#### Training Progression
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|-----------|----------|
| 1 | 85.2% | 87.3% | 0.412 | 0.356 |
| 2 | 88.7% | 89.1% | 0.268 | 0.301 |
| 3 | 90.4% | 90.0% | 0.195 | 0.287 |

#### Final Performance Metrics
| Metric | Score |
|--------|-------|
| **Test Accuracy** | **90.0%** |
| **Precision (AI-Generated)** | 88% |
| **Recall (AI-Generated)** | 91% |
| **F1-Score** | 0.89 |
| **Inference Time** | < 1 second |

#### Confusion Matrix Analysis
```
                Predicted
              AI      Human
Actual AI     8,180   810     (Recall: 91%)
       Human  240    8,770   (Specificity: 97%)
              ↓
           Precision: 97%
```

**Key Observations**:
- Model is conservative: High precision in detecting AI
- Better at identifying AI-generated content (91% recall)
- Low false positive rate for human content (3% error)
- Good generalization (minimal gap between train/val)

### Image Model Results

#### Phase-wise Performance
| Phase | Epochs | Accuracy | Strategy |
|-------|--------|----------|----------|
| Phase 1 (Head Training) | 20 | 88% | Frozen backbone |
| Phase 2 (Fine-tuning) | 15 | 94% | Unfrozen backbone |
| **Final (Benchmark)** | - | **96%** | Combined |

#### Real-World Performance (Test Set)
| Metric | Score |
|--------|-------|
| **Test Accuracy (Benchmark)** | 96% |
| **Real-world Accuracy** | 70% |
| **Precision** | 82% |
| **Recall** | 75% |
| **Inference Time** | < 2 seconds |

**Performance Gap Analysis**:
- 26% gap between benchmark (96%) and real-world (70%)
- Reasons:
  - Benchmark dataset: curated, high-quality images
  - Real-world: compressed, edited, mixed-quality images
  - Domain shift: Different AI models used after training
  - External artifacts: Watermarks, filters affecting detection

#### Confusion Matrix (Real-world)
```
                Predicted
              AI      Real
Actual AI     1,638   612     (Recall: 73%)
       Real   308    1,692   (Specificity: 85%)
              ↓
           Precision: 84%
```

---

## Methodology & Approach

### Data Preprocessing
1. **Text Cleaning**:
   - URL removal
   - HTML tag stripping
   - Markdown normalization
   - Whitespace standardization

2. **Image Preprocessing**:
   - Resize standardization
   - Normalization with ImageNet stats
   - Augmentation for training robustness

### Model Training Strategy

**Text Model**:
- Loss Function: CrossEntropyLoss with label smoothing (0.1)
- Optimizer: AdamW with weight decay (0.01)
- Scheduler: Linear warmup (10% of steps)
- Mixed Precision: For 2x speedup
- Early Stopping: Patience of 2 epochs

**Image Model**:
- Loss Function: CrossEntropyLoss
- Phase 1 Optimizer: SGD (LR=0.001)
- Phase 2 Optimizer: Adam (LR=1e-4)
- Data Balancing: Undersampling to 1:1 ratio
- Batch Size: 32 for both phases

### Hyperparameter Justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Batch Size (Text) | 16 | Memory efficiency for 512-token sequences |
| Learning Rate (Text) | 1e-5 | Conservative fine-tuning to preserve pre-training |
| Epochs (Text) | 3 | Avoid overfitting on relatively small dataset |
| Label Smoothing | 0.1 | Prevent model overconfidence |
| Max Tokens | 512 | RoBERTa standard; balance coverage vs. computation |
| Image Size | 224×224 | EfficientNet standard; trade-off between speed/accuracy |
| Phase 1 Epochs | 20 | Sufficient for head-only training |
| Phase 2 Epochs | 15 | Fine-tune without catastrophic forgetting |

---

## Key Findings & Insights

### What the Models Learned

**Text Model Insights**:
- AI-generated text exhibits **higher uniformity** in sentence length
- **Vocabulary diversity** is typically lower in AI content
- AI models tend to use more **formal, structured language**
- Heuristic features provide **70% of decision weight** for robustness

**Image Model Insights**:
- AI-generated images have **subtle pixel inconsistencies**
- **Lighting patterns** differ between AI and real images
- AI images show **unusual symmetry** in certain domains
- **Compression artifacts** provide discriminative features

### Performance Limitations

1. **Text Model**:
   - May flag non-native English writing as AI
   - Struggles with heavily paraphrased AI content
   - Short texts (< 50 tokens) produce uncertain predictions
   - Different AI models may have varying detection rates

2. **Image Model**:
   - Accuracy drops on heavily edited/compressed images
   - Struggles with AI models introduced **after training**
   - Watermarked images can confuse detection
   - Real-world gap due to domain shift

---

## Application & Use Cases

### Suitable Applications
- Educational institutions (essay verification)
- Content moderation platforms
- Fact-checking services
- Publishing houses (article authenticity)
- Social media content verification
- Research paper screening

### Current Limitations
- Not 100% reliable - use as **screening tool, not final verdict**
- Requires human review for critical decisions
- May fail on adversarially crafted content
- Needs retraining for new AI models

---

## Project Metadata

| Field | Value |
|-------|-------|
| **University** | K.R. Mangalam University |
| **Program** | Master of Computer Applications (MCA) |
| **Academic Year** | 2025-2026 |
| **Project Type** | Pre-Final Year Capstone |
| **Team ID** | 26E1053 |
| **Faculty Mentor** | Ms. Neha Kaushik |
| **Team Size** | 5 members |
| **Duration** | 6 months |
| **Status** | 90% Complete (Testing phase) |
| **Submission Date** | March 2026 |

---

## References & Academic Sources

### Datasets
| Dataset | Source | Citation |
|---------|--------|----------|
| HC3 | HuggingFace | [Hub Link](https://huggingface.co/datasets/Hello-SimpleAI/HC3) |
| MEGA | Kaggle | [Kaggle Link](https://www.kaggle.com/datasets/mobildevices/megadatasetv2) |
| Defactify COCOAI | HuggingFace | [Hub Link](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) |

### Pre-trained Models
| Model | Framework | Source |
|-------|-----------|--------|
| RoBERTa | HuggingFace Transformers | [Model Hub](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta) |
| EfficientNet B0 | PyTorch torchvision | [Official Docs](https://pytorch.org/vision/stable/models.html) |

### Related Research
- Solaiman et al., "Language Models are Unsupervised Multitask Learners" (GPT-2)
- OpenAI GPT-3 & ChatGPT Technical Reports
- Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs"
- Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach"

---

## Learning Outcomes

Through this project, the team gained expertise in:

1. **Deep Learning**:
   - Transfer learning and fine-tuning strategies
   - Ensemble methods and model combination
   - Mixed precision training for efficiency

2. **NLP**:
   - Transformer-based architectures (RoBERTa)
   - Tokenization and text preprocessing
   - Feature engineering for text analysis

3. **Computer Vision**:
   - CNN architecture (EfficientNet)
   - Image augmentation and normalization
   - Phase-wise training for optimization

4. **Project Management**:
   - Team collaboration and version control
   - Iterative development and testing
   - Documentation and presentation

5. **Production Deployment**:
   - Streamlit web application development
   - Model serving and inference optimization
   - Cloud deployment considerations

--- 

**AI Content Authenticity Detection Team**

*"Advancing content authenticity detection through machine learning and ensemble methods"*
