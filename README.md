# **Uncertain Missing Modality Audio-Visual Classification Framework**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Pytorch](https://img.shields.io/badge/pytorch-1.10%2B-red.svg)

## 🚀 **Overview**
This repository contains an **Audio-Visual Classification Framework** designed to handle **Uncertain Missing Modality** scenarios where missing modalities are unpredictable at test time. Our approach integrates **Prompt Learning** at both the **Input Level** and **Attention Level**, allowing the model to dynamically adapt to missing or noisy modalities.

## 🔥 **Key Contributions**
- **✅ End-to-End Framework for Uncertain Missing Modality**
  - Designed to handle **unpredictable modality loss** by training across multiple missing modality scenarios.
- **🎯 Prompt Learning for Robustness**
  - Introduces **Input-Level and Attention-Level Prompts** to reinforce missing modality adaptation.
- **💡 Efficient Training**
  - Reduces **memory usage by 82.3%** and **training time by 96%**, making it highly scalable.
- **📈 Performance Improvement**
  - Outperforms Fine-Tuning in **noisy and missing modality environments** by up to **10%**.

---

## 📌 **Motivation**
### **Challenges in Audio-Visual Classification**
Multimodal classification models often face:
- **❌ Missing Modality** (Sensor failure, transmission issues)
- **🔊 Noise** (Background noise, corrupted video frames)
- **🔄 Uncertain Missing Modality** (Test-time unpredictability in missing data)

### **Why Prompt Learning?**
Traditional **Fine-Tuning** adjusts all model parameters, making it computationally expensive. Instead, **Prompt Learning**:
- **Efficiently updates only learnable prompt tokens** (low memory & fast training)
- **Enhances modality interaction**, compensating for missing or noisy data.

---
## ⚙️ **Framework Overview**
![image](https://github.com/user-attachments/assets/0e39ca01-fa8f-40a0-98b3-2634118be8f9)

The framework is designed to address **Uncertain Missing Modality** scenarios using a robust integration of **learnable prompt tokens** at both the input and attention levels. This allows the model to adaptively handle incomplete or noisy data across modalities while maintaining computational efficiency.


### **1️⃣ Input-Level Prompt Integration**
![image](https://github.com/user-attachments/assets/45f46dbf-0446-40fe-a42b-06257a1dc56a)

**Description**: At the input stage, **learnable prompt tokens** are concatenated directly with the input features of each modality (audio and visual). This mechanism allows the model to embed prior knowledge about modality-specific patterns (e.g., noise or missing data) directly into the input representation.

- **Why it matters**:
  - The prompts act as auxiliary inputs that encode modality-specific signals, such as noise patterns or missing modality indicators.
  - Ensures each modality's encoder processes enriched inputs with context about the data's state.

- **Process**:
  - Learnable tokens (`prompt_a` for audio, `prompt_v` for visual) are **replicated to match the batch size**.
  - Tokens are **concatenated** with the input feature embeddings before being passed to modality-specific encoders.


### **2️⃣ Attention-Level Prompt Integration**
![image](https://github.com/user-attachments/assets/9b85d6c3-79b6-4c86-9271-53bbc0995710)

**Description**: Prompts are incorporated into **Cross-Attention layers** during the fusion phase. These prompts serve as **Key** and **Value** inputs in the attention mechanism, enabling enhanced interaction between audio and visual modalities.

- **Why it matters**:
  - Prompts improve information flow between modalities, ensuring that missing or noisy inputs are supplemented by the available modality.
  - Facilitates robust feature alignment, especially in noisy or incomplete scenarios.

- **Key Highlights**:
  - **Query**: Comes from one modality (e.g., audio embeddings).
  - **Key & Value**: Combines the corresponding modality embeddings and learnable prompt tokens.
  - Enables dynamic adjustment of attention weights based on the quality and completeness of input data.


### **3️⃣ Fusion Module**
![image](https://github.com/user-attachments/assets/b0ef7f20-3609-4b5b-b155-3d8c06de015d)

**Description**: The Fusion Module introduces **Cross-Attention** layers to effectively balance contributions from both modalities. This module resolves the natural imbalance caused by varying sequence lengths and noise levels in audio and visual data.

- **Why it matters**:
  - Aligns features from different modalities, ensuring mutual reinforcement and minimizing bias toward any single modality.
  - Handles discrepancies in sequence lengths (e.g., longer audio sequences vs. shorter visual sequences) through flexible attention mechanisms.

- **Structure**:
  - Two separate **Cross-Attention layers**:
    - **Visual-to-Audio**: Visual embeddings are used as queries to retrieve relevant information from audio embeddings and their associated prompts.
    - **Audio-to-Visual**: Audio embeddings serve as queries to access complementary visual features and their prompts.


---

## 📊 **Datasets**
### **Pre-Training Datasets**
- **AudioSet**: 1.7M videos, 632 classes.
- **VGGSound**: 200K+ videos, 300 classes.

### **Fine-Tuning Dataset**
- **UrbanSound8K-AV**: 8,732 samples, 10 classes (audio + visual).

---

## 🎯 **Training & Evaluation**
### **Training**
![image](https://github.com/user-attachments/assets/83e9fd24-87c0-44aa-b883-b49fd75874e4)

- **4 Training Scenarios:**
  - ✅ Complete (Audio + Visual)
  - 🎥 Vision Only (Noisy Audio)
  - 🎵 Audio Only (Noisy Visual)
  - ❌ Noise to Both (Noisy Audio + Visual)

```python
python train.py --dataset UrbanSound8K-AV --epochs 50 --batch_size 16 --lr 1e-4
```

### **Evaluation**
![image](https://github.com/user-attachments/assets/63c2fee6-42ed-4239-919b-d30afa48992e)

- Uses **all learned prompts concatenated** to handle **Uncertain Missing Modality**.

```python
python evaluation.py --dataset UrbanSound8K-AV --case noise_to_both
```

---

## 📈 **Results**
### **Performance Comparison**
| Case                     | Fine-Tuning (FT) | FT + Prompt Learning (PL) | Improvement |
|--------------------------|------------------|----------------------------|-------------|
| ✅ Complete               | 0.99            | 0.99                       | -           |
| 🎥 Vision Only (Noisy A)  | 0.69            | 0.79                       | +0.10       |
| 🎵 Audio Only (Noisy V)   | 0.83            | 0.86                       | +0.03       |
| ❌ Noise to Both         | 0.71            | 0.80                       | +0.09       |

### **Resource Efficiency**
| Method         | Total Memory (GiB) | Training Memory (GiB) | Memory Saving | Time per Epoch |
|----------------|---------------------|-----------------------|---------------|----------------|
| Fine-Tuning    | 95.12              | 93.89                | -             | 1 min          |
| Prompt Learning| 17.85              | 13.62                | **82.3%**     | **2.4 sec**    |

---

## 🛠️ **Installation**
1️⃣ Clone the repository:
```bash
git clone https://github.com/your-repo-name/Uncertain-Modality-AV.git
cd Uncertain-Modality-AV
```
2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔍 **Future Work**
- Extend to **other multimodal tasks** (video classification, captioning).
- Explore **alternative prompt learning strategies** for robustness.
- Optimize for **real-world deployment** in low-resource settings.



