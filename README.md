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

At the input stage, **learnable prompt tokens** are concatenated directly with the input features of each modality (audio and visual). This mechanism allows the model to embed prior knowledge about modality-specific patterns (e.g., noise or missing data) directly into the input representation.

- **Why it matters**:
  - The prompts act as auxiliary inputs that encode modality-specific signals, such as noise patterns or missing modality indicators.
  - Ensures each modality's encoder processes enriched inputs with context about the data's state.


### **2️⃣ Attention-Level Prompt Integration**
![image](https://github.com/user-attachments/assets/9b85d6c3-79b6-4c86-9271-53bbc0995710)

Prompts from the **Input Level** stage are utilized as **Key** and **Value** inputs in the **Cross-Attention** mechanism during the fusion phase. This integration enables enhanced interaction between audio and visual modalities by leveraging both learnable tokens and modality-specific embeddings.

- **Why it matters**:
  - **Enhanced Cross-Modal Information Flow**:
     - Prompts ensure that missing or noisy modality information is supplemented by the complementary modality.  
     - This improves the robustness of the model, especially in **Uncertain Missing Modality** scenarios where one or both modalities might be degraded or unavailable.
  - **Robust Feature Alignment**: 
     - Prompts facilitate better alignment between modalities by acting as an intermediary that strengthens the representation of shared information.  
     - This ensures that even when one modality is compromised, the overall feature alignment remains strong.
      
- **Key Highlights**:
  - **Query**: The Query originates from one modality's embeddings (e.g., audio embeddings in Audio-to-Visual Cross-Attention or visual embeddings in Visual-to-Audio Cross-Attention).
  - **Key & Value**: Combines the corresponding modality embeddings and learnable prompt tokens.
     - Corresponding modality embeddings (e.g., audio embeddings for Visual-to-Audio attention).  
     - Learnable prompt tokens from the **Input Level Integration**.
  - **Enhanced Fusion**:  
     - Prompts act as contextual signals that bridge modality-specific representations, ensuring efficient cross-modal feature exchange.

### **3️⃣ Fusion Module**
![image](https://github.com/user-attachments/assets/b0ef7f20-3609-4b5b-b155-3d8c06de015d)

The Fusion Module introduces **Cross-Attention** layers to effectively balance contributions from both modalities. This module resolves the natural imbalance caused by varying sequence lengths and noise levels in audio and visual data.

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

#### **Key Insights:**
1. **Complete Case**:  
   - Both Fine-Tuning (FT) and Prompt Learning (PL) achieve near-perfect performance.  
   - Indicates that Prompt Learning does not degrade performance in ideal conditions despite being computationally more efficient.

2. **Vision Only (Noisy Audio)**:  
   - PL demonstrates significant improvement (+0.10) over FT by leveraging visual features more effectively through cross-attention and prompts.  
   - Highlights the robustness of PL in compensating for noisy audio data by emphasizing the complementary modality.

3. **Audio Only (Noisy Visual)**:  
   - Improvement is smaller (+0.03) but still notable.  
   - Reflects that audio data inherently carries less noise sensitivity, and prompts enhance robustness without major dependency on visual data.

4. **Noise to Both**:  
   - PL provides a substantial gain (+0.09) in the most challenging scenario.  
   - Demonstrates the ability of prompts to optimize cross-modal interactions, ensuring stable performance even under severe noise.


### **Resource Efficiency**

| Method         | Total Memory (GiB) | Training Memory (GiB) | Memory Saving | Time per Epoch |
|----------------|---------------------|-----------------------|---------------|----------------|
| Fine-Tuning    | 95.12              | 93.89                | -             | 1 min          |
| Prompt Learning| 17.85              | 13.62                | **82.3%**     | **2.4 sec**    |

#### **Key Insights:**
1. **Memory Usage**:  
   - PL significantly reduces total memory usage by **82.3%**, lowering computational demands.  
   - This is achieved by learning only a small set of prompt parameters, unlike FT, which updates the entire model.

2. **Training Memory**:  
   - PL uses **13.62 GiB** compared to **93.89 GiB** in FT.  
   - Such drastic memory savings make PL scalable for larger datasets and models, particularly in resource-constrained environments.

3. **Training Time**:  
   - PL requires only **2.4 seconds per epoch**, a **96% reduction** compared to FT (1 minute per epoch).  
   - This efficiency is particularly critical for large-scale or real-time applications where training time is a bottleneck.


### 📊 **Overall Analysis**
Prompt Learning (PL) not only achieves competitive or superior performance compared to Fine-Tuning (FT) in noisy and missing modality scenarios but also drastically improves computational efficiency. These results establish PL as a practical and scalable solution for multimodal learning in resource-constrained or real-world environments.

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





