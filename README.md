# TxtVis: Multimodal Latent Space Representation & Analysis Studio

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

## 📖 Abstract
**TxtVis** is a dual-mode, PyTorch-based research environment engineered for the extraction, training, and visualization of high-dimensional embeddings across both text (Natural Language Processing) and image (Computer Vision) modalities. 

Designed for computational linguistics and computer vision research, TxtVis bridges the gap between model training and latent space interpretability. It features a custom, from-scratch PyTorch implementation of the Skip-Gram architecture for generating dense word vectors from raw, multilingual corpora, alongside a dual-path visual pipeline utilizing both a transfer-learning ResNet-18 model and a bespoke "Tabula Rasa" (from-scratch) metric learning network. Both modalities are visualized through a real-time, interactive PCA dimensionality reduction dashboard.

---

## 🔬 Research & Academic Applicability
This repository demonstrates a rigorous understanding of representation learning paradigms. It is structured to facilitate experimental research in:
*   **Multilingual Semantic Mapping:** Analyzing how non-Latin scripts (e.g., Bengali, Hindi, Arabic) map geometrically in a continuous vector space without relying on pre-trained English-centric tokenizers.
*   **Unbiased Latent Engineering:** Constructing topological spaces entirely from scratch using localized datasets to prevent the domain biases inherent in massive pre-trained models.
*   **Manifold Visualization:** Observing topological structures of learned representations through interactive Principal Component Analysis (PCA) projections.

---

## 🛠️ Tabula Rasa Metric Learning: The "Dummy" Network Framework

A core academic contribution of this project is the implementation of a foundational, untrained "dummy" neural network architecture. Rather than relying exclusively on transfer learning, this bespoke framework is designed to generate highly specialized similarity embeddings entirely from scratch. This approach is critical for specialized research domains (e.g., highly localized linguistic dialects or specialized microscopic/medical imagery) where pre-trained models introduce unwanted macro-domain bias.

### 1. Custom Text Embeddings (Closed-Domain Topology)
Unlike generalized models like GloVe or BERT, our custom NLP pipeline forces the network to construct a unique topological space based *exclusively* on the localized corpus provided by the researcher. 
*   **Initialization:** The untrained linear embedding matrix is initialized using a uniform distribution $W \sim \mathcal{U}(-\frac{1}{\sqrt{d}}, \frac{1}{\sqrt{d}})$.
*   **Objective:** The network is optimized using Cross-Entropy Loss to minimize the negative log-likelihood of observing the actual context words given the center word.
$$ \mathcal{L} = - \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j \neq 0} \log P(w_{t+j} | w_t) $$
This ensures absolute domain specificity, proving the capability to engineer the fundamental calculus of text-based latent spaces independent of massive pre-existing datasets.

### 2. Shallow Siamese Network for Visual Similarity (Custom CV Embeddings)
For specific visual data, TxtVis bypasses heavy architectures (like ResNet) to employ a lightweight, custom-built Convolutional Neural Network (CNN) trained via Metric Learning. This foundational "dummy" network utilizes a Siamese architecture to learn a custom distance metric from the ground up.
*   **Architecture Details:** A 3-layer convolutional base (Conv2D $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool) flattened into a dense projection layer $f_{\theta}(x) \in \mathbb{R}^{d}$.
*   **Mathematical Objective (Contrastive Loss):** The network does not classify; instead, it is optimized using a Contrastive Loss function, mathematically pushing similar images closer together and pulling dissimilar images apart in the $d$-dimensional manifold:
$$ \mathcal{L}_{contrastive}(W, Y, \vec{X_1}, \vec{X_2}) = (1 - Y) \frac{1}{2} (D_W)^2 + (Y) \frac{1}{2} \{ \max(0, m - D_W) \}^2 $$
where $D_W = \| f_{\theta}(\vec{X_1}) - f_{\theta}(\vec{X_2}) \|_2$ represents the Euclidean distance in the embedding space, $Y \in \{0, 1\}$ is the binary similarity label, and $m$ is the margin. 

---

## 🧠 Transfer Learning Module: Headless Residual Networks
Alongside the custom foundational networks, the CV pipeline also supports robust feature extraction using a deep Convolutional Neural Network (CNN) for generalized tasks.

#### **Architecture Details**
*   **Base Model:** ResNet-18 pre-trained on ImageNet (1.2 million images, 1000 classes).
*   **Residual Blocks:** Utilizes skip connections to mitigate the vanishing gradient problem in deep networks, formulated as $y = \mathcal{F}(x, \{W_i\}) + x$, allowing the network to learn identity mappings.
*   **Latent Output:** The classification head is programmatically truncated. Images passed through the network terminate at the Global Average Pooling (GAP) layer, yielding a translation-invariant, semantically dense vector $v \in \mathbb{R}^{512}$.

---

## 📐 Vector Space Analysis & Metrics

To interpret the high-dimensional geometry produced by the neural networks, TxtVis employs strict mathematical evaluation methods:

### Cosine Similarity (Semantic & Visual Distance)
Similarity between two entities (whether words or images) is computed purely on the orientation of their vectors, invariant to their magnitude. Given vectors $A$ and $B$:

$$ S_C(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}} $$

*   In the **NLP workspace**, this isolates words that appear in similar grammatical or semantic contexts.
*   In the **CV workspace**, this powers the Content-Based Reverse Image Search, finding images with the nearest spatial and textural features.

### Principal Component Analysis (PCA)
For human interpretability, the $d$-dimensional embedding space is orthogonally transformed into a 2D or 3D coordinate system. The covariance matrix of the data is eigendecomposed, and the vectors are projected onto the eigenvectors associated with the largest eigenvalues, maximizing data variance while minimizing lower-dimensional information loss.

---

## 📊 System Design & Telemetry

*   **Custom Training Loops:** Bypasses abstraction wrappers to execute native PyTorch training loops, utilizing `torch.utils.data.DataLoader` for optimized batch processing and GPU memory management.
*   **Asynchronous UI Updates:** The Streamlit rendering engine is hooked directly into the PyTorch epoch loop, forcing live telemetry updates (Loss curves, $\Delta$ Loss, Epoch Tracking) without bottlenecking tensor operations.
*   **Unicode-First Preprocessing:** Employs rigorous regex-based tokenization (`re.UNICODE`) to ensure unbiased processing of complex, non-Latin typographies (e.g., Bengali conjuncts).

---

## ⚙️ Installation & Usage

### Prerequisites
*   Python 3.10+
*   CUDA Toolkit (Optional, but recommended for GPU acceleration)

### Setup
Clone the repository, install the dependencies, and launch the workspace:

```bash
git clone [https://github.com/yourusername/txtvis.git](https://github.com/yourusername/txtvis.git)
cd txtvis
pip install -r requirements.txt
streamlit run app.py

### 🐳 Run via Docker
You can run the entire TxtVis environment instantly without installing local dependencies using Docker:

```bash
docker run -p 8501:8501 yourusername/txtvis:latest
