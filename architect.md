# System Architecture & Design Decisions

**Role:** Principal AI Architect

## 🏛️ Core Architecture Design

### RGB Branch (`model.py`)
Our baseline features a two-stream Siamese network built on an ImageNet-pretrained ResNet-50 backbone.
- **Pooling Strategy**: Configurable between Global Average Pooling (GAP), Generalized Mean (GeM), and Local Pattern Network (LPN) modules to extract both global semantics and distinct scale features.
- **Neck Design**: We utilize a BNNeck (ClassBlock) to stabilize features before metric loss evaluation. Removing the ReLU activation after BatchNorm promotes a smoother feature space that is structurally sound metric-wise.

### RGBD Branch (`model_rgbd.py`)
To incorporate structural and geometric priors from overhead satellite imagery, we extend the satellite branch to support 4 input channels.
- **4-Channel Input Logic**: A redesigned first convolutional layer (`conv1`) is specified to accept `(B, 4, H, W)`. We transfer the original RGB weights from standard ResNet-50, and compute the 4th channel (Depth) weight as the mean of the RGB channels, divided by 3. This mathematical strategy ensures consistent gradient flow and prevents activation dimension blowup at epoch 0.
- **Isolating Streams**: The counterpart drone/street stream remains 3-channel RGB. By design, only the satellite domain leverages the MiDaS depth proxy.

## 📏 Metric Learning Strategy
We deploy **Circle Loss** as the primary loss to optimize the deep representation space.
- CircleLoss dynamically re-weights the anchor-positive and anchor-negative similarity scores, penalizing confidently wrong pairings exponentially harder than easily distinguishable ones.
- Features are L2 normalized before cosine similarity geometry computes the separation margins.

## 🚀 GPU-Based GNN Re-ranking System
For large-scale retrieval applications (e.g., matching across 160K database samples), basic Euclidean distance searches suffer from massive CPU-bound sorting times. 
- **The Design**: We reformulate post-retrieval re-ranking as an intricate graph node classification problem. The Gallery forms a graph structure where node edges denote raw cosine similarities.
- **Acceleration Paradigm**: Using specifically compiled CUDA extensions, the system computes the complex adjacency matrix and feature propagation purely down at the GPU level, delivering significant real-time performance boosts.

## 📁 Repository Structure & Data Flow
```text
Raw CVUSA ──> MiDaS Depth Generator ──> (RGB + Depth Proxy Maps) 
                                              │
                                              ▼
                        Data Layer (`dataset_rgbd.py`)
                                              │
                                              ▼
                        Model Layer (`model_rgbd.py`) ──────> Grad-CAM Vis Layer
                                              │               (`gradcam_visualization.py`)
                                              ▼
                        Training Layer (`train.py` w/ CircleLoss & FP16)
                                              │
                                              ▼
             Evaluation Layer (`test.py`, `evaluate_gpu.py`) ──> Retrieval Metrics (mAP)
                                              │
                                              ▼
                        GNN Re-ranking Subsystem (CUDA GNN)
```
