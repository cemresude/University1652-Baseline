# Project Overview & Master Tracker

**Role:** Product Manager
**Project:** AI-Based Framework for Cross-View Geo-Localization

## 🗺️ High-Level Roadmap & Milestones
- **Phase 1: Baseline Architecture** - Establish robust standard RGB cross-view retrieval.
- **Phase 2: RGBD Modality Integration** - Incorporate MiDaS depth estimates as a 4th channel.
- **Phase 3: Visual Interpretability** - Deploy Grad-CAM/Grad-CAM++ to analyze focus areas.
- **Phase 4: GNN Re-ranking** - Implement GPU-accelerated GNN for large-scale post-processing.
- **Phase 5: Large-scale Evaluation** - Scale to 160k gallery sets and finalize metrics.

## 📦 Project Inventory Status
- [x] Base ResNet-50 Feature Extractor (`model.py`)
- [x] LPN (Local Pattern Network) Architecture Integration
- [x] Training pipeline for CVUSA/University-1652 (`train_cvusa.py`, `train.py`)
- [x] RGBD Forward Pass and 4-channel weights adaptation (`model_rgbd.py`)
- [ ] MiDaS Depth Generation Pipeline Automation
- [ ] GNN Re-ranking CUDA extension compilation & integration
- [ ] Grad-CAM evaluation scripts

## 📋 Task Backlog by Functional Layer

### 1. Data Preparation
- [ ] Automate MiDaS depth map proxy generation for CVUSA satellite imagery.
- [ ] Validate RGB and Depth alignment and scaling.

### 2. Modeling
- [x] Implement GeM pooling and ClassBlock BNNeck.
- [x] Adapt `conv1` for 4-channel input (RGBD).
- [x] Integrate LPN capabilities for L-dimensional localized feature maps.

### 3. Training
- [ ] Execute hyperparameter sweep for CircleLoss margins.
- [ ] Finalize custom learning rate scheduler adjustments for 4-channel early layers.

### 4. Testing & Evaluation
- [ ] Run base evaluation (R@1, R@5, R@10, R@1%) on 4K gallery.
- [ ] Extend evaluation scale to 160K images.

### 5. Visualization
- [ ] Implement `gradcam_visualization.py`.
- [ ] Create `compare_gradcam.py` side-by-side comparison for RGB vs. RGBD.

### 6. Deployment (Re-ranking)
- [ ] Compile CUDA extensions for GNN subsystem.
- [ ] Integrate GNN inference efficiently within the evaluation loop.
