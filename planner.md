# Sprint & Resource Planning

**Role:** Agile Project Planner

## 🏃‍♂️ Sprint Workflow Overview

### Sprint 1: Baseline RGB Training & Verification
**Goal:** Establish a solid ground-truth baseline leveraging standard 3-channel resilient ResNet-50 streams.
- **Tasks:**
  - Verify dataloaders for the CVUSA dataset alignment.
  - Execute the command via `train_baseline_rgb.sh` to begin model fitting.
  - Rigorously validate evaluation metrics on the 4K validation gallery using `test.py`.
- **Deliverable:** Fully functional and pre-trained RGB baseline model checkpoints (`.pth`), with recorded Tensorboard/Plot data.

### Sprint 2: RGBD Pipeline Setup & MiDaS Integration
**Goal:** Evaluate and prove the hypothesis that structural depth priors yield improved top-k retrieval performance.
- **Dependencies:** 
  - **CRITICAL BLOCKER:** Depth map proxies *must* be successfully generated for the entire satellite dataset split *before* launching the training subsystem.
- **Tasks:**
  - Execute batch processing MiDaS extraction over all satellite sub-directories.
  - Dry run the `model_rgbd.py` forward pass to intercept and solve shape/tensor mismatches.
  - Launch standard long-running `sh train_rgbd.sh` (wraps `train.py --use_rgbd`).
- **Deliverable:** Fully trained RGBD dual-stream checkpoint + comparative QA report versus Sprint 1 baseline results.

### Sprint 3: Visual Interpretability & Grad-CAM QA
**Goal:** Scientifically understand *why* and *how* the RGBD stream operates differently inside the black-box CNN.
- **Tasks:**
  - Write and standardize hooking mechanisms for `gradcam_visualization.py` on `layer4` of ResNet.
  - Programmatically contrast edge-case situations (where RGB fails but RGBD hits Target Top-1) and extract output heatmaps.
- **Deliverable:** A documented matrix of attention visual maps.

### Sprint 4: GPU Re-ranking Extension Optimization
**Goal:** Supercharge inference scale past standard computational constraints using Graph Neural Networks.
- **Tasks:**
  - Audit C++ and dependencies and compile the `re_ranking` GNN CUDA extension smoothly against system headers.
  - Adapt native Python evaluation outputs into the CUDA bridge.
  - Stress test massively scaled evaluation queries (160k subset).
- **Deliverable:** Highly parallelized end-to-end evaluation pipeline with ultra-fast sorting operations.

## 🚥 Execution Strategy
1. **Config Management:** Continually use YAML parsing configurations natively bridged via argparse for highly decoupled scripts.
2. **Resource Allocation Guidelines:** Compute environments utilizing standard instances (e.g., T4/V100) are reserved for Sprint 1 and Sprint 3 inference tasks, while heavier processing allocations (A100 instances) will be redirected for Sprint 2 iterations and Sprint 4 database querying.
