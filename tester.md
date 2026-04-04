# QA, Evaluation & Validation Strategy

**Role:** Lead ML Test Engineer

## 🧪 Methodological Testing Protocols

### 1. Functional Integrity Testing (`test_cvusa.py`, `test.py`)
- Extract feature embeddings (base dimensionality `2048` or conditionally LPN-derived shapes `2048 * blocks`) leveraging the dynamically selected frozen weights in `two_view_net` or `two_view_net_rgbd`.
- Safely assure embeddings undergo an active `L2 Normalization` stage internally upon the evaluation hook's initialization line.
- Execute metric computations calculating Euclidean Distance (or strict Cosine Similarity) maps contrasting generated query features against the stored gallery tensor arrays.

### 2. Key Performance Indicators (`evaluate_gpu.py`)
- **Recall@K (R@1, R@5, R@10, R@1%)**: The primary statistical measure demonstrating direct accuracy success.
- **mAP (Mean Average Precision)**: The macro-averaging metric highlighting generic network rank quality spanning all categories.
- *Failing Benchmark Trigger:* Tests assert a failure condition if ANY experimental structure scores an integrated R@1 < 50% relying on standard validation setups.

## 👁️ Visual System Validation Process
Metrics can obscure systematic behavioral faults. Explainability is mandated for this architecture prior to merging.
1. **Hook Interception (`gradcam_visualization.py`)**: Seamlessly attach backward/forward PyTorch hooks accessing the highest semantic convolution (`layer4`).
2. **Visual Delta Script (`compare_gradcam.py`)**: Plot high-res comparison grids compiling:
   - Evaluated Query Source Image + Corresponding Query Heatmap.
   - Evaluated Satellite Domain Image + Satellite Network Heatmap.
   - MiDaS Extracted Depth Map Reference point.
- **Goal Verification:** Validate mathematically/visually if structural priors genuinely retarget model focus away from shifting phenomena (e.g. foliage/cars) to fixed road schemas.

## 🚧 Known Edge Cases & Mitigation Matrix

| Testing Scenario | Technical Action / Case Injection | Required Expected Behavior |
| :--- | :--- | :--- |
| **Missing Proxy Depth Channels** | Delete a subset of pre-generated `.pkl`/`.jpg` MiDaS datasets to force read faults. | Dataloader logic must gracefully catch exceptions, instantiate a zero-array tensor placeholder, output a standard shell warning string, and prevent any runtime crashes. |
| **160k Scale Gallery Operations** | Deliberately pack 160K random unverified image inputs into memory processing contexts. | Prevent Out-Of-Memory (OOM) fatal kills. Matrix sorting MUST dynamically shard calculations to CPU memory when 16GB GPU VRAM allocation boundaries trigger limit warnings. |
| **LPN Dimension Mismath Injection** | Input `--pool lpn` launch argument, but maliciously pass an outdated Generalized Mean (GeM) architecture weights dictionary file. | `load_network()` MUST properly execute PyTorch structural audits internally, intercept the severe tensor shape imbalance, throw assertion faults, and aggressively halt execution. |
