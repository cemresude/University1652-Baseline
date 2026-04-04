# Get Shit Done (GSD) - Actionable Dev Tasks

**Role:** Lead Developer / GSD Engineer

## 🔥 Immediate Action List & Focal Execution

### 1. Fix CUDA GNN Compilation Pipeline
**Problem:** The GPU Re-ranking subsystem extension constantly fails compilation against the host's current NVCC / PyTorch environment libraries when using raw CMake arguments.
**Action:** Move cleanly into the source directory and forcefully execute the shell instructions.
```bash
cd re_ranking/gnn_ext
chmod +x make.sh
./make.sh
```
*Debug Focus:* If the final output adjacency matrix propagates incorrectly or errors out mathematically inside PyTorch logic, confirm contiguous formatting via `feature_tensor = feature_tensor.contiguous()` just before pushing pointers into C++ bindings!

### 2. Verify 4-Channel First Convolution Mapping Check
**Problem:** Resolving severe dimensionality fault exceptions during RGBD weights initialization (`model_rgbd.py`).
**Action:** Audit how `convert_conv1_to_4channel()` grabs pretrained ImageNet data, manipulates shapes, and distributes mean layers. Watch specific PyTorch states carefully; ensure `requires_grad=True` was not silently stripped during your deep `.data` copying methodologies.

### 3. Launch Massive Run Scripts: RGBD
**Action:** Theory ends here. Push the architecture immediately.
```bash
# Push process to background nohup and stream data out
nohup sh train_rgbd.sh > logs/rgbd_training_$(date +%s).log 2>&1 &
tail -f logs/rgbd_training_*.log
```
*Shell Configuration Blueprint (`train_rgbd.sh` excerpt):*
```bash
python train.py \
    --name bleeding_edge_rgbd \
    --use_rgbd \
    --pool lpn --lpn_blocks 4 \
    --batchsize 8 \
    --lr 0.01 \
    --circle \
    --fp16
```

### 4. Bulletproof Dataloader Previews
**Action:** Manually write dry-run execution checks targeting `RGBDSatelliteDataset`. It is an absolute waste of GPU costs to have an automated pipeline loop for 14 hours gracefully only to error on epoch 3 because `05321_depth.png` fails I/O mapping checks.

```python
# quick_io_verify.py
from dataset_rgbd import RGBDSatelliteDataset

print("BOOTING SANITY CHECK...")
dataset = RGBDSatelliteDataset(rgb_folder='train/satellite', depth_folder='train/satellite_depth')
print(f"Missing depth maps total: {dataset.missing_depth_count}")

assert dataset.missing_depth_count == 0, "FATAL: Rectify depth repository outputs immediately before assigning cluster allocations!"
print("System stable. Go launch it.")
```

**NO EXCUSES. DEBUG IT. COMPILE IT. SHIP IT.**
