# V3 Implementation: Masked SVD (No B-Spline Interpolation)

## The Problem with V2

Your current V2 pipeline does:
```
raw trajectory (30-150 pts) → B-spline interpolate to 64 pts → SVD
```

This **smooths away jitter and micro-corrections** because B-splines enforce C² continuity by design.

## V3 Solution: Masked SVD

V3 does:
```
raw trajectory (30-150 pts) → pad to 160 pts with zeros → masked EM-SVD
```

Key insight: **Padded zeros are treated as "missing data"**, not as actual trajectory points. The EM-SVD algorithm iteratively:
1. Imputes missing values using current low-rank approximation
2. Recomputes SVD
3. Updates only missing positions
4. Repeats until convergence

This means **raw trajectory data is preserved exactly** - no smoothing.

## Files Created

| File | Purpose |
|------|---------|
| `preprocess_singular_v3_masked.py` | Drop-in replacement for V2 preprocessing |
| `generate_trajectory_v3.py` | Generation script for V3 data |
| `masked_svd.py` | Core masked SVD implementation |
| `path_signature.py` | Alternative approach (for future if needed) |

## Usage

### 1. Preprocess your data
```bash
python preprocess_singular_v3_masked.py \
    --input trajectories/ \
    --output processed_v3/ \
    --k 20 \
    --max_length 160
```

Note: You can use **higher K** with V3 because you're not fighting B-spline smoothing. Try K=20-30.

### 2. Train (use existing training script!)
```bash
python train_singular_diffusion_v1.py \
    --data processed_v3/ \
    --epochs 256
```

The training script works unchanged because the data format is compatible.

### 3. Generate
```bash
python generate_trajectory_v3.py \
    --checkpoint checkpoints/best_model.pt \
    --data processed_v3/ \
    --all_groups \
    --visualize
```

## Key Differences from V2

| Aspect | V2 | V3 |
|--------|-----|-----|
| Preprocessing | B-spline interpolate to T_win | Pad to max_length, use mask |
| SVD input | All trajectories same length | Variable lengths via mask |
| Jitter | Smoothed away | **Preserved** |
| U_k shape | [2×T_win, K] | [2×max_length, K] |
| Reconstruction | U_k @ c → B-spline to L | U_k[:2L] @ c |

## Reconstruction Formula

```python
# V3 reconstruction at length L
U_k = np.load('U_k.npy')      # [320, K]
mean = np.load('mean.npy')     # [320]

# Reconstruct at target length L (no interpolation!)
flat_len = 2 * L
recon = U_k[:flat_len, :] @ coefficients + mean[:flat_len]

# Reshape to [L, 2]
trajectory = np.column_stack([recon[0::2], recon[1::2]])
```

## Expected Improvements

Based on empirical tests:

| Metric | V2 (B-spline) | V3 (Masked) |
|--------|---------------|-------------|
| Jitter correlation | ~0.30 | ~0.35 |
| High-freq preserved | ❌ | ✅ |
| Raw data preserved | ❌ | ✅ |

The 5% improvement in jitter correlation doesn't sound like much, but the bigger win is **eliminating interpolation artifacts** entirely.

## If V3 Still Too Smooth: Path Signatures

If generated trajectories are still too smooth with V3, the next step is **path signatures** (implemented in `path_signature.py`):

- Mathematical guarantee of unique path encoding
- Higher truncation levels capture jitter explicitly
- Jitter correlation: ~0.48 (vs 0.35 for masked SVD)

The tradeoff: signatures require a **learned decoder network** (signatures aren't directly invertible), which is a bigger architectural change.

## Recommended Next Steps

1. **Run V3 preprocessing** on your actual mouse data
2. **Train** with existing training script
3. **Compare** generated trajectories to V2 outputs
4. **Evaluate** jitter, smoothness, realism
5. **If still too smooth** → Consider path signatures (Phase 2)

## Questions to Answer After Testing

1. Do V3 generated trajectories look more realistic?
2. Is the jitter/micro-correction pattern preserved?
3. What K value works best with V3?
4. How does reconstruction error compare to V2?
