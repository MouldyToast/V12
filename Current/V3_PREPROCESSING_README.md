# V3 Preprocessing Script Analysis & Validation Guide

## Document Purpose
This README documents `preprocess_singular_v3_basis_transform.py` (1431 lines) so that another AI can help validate and test it. The script was built across multiple context compactions, which may have caused drift from the original plan.

---

## What This Script Is Supposed To Do

### Original Plan (from Phase 1 Summary)
The V3 preprocessing was supposed to:
1. Load raw trajectories (keeping original lengths) 
2. Learn reference basis using `learn_reference_basis()` from bspline_basis.py
3. Project all trajectories using `BasisAdapter.project()`
4. Generate group anchors (K-means on coefficients)
5. Compute residuals (coefficient - nearest_anchor)
6. Save: U_ref, T_ref, original_lengths, coefficients, residuals, anchors

**Key Innovation**: Transform the BASIS to match trajectory length, rather than transforming trajectory DATA to match a fixed basis. This preserves jitter and micro-corrections.

### The V2 vs V3 Difference
```
V2 (OLD):  trajectory → B-spline interpolate → canonical_length → SVD → coefficients
           Problem: B-spline smoothing DESTROYS jitter before SVD sees it
           
V3 (NEW):  trajectory → project onto length-adapted basis → coefficients
           Benefit: Raw trajectory preserved, jitter/micro-corrections intact
```

---

## Actual File Structure (1431 lines)

### Line Count Breakdown
| Section | Lines | Purpose |
|---------|-------|---------|
| Docstring & imports | 1-77 | Documentation, configuration |
| Configuration | 78-134 | DEFAULT_CONFIG, ORIENTATIONS, DISTANCE_GROUPS |
| Helper functions | 136-191 | get_orientation_id, get_distance_group_id, etc. |
| STEP 1: load_trajectories | 193-315 | Load JSON files, filter, convert to relative coords |
| STEP 2: learn_basis_from_trajectories | 317-456 | Fit control points, SVD |
| STEP 3: project_trajectories | 458-576 | Project using BasisAdapter |
| STEP 4: generate_group_anchors | 578-686 | K-means clustering per group |
| STEP 5: compute_residuals | 688-786 | coefficient - nearest_anchor |
| STEP 6: split_data | 788-889 | Train/val/test stratified splits |
| STEP 7: save_all_data | 891-1009 | Write all .npy files |
| STEP 8: verify_preprocessing | 1011-1180 | Validation tests |
| analyze_length_distributions | 1182-1249 | Per-group length stats |
| main() | 1251-1431 | CLI parsing, orchestration |

### Functions
1. `load_trajectories(data_dir, config)` → List[Dict]
2. `learn_basis_from_trajectories(trajectories, n_control_points, K, center_data)` → (U_ref, T_ref, mean, stats)
3. `project_trajectories(trajectories, U_ref, T_ref, mean)` → (coefficients, adapter, stats)
4. `generate_group_anchors(trajectories, coefficients, anchors_per_group, seed)` → (group_anchors, group_stats)
5. `compute_residuals(trajectories, coefficients, group_anchors)` → (residuals, anchor_indices, stats)
6. `split_data(trajectories, coefficients, residuals, anchor_indices, config)` → splits dict
7. `save_all_data(output_dir, U_ref, T_ref, mean, group_anchors, splits, config, stats)` → output_path
8. `verify_preprocessing(output_dir, trajectories, adapter, coefficients, n_samples)` → bool
9. `analyze_length_distributions(trajectories, group_anchors, output_dir)` → length_stats

---

## Dependencies

### Required Files
- `bspline_basis.py` - Must be in same directory or Python path

### Imported from bspline_basis.py
```python
from bspline_basis import (
    BasisAdapter,
    learn_reference_basis,
    fit_bspline_control_points,
    evaluate_control_point_fit,
)
```

### External Dependencies
- numpy
- sklearn (KMeans, train_test_split)
- scipy (optional, for jitter verification)

---

## Output Structure

```
processed_v3/
├── U_ref.npy              # [2*T_ref × K] reference basis
├── mean.npy               # [2*T_ref] mean control points (if centering)
├── config.npy             # Configuration dictionary
├── group_anchors.npy      # Dict: (dist_id, orient_id) → [n_anchors × K]
├── length_distributions.npy  # Per-group length statistics
└── train/val/test/
    ├── coefficients.npy   # [N × K]
    ├── residuals.npy      # [N × K] 
    ├── anchor_indices.npy # [N] int
    ├── orientation_ids.npy # [N] int (0-7)
    ├── distance_ids.npy   # [N] int (0-4)
    └── original_lengths.npy  # [N] int - CRITICAL for V3!
```

---

## ⚠️ POTENTIAL ISSUES FOR VALIDATION

### Issue 1: Mean Centering Mismatch
**Location**: Step 2 vs Step 3

**The Problem**:
- Step 2 (line 406-411) centers control points before SVD: `mean = control_points.mean(axis=0); centered = control_points - mean`
- Step 3 (line 527) projects RAW trajectories directly: `c = adapter.project(flat)` WITHOUT subtracting mean

**Expected Behavior** (from PHASE1_SUMMARY.md):
```python
# During preprocessing (after learning basis):
trajectory_centered = trajectory - mean  # If using centering
coefficients = adapter.project(trajectory_centered)
```

**Question**: Should trajectories be centered before projection? The mean is in control point space [2*T_ref], not trajectory space [2*T], so transformation is needed. The `BasisAdapter` has a `reconstruct_mean(mean, T)` method that transforms mean to trajectory length - should this be used?

**Validation Test**:
1. Project a trajectory WITH centering
2. Project same trajectory WITHOUT centering  
3. Compare coefficients - they should be different if centering matters

### Issue 2: Not Using learn_reference_basis()
**Location**: Step 2

**The Problem**: 
- The PHASE1_SUMMARY says to use `learn_reference_basis()` from bspline_basis.py
- Step 2 reimplements similar logic inline instead

**Questions**:
1. Is the reimplementation mathematically identical?
2. Why not use the provided function?

**Validation Test**:
- Call both functions on same data, compare outputs

### Issue 3: Projection Space Mismatch
**Location**: Step 2 and Step 3

**The Problem**:
- Step 2 works in **control point space**: fits trajectories to control points, does SVD on control points
- Step 3 works in **trajectory space**: projects raw trajectories directly

**Mathematical Question**:
If `U_ref` is learned from control point representations, but projection uses `U_T = C_2T @ U_ref` on raw trajectory data, is this mathematically correct?

The BasisAdapter does: `c = U_T.T @ trajectory` where `U_T = C_2T @ U_ref`

But U_ref columns are principal directions in control point space. Does applying C_2T properly transform them to work directly on trajectory space?

---

## Testing Recommendations

### Unit Tests for Each Step

**Test 1: Load Trajectories**
```python
# Verify correct filtering
# Verify length is preserved
# Verify orientation/distance classification
```

**Test 2: Basis Learning**
```python
# Verify U_ref shape is [2*T_ref × K]
# Verify orthonormality: U_ref.T @ U_ref ≈ I
# Compare to learn_reference_basis() output
```

**Test 3: Projection**
```python
# Round-trip: project then reconstruct at same length
# Should have low RMSE
# Test with/without mean centering
```

**Test 4: Anchors**
```python
# Verify anchor shapes
# Verify each group has correct number
```

**Test 5: Residuals**
```python
# residual = coefficient - anchor[nearest]
# norm(residual) < norm(coefficient) for most samples
```

### End-to-End Test
```bash
# Generate test trajectories
python preprocess_singular_v3_basis_transform.py --input test_data/ --output test_output/

# Verify outputs exist
ls test_output/*.npy
ls test_output/train/*.npy

# Load and check shapes
python -c "
import numpy as np
config = np.load('test_output/config.npy', allow_pickle=True).item()
U_ref = np.load('test_output/U_ref.npy')
train_coeffs = np.load('test_output/train/coefficients.npy')
train_lengths = np.load('test_output/train/original_lengths.npy')
print(f'U_ref: {U_ref.shape}')
print(f'coefficients: {train_coeffs.shape}')
print(f'lengths: {train_lengths.shape}')
print(f'K: {config[\"K\"]}, T_ref: {config[\"T_ref\"]}')
"
```

### Jitter Preservation Test (Critical for V3)
```python
# V3's main claim: preserves jitter better than V2
# Test: project → reconstruct, measure jitter before/after

from scipy.ndimage import gaussian_filter1d
import numpy as np

def jitter_metric(coords):
    """Jitter = RMS of acceleration (second derivative)"""
    accel = np.diff(np.diff(coords))
    return np.sqrt(np.mean(accel ** 2))

# Original trajectory
orig_jitter = jitter_metric(x_original)

# Reconstructed trajectory  
recon_jitter = jitter_metric(x_reconstructed)

# Preservation ratio should be > 80% for V3
preservation = recon_jitter / orig_jitter
assert preservation > 0.8, f"Jitter preservation too low: {preservation:.1%}"
```

---

## Configuration Reference

```python
DEFAULT_CONFIG = {
    'K': 8,                     # Singular space dimensions
    'n_control_points': 20,     # B-spline control points (= T_ref)
    'anchors_per_group': 6,     # K-means clusters per group
    'random_seed': 42,
    'min_length': 15,           # Minimum trajectory points
    'max_length_filter': 200,   # Maximum trajectory points
    'max_distance_ratio': 1.6,  # Filter erratic trajectories
    'min_distance': 20.0,       # Minimum ideal distance (pixels)
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'center_data': True,        # Whether to center before SVD
}
```

---

## CLI Usage

```bash
# Basic usage
python preprocess_singular_v3_basis_transform.py \
    --input trajectories/ \
    --output processed_v3/

# With custom parameters
python preprocess_singular_v3_basis_transform.py \
    --input trajectories/ \
    --output processed_v3/ \
    --n_control_points 16 \
    --k 8 \
    --anchors_per_group 6

# Skip verification
python preprocess_singular_v3_basis_transform.py \
    --input trajectories/ \
    --output processed_v3/ \
    --skip_verify
```

---

## Summary: Why 1431 Lines?

The file is large because it includes:
1. **Comprehensive documentation** (~150 lines of docstrings/comments)
2. **Detailed statistics reporting** (prints at every step)
3. **8 processing steps** with clear separation
4. **Verification step** with 4 different tests
5. **Length distribution analysis** (for generation)
6. **Full CLI interface** with many options
7. **Helper functions** for orientation/distance classification

The core logic (load → basis → project → anchors → residuals → save) is about 600 lines. The remaining ~800 lines are documentation, verification, statistics, and CLI.

---

## Detailed Mathematical Analysis

### The Centering Problem (Potential Bug)

**Training Phase (Step 2):**
1. Fit control points to each trajectory: `CP_i` for trajectory i
2. Compute mean: `μ = mean(CP_1, CP_2, ..., CP_N)` where μ is shape [2*T_ref]
3. Center: `CP_centered_i = CP_i - μ`
4. SVD: `stack(CP_centered) ≈ U_ref @ Σ @ V.T`

So U_ref columns are principal directions in **centered control point space**.

**Projection Phase (Step 3):**
Current code does:
```python
U_T = C_interleaved @ U_ref    # [2*T × K]
c = U_T.T @ trajectory         # Project raw trajectory
```

**What SHOULD happen (mathematically):**
```python
U_T = C_interleaved @ U_ref    # Adapt basis to length T
mean_T = C_interleaved @ μ     # Transform mean to trajectory space  
traj_centered = trajectory - mean_T  # Center the trajectory
c = U_T.T @ traj_centered      # Project centered data
```

**Why This Matters:**
- If U_ref captures directions around the mean (centered data), projecting uncentered data will give wrong coefficients
- The first coefficient will absorb the mean offset, corrupting the representation
- Reconstruction will be off by the mean

**The Counter-Argument:**
Maybe the math works out because:
- `c = U_T.T @ traj = U_T.T @ (traj_centered + mean_T) = U_T.T @ traj_centered + U_T.T @ mean_T`
- If U_T is orthonormal, and the mean projects to a constant offset in coefficient space, this might still work...

**Validation Required:**
```python
# Test if centering matters
adapter = BasisAdapter(U_ref, T_ref)
mean_T = adapter.reconstruct_mean(mean, T)  # Use existing method

# Method 1: Current code (no centering)
c1 = adapter.project(trajectory)

# Method 2: With centering
c2 = adapter.project(trajectory - mean_T)

# Compare
print(f"Difference: {np.linalg.norm(c1 - c2)}")
# If this is large relative to coefficient magnitude, there's a bug
```

---

## Questions for Human Review

1. **Mean centering**: Is the current implementation correct, or should trajectories be centered before projection? See mathematical analysis above.

2. **Basis learning**: Why doesn't Step 2 use `learn_reference_basis()` from bspline_basis.py? Is the reimplementation identical?

3. **Math verification**: Is `c = (C_2T @ U_ref).T @ trajectory` mathematically equivalent to projecting control points onto U_ref?

4. **Test with real data**: Has this been tested on actual mouse trajectory data? What were the results?

5. **Comparison to V2**: Is there a script to compare V2 vs V3 output quality on the same data?

6. **Reconstruction correctness**: When reconstructing, does the generate_trajectory_v3.py properly add back the mean? Check if it uses `adapter.reconstruct_mean()`.

**Note**: generate_trajectory_v3.py DOES add back the mean during reconstruction (lines 265-268, 348-350). This means:
- Preprocessing projects UNCENTERED trajectories
- Generation reconstructs and ADDS the mean

If preprocessing should have SUBTRACTED the mean before projection, then generation's ADDING would be correct. But if preprocessing doesn't subtract, then generation's adding may cause a mean mismatch.

---

## File Comparison: What Changed from Plan?

| Planned (PHASE1_SUMMARY) | Implemented | Status |
|--------------------------|-------------|--------|
| Use `learn_reference_basis()` | Reimplemented inline | ⚠️ Different |
| 6 steps total | 8 steps + extras | ➕ Expanded |
| Save U_ref, T_ref, lengths, coeffs, residuals, anchors | All of these + config, length_distributions | ✓ + More |
| No verification mentioned | Full verification step | ➕ Added |
| No length distribution analysis | analyze_length_distributions() | ➕ Added |

The implementation is MORE comprehensive than planned, which is good, but uses different basis learning (reimplemented vs using bspline_basis.py function).

---

## Quick Validation Checklist

- [ ] Run script on sample data: `python preprocess_singular_v3_basis_transform.py --input test_data/ --output test_out/`
- [ ] Check all output files exist
- [ ] Verify shapes: `U_ref` is [2*T_ref × K], `coefficients` is [N × K]
- [ ] Run verification step (not --skip_verify)
- [ ] Test round-trip accuracy on known trajectory
- [ ] Compare output to bspline_basis.py's `learn_reference_basis()` function
- [ ] Test mean centering impact (see validation test above)
- [ ] Compare V3 vs V2 reconstruction quality on same data
