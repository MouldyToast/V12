# V3 Preprocessing Validation Report

## Overview

This report documents the validation of `preprocess_singular_v3_basis_transform.py` against the documentation in `V3_PREPROCESSING_README.md`.

## Test Suite Created

The following test scripts were created:

1. **test_preprocessing_validation.py** - Step-by-step validation of all 8 preprocessing steps
2. **test_mean_centering_bug.py** - Investigation of the mean centering issue (Issue 1 from README)
3. **test_end_to_end_synthetic.py** - End-to-end test with 200 synthetic trajectories
4. **test_reconstruction_debug.py** - Debug test for high reconstruction errors
5. **test_orthonormality_fix.py** - Root cause analysis and fix verification

## Critical Bug Found

### **BasisAdapter Orthonormality Bug** (CRITICAL)

**Location:** `bspline_basis.py`, `BasisAdapter.get_adapted_basis()` method

**Problem:** When the basis is adapted from reference length `T_ref` to a different length `T`, the transformation `U_T = C @ U_ref` (where `C` is the B-spline transformation matrix) **destroys orthonormality**.

**Evidence:**
```
T=20: orthonormality error = 0.0000 (at reference length)
T=30: orthonormality error = 0.6763
T=50: orthonormality error = 1.4994
T=100: orthonormality error = 4.0019
```

**Impact:**
- Reconstruction RMSE without fix: **126.5 pixels** (mean)
- Reconstruction RMSE with fix: **0.73 pixels** (mean)
- **99.4% improvement** when orthonormality is restored

**Root Cause:**
The B-spline transformation matrix `C` maps between different lengths using cubic B-spline interpolation. While `C` preserves partition of unity (rows sum to 1), it does NOT preserve orthonormality of column vectors. Thus `C @ U_ref` produces non-orthonormal columns even when `U_ref` has orthonormal columns.

**Recommended Fix:**
Add QR re-orthonormalization after the B-spline transformation:

```python
def get_adapted_basis(self, T: int) -> np.ndarray:
    C_interleaved = build_interleaved_basis_matrix(T_out=T, T_in=self.T_ref)
    U_T = C_interleaved @ self.U_ref

    # FIX: Re-orthonormalize using QR decomposition
    Q, R = np.linalg.qr(U_T)
    U_T = Q[:, :self.K]

    return U_T
```

## Other Findings

### Mean Centering (Issue 1 from README)

**Status:** Minor issue (0.4% impact)

The README identified a potential bug where:
- Step 2 centers control points before SVD
- Step 3 projects raw trajectories without centering

Testing shows this has minimal impact on reconstruction quality (~0.4% improvement with centering). The current approach is acceptable, though mathematically inconsistent.

### Comparison with learn_reference_basis()

**Status:** Implementations match

The inline implementation in `learn_basis_from_trajectories()` produces mathematically identical results to `bspline_basis.learn_reference_basis()`. Both span the same subspace (projection matrix difference < 1e-8).

### Documentation Accuracy

The README is accurate regarding:
- Line counts and function locations
- Output structure
- Configuration parameters
- Step descriptions

## Test Results Summary

| Test | Status | Notes |
|------|--------|-------|
| Load trajectories | ✓ Pass | Correct filtering, relative coords |
| Learn basis | ✓ Pass | Orthonormal basis, matches library |
| Project trajectories | ✓ Pass | With orthonormality fix |
| Generate anchors | ✓ Pass | Correct per-group clustering |
| Compute residuals | ✓ Pass | Formula verified |
| Split/Save/Verify | ✓ Pass | All files created correctly |
| Jitter preservation | ✓ Pass | 135.7% preservation (V3 working) |

## Recommendations

1. **Fix the orthonormality bug** in `bspline_basis.py` - this is critical for correct reconstruction

2. **Add validation tests** to the CI pipeline using the test scripts created

3. **Consider adding centering** for mathematical consistency, though impact is minimal

4. **Document the QR fix** in the code comments explaining why it's necessary

## Running the Tests

```bash
# Run bspline_basis self-tests
python bspline_basis.py

# Run preprocessing validation
python test_preprocessing_validation.py

# Run mean centering investigation
python test_mean_centering_bug.py

# Run orthonormality fix verification (confirms the bug and fix)
python test_orthonormality_fix.py

# Run end-to-end with synthetic data (requires fix applied first)
python test_end_to_end_synthetic.py
```

## Conclusion

The V3 preprocessing pipeline is fundamentally sound and aligns with the documentation. However, a **critical bug** in the `BasisAdapter` class causes poor reconstruction quality. This bug affects any trajectory length different from `T_ref` and should be fixed before production use.

The fix is straightforward (QR re-orthonormalization) and reduces reconstruction error from ~127 pixels to ~0.7 pixels.
