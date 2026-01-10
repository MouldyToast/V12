"""
bspline.py - B-Spline Interpolation for Variable-Length Trajectories

Simple, standalone module. No package structure needed.

Usage:
    from bspline import BSplineInterpolator, build_bspline_matrix
    
    # Interpolate trajectory to different length
    interp = BSplineInterpolator()
    x_out, y_out = interp.interpolate(x_in, y_in, T_out=100)
    
    # Or use matrix form
    B = build_bspline_matrix(T_out=100, T_in=50)
    x_out = B @ x_in
"""

import numpy as np
from typing import Tuple


def cubic_bspline(t: float) -> float:
    """
    Evaluate cubic B-spline basis function at t.
    
    Bâ‚ƒ(t) = (2/3) - |t|Â² + |t|Â³/2      for |t| â‰¤ 1
    Bâ‚ƒ(t) = (2-|t|)Â³/6                  for 1 < |t| â‰¤ 2
    Bâ‚ƒ(t) = 0                           otherwise
    """
    t = abs(t)
    
    if t >= 2.0:
        return 0.0
    elif t >= 1.0:
        u = 2.0 - t
        return (u * u * u) / 6.0
    else:
        return (2.0/3.0) - t*t + (t*t*t) / 2.0


def build_bspline_matrix(T_out: int, T_in: int) -> np.ndarray:
    """
    Build B-spline interpolation matrix with endpoint clamping.
    
    Returns B such that: y_out = B @ y_in
    
    The first output point exactly equals the first input point,
    and the last output point exactly equals the last input point.
    
    Args:
        T_out: Number of output points
        T_in: Number of input points
        
    Returns:
        B: Matrix of shape (T_out, T_in)
    """
    if T_out < 1 or T_in < 1:
        raise ValueError(f"T_out and T_in must be >= 1")
    
    if T_in == 1:
        return np.ones((T_out, 1))
    
    if T_in == 2:
        # Linear interpolation for 2 points
        t = np.linspace(0, 1, T_out)
        B = np.zeros((T_out, 2))
        B[:, 0] = 1 - t
        B[:, 1] = t
        return B
    
    # Parameter values: output points mapped to input parameter space
    t_out = np.linspace(0, T_in - 1, T_out)
    
    # Build basis matrix
    B = np.zeros((T_out, T_in))
    
    for i in range(T_out):
        for j in range(T_in):
            B[i, j] = cubic_bspline(t_out[i] - j)
    
    # Normalize rows to sum to 1
    row_sums = B.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-10, row_sums, 1.0)
    B = B / row_sums
    
    # Clamp endpoints: first output = first input, last output = last input
    B[0, :] = 0
    B[0, 0] = 1
    B[-1, :] = 0
    B[-1, -1] = 1
    
    return B


class BSplineInterpolator:
    """Simple B-spline interpolator for trajectories."""
    
    def __init__(self):
        self._cache = {}
    
    def get_matrix(self, T_out: int, T_in: int) -> np.ndarray:
        """Get interpolation matrix, using cache."""
        key = (T_out, T_in)
        if key not in self._cache:
            self._cache[key] = build_bspline_matrix(T_out, T_in)
        return self._cache[key]
    
    def interpolate(self, x: np.ndarray, y: np.ndarray, T_out: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate 2D trajectory to different length.
        
        Args:
            x: Input x-coordinates, shape (T_in,)
            y: Input y-coordinates, shape (T_in,)
            T_out: Desired output length
            
        Returns:
            x_out, y_out: Interpolated coordinates, each shape (T_out,)
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length")
        
        T_in = len(x)
        B = self.get_matrix(T_out, T_in)
        
        return B @ x, B @ y
    
    def interpolate_flat(self, traj_flat: np.ndarray, T_in: int, T_out: int) -> np.ndarray:
        """
        Interpolate flattened trajectory [xâ‚,yâ‚,xâ‚‚,yâ‚‚,...].
        
        Args:
            traj_flat: Flattened trajectory, shape (2*T_in,)
            T_in: Number of input points
            T_out: Desired output points
            
        Returns:
            Interpolated flat trajectory, shape (2*T_out,)
        """
        traj_flat = np.asarray(traj_flat).flatten()
        
        # Extract x and y (interleaved format)
        x_in = traj_flat[0::2]
        y_in = traj_flat[1::2]
        
        # Interpolate
        x_out, y_out = self.interpolate(x_in, y_in, T_out)
        
        # Recombine
        result = np.empty(2 * T_out)
        result[0::2] = x_out
        result[1::2] = y_out
        
        return result


# =============================================================================
# Quick self-test when run directly
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("B-SPLINE INTERPOLATION TESTS")
    print("=" * 60)
    
    interp = BSplineInterpolator()
    
    # Test 1: Straight line should stay straight
    print("\n[Test 1] Straight line interpolation...")
    x_in = np.linspace(0, 100, 50)
    y_in = np.linspace(0, 50, 50)
    x_out, y_out = interp.interpolate(x_in, y_in, T_out=100)
    
    expected_x = np.linspace(0, 100, 100)
    expected_y = np.linspace(0, 50, 100)
    
    x_err = np.max(np.abs(x_out - expected_x))
    y_err = np.max(np.abs(y_out - expected_y))
    print(f"  Max error: x={x_err:.6f}, y={y_err:.6f}")
    assert x_err < 0.5 and y_err < 0.5, "FAILED: Straight line not preserved"
    print("  PASSED")
    
    # Test 2: Endpoints preserved
    print("\n[Test 2] Endpoint preservation...")
    x_in = np.array([0, 10, 25, 50, 100])
    y_in = np.array([0, 20, 10, 30, 0])
    
    for T_out in [10, 50, 100, 200]:
        x_out, y_out = interp.interpolate(x_in, y_in, T_out)
        start_err = np.sqrt((x_out[0] - x_in[0])**2 + (y_out[0] - y_in[0])**2)
        end_err = np.sqrt((x_out[-1] - x_in[-1])**2 + (y_out[-1] - y_in[-1])**2)
        print(f"  T_out={T_out}: start_err={start_err:.4f}, end_err={end_err:.4f}")
        assert start_err < 1.0 and end_err < 1.0, f"FAILED at T_out={T_out}"
    print("  PASSED")
    
    # Test 3: Round-trip (T_orig -> T_win -> T_orig)
    print("\n[Test 3] Round-trip consistency...")
    np.random.seed(42)
    
    for T_orig, T_win in [(30, 100), (150, 100), (50, 75), (100, 100)]:
        # Create random smooth trajectory
        t = np.linspace(0, 2*np.pi, T_orig)
        x_orig = t * 20 + np.cumsum(np.random.randn(T_orig)) * 2
        y_orig = np.sin(t) * 30 + np.cumsum(np.random.randn(T_orig)) * 2
        
        # Forward: T_orig -> T_win
        x_win, y_win = interp.interpolate(x_orig, y_orig, T_win)
        
        # Backward: T_win -> T_orig
        x_back, y_back = interp.interpolate(x_win, y_win, T_orig)
        
        # Compare
        x_corr = np.corrcoef(x_orig, x_back)[0, 1]
        y_corr = np.corrcoef(y_orig, y_back)[0, 1]
        
        print(f"  {T_orig} -> {T_win} -> {T_orig}: x_corr={x_corr:.4f}, y_corr={y_corr:.4f}")
        assert x_corr > 0.95 and y_corr > 0.95, f"FAILED: correlation too low"
    print("  PASSED")
    
    # Test 4: Partition of unity (rows sum to 1)
    print("\n[Test 4] Partition of unity...")
    for T_in, T_out in [(50, 100), (100, 50), (30, 70)]:
        B = build_bspline_matrix(T_out, T_in)
        row_sums = B.sum(axis=1)
        max_err = np.max(np.abs(row_sums - 1.0))
        print(f"  ({T_in} -> {T_out}): max row sum error = {max_err:.10f}")
        assert max_err < 1e-10, "FAILED: rows don't sum to 1"
    print("  PASSED")
    
    # Test 5: Flat trajectory format
    print("\n[Test 5] Flat trajectory interpolation...")
    x_in = np.array([0, 50, 100])
    y_in = np.array([0, 25, 0])
    flat_in = np.empty(6)
    flat_in[0::2] = x_in
    flat_in[1::2] = y_in
    
    flat_out = interp.interpolate_flat(flat_in, T_in=3, T_out=5)
    assert len(flat_out) == 10, "FAILED: wrong output length"
    assert abs(flat_out[0] - 0) < 0.1, "FAILED: start x"
    assert abs(flat_out[-2] - 100) < 0.1, "FAILED: end x"
    print(f"  Input:  {flat_in}")
    print(f"  Output: {flat_out}")
    print("  PASSED")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
