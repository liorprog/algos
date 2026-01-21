"""
algos/mesh.py

MeshND: N-dimensional mesh/grid implemented in pure Python (no numpy).

Added methods:
- normalize_counts(...) : zero small values, divide or cap large values
- get_peaks(...) : non-maximum suppression local maxima extraction

Pure-Python, easy to step through for debugging on a phone.
"""
from typing import Sequence, Tuple, List, Optional, Dict, Iterable
import math
import itertools


class MeshND:
    def __init__(self, bounds: Sequence[Tuple[float, float]], cells: Sequence[int]):
        if len(bounds) != len(cells):
            raise ValueError("bounds and cells must have the same length (ndim).")
        self.ndim = len(bounds)
        # Validate and store bounds and cells
        self.bounds: List[Tuple[float, float]] = [(float(a), float(b)) for (a, b) in bounds]
        self.cells: List[int] = [int(c) for c in cells]
        for (mn, mx) in self.bounds:
            if mx <= mn:
                raise ValueError("Each bound must have max > min.")
        for c in self.cells:
            if c <= 0:
                raise ValueError("Each cell count must be positive integer.")

        # Compute cell sizes and centers per axis
        self.cell_sizes: List[float] = []
        self.centers: List[List[float]] = []
        for d in range(self.ndim):
            mn, mx = self.bounds[d]
            n = self.cells[d]
            size = (mx - mn) / n
            self.cell_sizes.append(size)
            half = 0.5 * size
            centers_d = [mn + half + i * size for i in range(n)]
            self.centers.append(centers_d)

        # Create nested counts structure
        self.counts = self._make_nested_list(self.cells, 0.0)

    # -----------------------
    # Nested-list utilities
    # -----------------------
    def _make_nested_list(self, sizes: Sequence[int], fill):
        """Recursively create nested lists of shape sizes filled with `fill`."""
        if len(sizes) == 1:
            return [fill for _ in range(sizes[0])]
        return [self._make_nested_list(sizes[1:], fill) for _ in range(sizes[0])]

    def _get_at(self, idx: Tuple[int, ...]):
        """Return value at nested index tuple."""
        ref = self.counts
        for k in idx:
            ref = ref[k]
        return ref

    def _set_at(self, idx: Tuple[int, ...], value: float):
        """Set value at nested index tuple."""
        ref = self.counts
        for k in idx[:-1]:
            ref = ref[k]
        ref[idx[-1]] = value

    def _add_at(self, idx: Tuple[int, ...], delta: float):
        """Add delta to the nested element at idx."""
        ref = self.counts
        for k in idx[:-1]:
            ref = ref[k]
        ref[idx[-1]] += delta

    def _all_indices(self) -> Iterable[Tuple[int, ...]]:
        """Yield all valid index tuples in the grid (product of ranges)."""
        ranges = [range(n) for n in self.cells]
        return itertools.product(*ranges)

    # -----------------------
    # Public API
    # -----------------------
    def reset(self) -> None:
        """Reset all amplitudes to zero."""
        self.counts = self._make_nested_list(self.cells, 0.0)

    def get_counts(self):
        """Return the nested list of counts (view)."""
        return self.counts

    def get_centers(self) -> List[List[float]]:
        """Return per-axis centers lists."""
        return self.centers

    def _validate_point(self, point: Sequence[float]) -> List[float]:
        p = [float(x) for x in point]
        if len(p) != self.ndim:
            raise ValueError(f"Point must have dimension {self.ndim}.")
        for d in range(self.ndim):
            mn, mx = self.bounds[d]
            if p[d] < mn or p[d] > mx:
                raise ValueError(f"Point {p} outside bounds: axis {d} in [{mn}, {mx}].")
        return p

    def _nearest_index(self, p: Sequence[float]) -> Tuple[int, ...]:
        """Return nearest grid index tuple to point p by scanning centers per-axis."""
        idxs = []
        for d in range(self.ndim):
            centers_d = self.centers[d]
            best = 0
            best_dist = abs(centers_d[0] - p[d])
            for i in range(1, len(centers_d)):
                dist = abs(centers_d[i] - p[d])
                if dist < best_dist:
                    best_dist = dist
                    best = i
            idxs.append(best)
        return tuple(idxs)

    def _axis_index_range_within_radius(self, axis: int, coord: float, radius: float) -> Tuple[int, int]:
        """
        Return inclusive range [min_i, max_i] of indices on axis whose centers are within radius.
        If none found return (nearest, nearest).
        """
        centers = self.centers[axis]
        min_i = None
        max_i = None
        for i, c in enumerate(centers):
            if abs(c - coord) <= radius:
                if min_i is None:
                    min_i = i
                max_i = i
        if min_i is None:
            # fallback to nearest single index
            nearest = 0
            best = abs(centers[0] - coord)
            for i in range(1, len(centers)):
                d = abs(centers[i] - coord)
                if d < best:
                    best = d
                    nearest = i
            return nearest, nearest
        return min_i, max_i

    def normalize_weights(self, weights: Dict[Tuple[int, ...], float]) -> Dict[Tuple[int, ...], float]:
        """Normalize a weights dict so their sum equals 1. Returns new dict."""
        total = sum(weights.values())
        if total == 0:
            return weights.copy()
        return {idx: w / total for idx, w in weights.items()}

    def add(
        self,
        point: Sequence[float],
        amount: float = 1.0,
        radius: float = 0.0,
        mode: str = "gaussian",
        sigma: Optional[float] = None,
        normalize: bool = False,
    ) -> None:
        """
        Add `amount` distributed to the nearest cell and neighbors within `radius`.

        Args:
            point: coordinate sequence of length ndim.
            amount: base increment value (float).
            radius: influence radius. If <= 0 only the nearest cell is incremented.
            mode: 'gaussian', 'flat', or 'inverse'.
            sigma: gaussian sigma (if None, defaults to radius/3).
            normalize: if True, scale weights so total applied == amount.
                       If False, each cell receives amount * weight (un-normalized weights).
        """
        p = self._validate_point(point)
        amount = float(amount)

        if radius is None or radius <= 0.0:
            idx = self._nearest_index(p)
            self._add_at(idx, amount)
            return

        # Determine index ranges on each axis
        ranges = []
        for d in range(self.ndim):
            mn_i, mx_i = self._axis_index_range_within_radius(d, p[d], radius)
            ranges.append(range(mn_i, mx_i + 1))

        # Prepare sigma for gaussian
        if mode == "gaussian":
            if sigma is None:
                sigma = max(1e-12, radius / 3.0)

        # Collect weights for all candidate cells inside radius
        weights: Dict[Tuple[int, ...], float] = {}
        for idx in itertools.product(*ranges):
            # compute Euclidean distance from point p to cell center at idx
            dist2 = 0.0
            for d, idd in enumerate(idx):
                c = self.centers[d][idd]
                diff = c - p[d]
                dist2 += diff * diff
            dist = math.sqrt(dist2)
            if dist > radius:
                continue  # outside influence
            if mode == "gaussian":
                w = math.exp(-0.5 * (dist / sigma) ** 2) if sigma > 0.0 else 0.0
            elif mode == "flat":
                w = 1.0
            elif mode == "inverse":
                w = 1.0 / (1.0 + dist)
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'gaussian', 'flat', or 'inverse'.")
            if w != 0.0:
                weights[idx] = weights.get(idx, 0.0) + w

        # If nothing found (radius too small), fallback to nearest cell
        if not weights:
            idx = self._nearest_index(p)
            self._add_at(idx, amount)
            return

        # Normalize weights if requested
        if normalize:
            normed = self.normalize_weights(weights)
            for idx, w in normed.items():
                self._add_at(idx, amount * w)
        else:
            for idx, w in weights.items():
                self._add_at(idx, amount * w)

    # -----------------------
    # Post-processing helpers
    # -----------------------
    def normalize_counts(
        self,
        low_threshold: float = 0.0,
        zero_small: bool = True,
        high_threshold: Optional[float] = None,
        divide_const: Optional[float] = None,
        cap_only: bool = False,
    ) -> None:
        """
        Post-process counts.

        Behavior:
          - If zero_small=True, values strictly < low_threshold are set to 0.
          - If high_threshold is set, cells with value > high_threshold are either:
                * divided by divide_const (if divide_const provided), or
                * capped to high_threshold (if cap_only=True), or
                * left unchanged (if neither provided).
        Parameters:
          low_threshold: values below this are considered noise.
          zero_small: if True, zero those small values.
          high_threshold: above this is considered "too large" (e.g., outlier amplification).
          divide_const: if provided and high_threshold is set, values > high_threshold are divided by this const.
          cap_only: if True and divide_const is None, values > high_threshold are capped to high_threshold.
        """
        if low_threshold is None:
            low_threshold = 0.0
        for idx in self._all_indices():
            v = self._get_at(idx)
            if zero_small and v < low_threshold:
                if v != 0.0:
                    self._set_at(idx, 0.0)
                continue
            if high_threshold is not None and v > high_threshold:
                if divide_const is not None and divide_const != 0.0:
                    newv = v / float(divide_const)
                    self._set_at(idx, newv)
                elif cap_only:
                    self._set_at(idx, float(high_threshold))
                # else leave value unchanged (user chose no-op behavior)

    def get_peaks(self, min_value: float = 0.0, neighborhood: int = 1) -> List[Tuple[Tuple[int, ...], float]]:
        """
        Return list of (index_tuple, value) that are local maxima within the given neighborhood radius (in cells)
        and >= min_value.

        neighborhood: integer radius in cells to consider as neighbors (1 means the immediate hypercube).
        """
        peaks: List[Tuple[Tuple[int, ...], float]] = []
        # precompute axis neighbor ranges per index inside loop
        for idx in self._all_indices():
            v = self._get_at(idx)
            if v < min_value:
                continue
            # build neighbor ranges
            neigh_ranges = []
            is_local_max = True
            for d, idd in enumerate(idx):
                start = max(0, idd - neighborhood)
                end = min(self.cells[d] - 1, idd + neighborhood)
                neigh_ranges.append(range(start, end + 1))
            # check neighbors
            for nidx in itertools.product(*neigh_ranges):
                if nidx == idx:
                    continue
                nv = self._get_at(nidx)
                if nv > v:
                    is_local_max = False
                    break
            if is_local_max:
                peaks.append((idx, v))
        return peaks

    # Utility repr
    def __repr__(self):
        return f"MeshND(ndim={self.ndim}, bounds={self.bounds}, cells={self.cells})"
