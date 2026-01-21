"""
examples/example_combined.py

Combined demo:
- Choose some random "true" points (plotted in red).
- For several cycles, generate many noisy vectors around those true points (large error).
- Accumulate vectors into MeshND.
- After each cycle update the GUI: heatmap (blue) of grid amplitudes and red true points.

Run:
    pip install PyQt5
    python3 examples/example_combined.py

Notes:
- The GUI updates every cycle using QTimer so you can watch accumulation happen.
- Tweak parameters: NUM_TRUE, SAMPLES_PER_TRUE_PER_CYCLE, CYCLES, ERROR_STD to see behavior.
"""
import random
import math
from functools import partial

from algos.mesh import MeshND
from algos.mesh_visualizer import MeshWindow, run_gui

from PyQt5 import QtWidgets, QtCore


def generate_noisy_samples(true_points, samples_per_point, error_std, bounds):
    """Yield noisy samples around each true point, clipped to bounds."""
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    for p in true_points:
        tx, ty = p
        for _ in range(samples_per_point):
            nx = random.gauss(tx, error_std)
            ny = random.gauss(ty, error_std)
            # clip to bounds
            nx = max(xmin, min(xmax, nx))
            ny = max(ymin, min(ymax, ny))
            yield (nx, ny)


class Controller(QtCore.QObject):
    """
    Runs cycles of noisy sample generation, accumulates into the mesh, and updates the GUI.
    """

    def __init__(self, mesh, true_points, window, *, cycles=10, samples_per_point=200, error_std=0.12, radius=0.08):
        super().__init__()
        self.mesh = mesh
        self.true_points = true_points
        self.window = window
        self.cycles = cycles
        self.samples_per_point = samples_per_point
        self.error_std = error_std
        self.radius = radius
        self.current_cycle = 0
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(700)  # ms between cycles
        self.timer.timeout.connect(self.step)

    def start(self):
        self.timer.start()

    def step(self):
        if self.current_cycle >= self.cycles:
            self.timer.stop()
            print("Done cycles.")
            # Optionally compute peaks and print them
            peaks = self.mesh.get_peaks(min_value=0.5, neighborhood=1)
            print("Extracted peaks (index,value):", peaks)
            # Map peaks to coordinates
            centers = self.mesh.get_centers()
            mapped = [([centers[d][idx[d]] for d in range(len(idx))], v) for idx, v in peaks]
            print("Mapped peaks to coords:", mapped)
            return

        # Generate noisy samples and add to mesh
        samples = list(generate_noisy_samples(self.true_points, self.samples_per_point, self.error_std, self.mesh.bounds))
        for s in samples:
            # each noisy sample contributes amount=1.0; influence radius controls spread to neighbours
            self.mesh.add(s, amount=1.0, radius=self.radius, mode="gaussian", normalize=False)

        # Update GUI: set mesh and true points; use blue heatmap
        self.window.visualizer.set_mesh(self.mesh)
        self.window.visualizer.set_true_points(self.true_points)
        # Optionally also compute and display peaks (not strictly necessary each cycle)
        peaks = self.mesh.get_peaks(min_value=0.5, neighborhood=1)
        self.window.visualizer.set_peaks(peaks)

        self.window.visualizer._max_value = None  # recompute color scaling
        self.window.visualizer.update()

        self.current_cycle += 1
        print(f"Cycle {self.current_cycle}/{self.cycles}: added {len(samples)} samples.")


def main():
    random.seed(1234)

    # Parameters (tweakable)
    NUM_TRUE = 3
    CYCLES = 12
    SAMPLES_PER_TRUE_PER_CYCLE = 200
    ERROR_STD = 0.12  # make error large relative to grid cell distances
    GRID_CELLS = [50, 50]

    # Choose ground-truth points uniformly in (0,1)x(0,1)
    true_points = [(random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)) for _ in range(NUM_TRUE)]
    print("True points:", true_points)

    mesh = MeshND(bounds=[(0.0, 1.0), (0.0, 1.0)], cells=GRID_CELLS)

    # Create window with blue heatmap so grid amplitudes appear blue
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    window = MeshWindow(mesh, peaks=None, true_points=true_points, show_values=False, blue_heatmap=True)
    window.show()

    controller = Controller(
        mesh,
        true_points,
        window,
        cycles=CYCLES,
        samples_per_point=SAMPLES_PER_TRUE_PER_CYCLE,
        error_std=ERROR_STD,
        radius=0.08,
    )
    controller.start()

    app.exec_()


if __name__ == "__main__":
    main()
