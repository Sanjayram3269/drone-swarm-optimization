import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import numpy as np
import matplotlib.pyplot as plt

from behaviors.consensus_algorithm import ConsensusAlgorithm
from behaviors.collision_avoidance_algorithm import CollisionAvoidanceAlgorithm
from behaviors.formation_control_algorithm import FormationControlAlgorithm
from visualizer import DroneSwarmVisualizer
from drone import Drone
from metrics import MetricsTracker  # NEW

class DroneSwarmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Swarm Simulation")

        self.target_point = np.array([0, 0, 0])
        self.is_x_at_origin = True

        # Simulation parameters
        self.num_drones = 100
        self.epsilon = 0.1
        self.collision_threshold = 1.0
        self.interval = 200

        # Metrics
        self.metrics = MetricsTracker()  # NEW

        # UI variables
        self.formation_type = tk.StringVar(value="line")
        self.zoom_level = tk.DoubleVar(value=10.0)

        # Algorithms
        self.behavior_algorithms = [
            ConsensusAlgorithm(self.epsilon),
            CollisionAvoidanceAlgorithm(self.collision_threshold),
            FormationControlAlgorithm(self.formation_type.get())
        ]

        # Drones
        self.drones = [Drone(np.random.rand(3) * 10, i) for i in range(self.num_drones)]

        # Visualizer
        self.visualizer = DroneSwarmVisualizer(self.drones, self.formation_type.get())

        self.setup_ui()
        self.running = False

    def setup_ui(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(control_frame, text="Formation:").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Line", variable=self.formation_type, value="line", command=self.update_formation).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Circle", variable=self.formation_type, value="circle", command=self.update_formation).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Square", variable=self.formation_type, value="square", command=self.update_formation).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Random", variable=self.formation_type, value="random", command=self.update_formation).pack(anchor=tk.W)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Zoom Level:").pack(anchor=tk.W)
        zoom_scale = ttk.Scale(control_frame, from_=5.0, to=20.0, orient=tk.HORIZONTAL, variable=self.zoom_level, command=self.update_zoom)
        zoom_scale.pack(anchor=tk.W)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Color Mode:").pack(anchor=tk.W)
        self.color_mode = tk.StringVar(value="by_index")
        ttk.Radiobutton(control_frame, text="By Index", variable=self.color_mode, value="by_index", command=self.update_color_mode).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="By Distance", variable=self.color_mode, value="by_distance", command=self.update_color_mode).pack(anchor=tk.W)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Change X Position", command=self.change_x_position).pack(pady=10)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        self.start_button = ttk.Button(control_frame, text="Animate", command=self.toggle_simulation)
        self.start_button.pack(pady=10)

        # NEW: Plot button
        ttk.Button(control_frame, text="Show Graph", command=self.plot_graph).pack(pady=10)

        self.canvas = FigureCanvasTkAgg(self.visualizer.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def update_formation(self):
        self.behavior_algorithms[-1] = FormationControlAlgorithm(self.formation_type.get())
        self.visualizer.formation_type = self.formation_type.get()
        self.canvas.draw()

    def update_zoom(self, event):
        self.visualizer.update_zoom(self.zoom_level.get())
        self.canvas.draw()

    def update_color_mode(self):
        self.visualizer.color_mode = self.color_mode.get()
        self.visualizer.update_colors()
        self.canvas.draw()

    def toggle_simulation(self):
        if self.running:
            self.running = False
            self.start_button.config(text="Animate")
        else:
            self.running = True
            self.start_button.config(text="Stop")
            threading.Thread(target=self.run_simulation, daemon=True).start()

    def change_x_position(self):
        if self.is_x_at_origin:
            self.target_point = np.array([20, 0, 0])
        else:
            self.target_point = np.array([0, 0, 0])

        self.is_x_at_origin = not self.is_x_at_origin
        self.update_target_positions()

    def update_target_positions(self):
        formation = self.behavior_algorithms[-1].get_formation(self.drones)
        for drone, target in zip(self.drones, formation):
            drone.target_position = self.target_point + target

        self.behavior_algorithms[-1].set_target_point(self.target_point)

    def run_simulation(self):
        while self.running:
            for drone in self.drones:
                neighbor_positions = [
                    other_drone.communicate()
                    for other_drone in self.drones
                    if other_drone != drone
                ]
                drone.update_position(neighbor_positions, self.behavior_algorithms)

            # NEW: Track error
            self.metrics.compute_total_error(self.drones)

            self.visualizer.update_view(self.drones)
            self.visualizer.update()
            self.canvas.draw()

    # NEW: Graph plotting
    def plot_graph(self):
        errors = self.metrics.get_errors()

        plt.figure()
        plt.plot(errors)
        plt.title("Formation Error Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Error")
        plt.grid()
        plt.show()


def main():
    root = tk.Tk()
    app = DroneSwarmApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()