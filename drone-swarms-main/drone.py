import numpy as np
from pso_optimizer import PSOOptimizer

# -------------------------------
# GLOBAL SETTINGS (FOR EXPERIMENTS)
# -------------------------------
WEATHER_MODE = "strong"   # options: "none", "medium", "strong"
USE_PSO = True        # True = PSO ON, False = PSO OFF


class Drone:
    """
    Represents a single drone in the swarm with movement,
    disturbance handling, and optional PSO-based optimization.
    """

    def __init__(self, position, index):
        self.position = np.array(position, dtype=float)
        self.index = index
        self.target_position = np.array(position, dtype=float)

        # Initialize PSO optimizer
        self.pso = PSOOptimizer()

    def compute_formation_error(self, desired_position):
        """
        Calculates how far the drone is from its desired formation position.
        """
        return np.linalg.norm(self.position - desired_position)

    def get_wind(self):
        """
        Returns wind vector based on selected weather mode.
        """

        if WEATHER_MODE == "none":
            return np.zeros(3)

        elif WEATHER_MODE == "medium":
            return np.random.uniform(-0.3, 0.3, size=3)

        elif WEATHER_MODE == "strong":
            return np.random.uniform(-1.0, 1.0, size=3)

        else:
            return np.zeros(3)

    def update_position(self, neighbor_positions, behavior_algorithms):
        """
        Updates the drone's position using:
        - Behavior algorithms
        - Weather disturbance
        - Optional PSO optimization
        """

        # Step 1: Apply behavior algorithms
        new_positions = []
        for algorithm in behavior_algorithms:
            new_pos = algorithm.apply(self, neighbor_positions, self.position.copy())
            new_positions.append(new_pos)

        # Step 2: Average result
        if len(new_positions) > 0:
            new_position = np.mean(new_positions, axis=0)
        else:
            new_position = self.position.copy()

        # Step 3: Apply wind disturbance
        wind = self.get_wind()
        new_position = new_position + wind

        # Step 4: Update position (before optimization)
        self.position = new_position

        # Step 5: Update target position
        if behavior_algorithms:
            self.target_position = behavior_algorithms[-1].apply(
                self, neighbor_positions, self.position.copy()
            )

        # Step 6: PSO Optimization (OPTIONAL)
        if USE_PSO:
            optimized_position = self.pso.optimize(self, self.target_position)
            self.position = optimized_position

    def communicate(self):
        return self.position

    def get_position(self):
        return self.position