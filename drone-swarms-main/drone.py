import numpy as np
from pso_optimizer import PSOOptimizer

class Drone:
    """
    Represents a single drone in the swarm with basic movement and communication capabilities.
    """

    def __init__(self, position, index):
        """
        Initializes a drone with a given position and index.
        """
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

    def update_position(self, neighbor_positions, behavior_algorithms):
        """
        Updates the drone's position using:
        - Behavior algorithms
        - Wind disturbance
        - PSO optimization
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

        # Step 3: Add dynamic wind disturbance
        wind = np.random.uniform(-0.3, 0.3, size=3)
        new_position = new_position + wind

        # Step 4: Temporary update (before optimization)
        self.position = new_position

        # Step 5: Update target position
        if behavior_algorithms:
            self.target_position = behavior_algorithms[-1].apply(
                self, neighbor_positions, self.position.copy()
            )

        # Step 6: PSO Optimization (MAIN INTELLIGENCE)
        optimized_position = self.pso.optimize(self, self.target_position)
        self.position = optimized_position

    def communicate(self):
        """
        Returns the position information that the drone shares with others.
        """
        return self.position

    def get_position(self):
        """
        Retrieves the current position of the drone.
        """
        return self.position