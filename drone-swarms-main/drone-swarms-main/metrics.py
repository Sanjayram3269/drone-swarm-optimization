import numpy as np

class MetricsTracker:
    def __init__(self):
        self.errors = []

    def compute_total_error(self, drones):
        """
        Computes total formation error of all drones.
        """
        total_error = 0

        for drone in drones:
            error = np.linalg.norm(drone.position - drone.target_position)
            total_error += error

        self.errors.append(total_error)

    def get_errors(self):
        return self.errors