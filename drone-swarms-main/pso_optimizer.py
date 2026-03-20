import numpy as np

class PSOOptimizer:
    def __init__(self, num_particles=10, inertia=0.5, cognitive=1.5, social=1.5):
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def optimize(self, drone, target_position):
        """
        Optimizes drone position using PSO.
        """

        # Initialize particles around current position
        particles = np.array([
            drone.position + np.random.uniform(-1, 1, 3)
            for _ in range(self.num_particles)
        ])

        velocities = np.zeros_like(particles)

        personal_best = particles.copy()
        personal_best_scores = np.array([
            np.linalg.norm(p - target_position) for p in particles
        ])

        global_best = personal_best[np.argmin(personal_best_scores)]

        # PSO iterations (small for speed)
        for _ in range(10):
            for i in range(self.num_particles):
                # Update velocity
                r1, r2 = np.random.rand(), np.random.rand()

                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive * r1 * (personal_best[i] - particles[i])
                    + self.social * r2 * (global_best - particles[i])
                )

                # Update position
                particles[i] += velocities[i]

                # Evaluate
                score = np.linalg.norm(particles[i] - target_position)

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = score

            # Update global best
            global_best = personal_best[np.argmin(personal_best_scores)]

        return global_best