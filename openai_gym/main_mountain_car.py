import numpy as np
from q_learning import QLearning


if __name__ == "__main__":
    
    # Q-Learning
    parameter = (0.1, 0.95, 0.1)  # Learning rate, discount rate, epsilon
    bin_size = 20
    bins = [
            np.linspace(-1.2, 0.6, bin_size),      # Position
            np.linspace(-0.07, 0.07, bin_size)     # Velocity
        ]
    q = QLearning("MountainCar-v0", parameter, bins)
    q.train_agent(5000, 1000)
    q.save(".work", "Q-MountainCar")
    q.plot_performance()
    