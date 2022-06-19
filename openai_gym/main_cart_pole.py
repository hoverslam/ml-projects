import numpy as np
from q_learning import QLearning


if __name__ == "__main__":
    
    # Q-Learning
    parameter = (0.1, 0.99, 0.1)  # Learning rate, discount rate, epsilon
    bin_size = 30
    bins = [
        np.linspace(-4.8, 4.8, bin_size),      # Cart Position
        np.linspace(-4.0, 4.0, bin_size),      # Cart Velocity
        np.linspace(-0.418, 0.418, bin_size),  # Pole Angle
        np.linspace(-4.0, 4.0, bin_size)       # Pole Angular Velocity
    ]
    q = QLearning("CartPole-v1", parameter, bins)
    q.train_agent(5000, 1000)
    q.save(".work", "Q-CartPole")
    q.plot_performance()    
    