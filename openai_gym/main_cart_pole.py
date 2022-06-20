import numpy as np
from q_learning import QLearning
from actor_critic import TD0ActorCritic


if __name__ == "__main__":
    
    # Q-Learning
    parameter = (0.99, [1.0, 0.01], [1.0, 0.01])  # Discount rate, learning rate, epsilon
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
    
    # TD(0) Actor-Critic
    actor = (256, 128, 0.0001)    # Units for layer1, layer2, learning rate
    critic = (256, 128, 0.0003)   # Units for layer1, layer2, learning rate
    ac = TD0ActorCritic("CartPole-v1", actor, critic)
    ac.train_agent(1000, 100)
    ac.save(".work", "AC-CartPole")
    
    # Performance
    q.plot_performance()
    ac.plot_performance()
    