import numpy as np
from q_learning import QLearning
from actor_critic import TD0ActorCritic


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
    
    # TD(0) Actor-Critic
    actor = (32, 32, 0.0001)    # Units for layer1, layer2, learning rate
    critic = (64, 64, 0.0005)   # Units for layer1, layer2, learning rate
    ac = TD0ActorCritic("MountainCar-v0", actor, critic)
    ac.train_agent(5000, 1000)
    ac.save(".work", "AC-MountainCar")    
    
    # Performance
    q.plot_performance()
    ac.plot_performance() 
    