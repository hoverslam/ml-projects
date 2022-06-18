import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Q-learning agent
class CartPolePlayer():
    
    def __init__(self, alpha: float, gamma: float, epsilon: float, 
                 num_actions: int, num_inputs: int, bin_size: int):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.bin_size = bin_size
        
        self.Q = np.random.uniform(-1.0, 1.0, size=([bin_size] * num_inputs + [num_actions]))
        self.bins = self.create_bins()        
    
    def update_table(self, reward: float, action: int, obs: list[float], new_obs: list[float]):
        td_target = reward + self.gamma * np.max(self.Q[self.get_state(new_obs)])
        td_error = td_target - self.Q[self.get_state(obs)][action]
        self.Q[self.get_state(obs)][action] += self.alpha * td_error
        
    def create_bins(self):
        return [
            np.linspace(-4.8, 4.8, self.bin_size),      # Cart Position
            np.linspace(-4.0, 4.0, self.bin_size),      # Cart Velocity
            np.linspace(-0.418, 0.418, self.bin_size),  # Pole Angle
            np.linspace(-4.0, 4.0, self.bin_size)       # Pole Angular Velocity
        ]
        
    def get_state(self, obs: list[int]) -> tuple[int]:
        states = []
        for i in range(len(obs)):
            states.append(np.digitize(obs[i], self.bins[i]))
            
        return tuple(states)
        
    def policy(self, obs: list[int]) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[self.get_state(obs)])
        
    def save_table(self, file: str):
        np.save(file, self.Q)
        
    def load_table(self, file: str):
        self.Q = np.load(file)
        

# Settings
EPISODES = 10000
SHOW_EVERY = 1000


# Initialize environment
env = gym.make("CartPole-v1")
player = CartPolePlayer(alpha=0.1, gamma=0.99, epsilon=0.1, 
                        num_actions=2, num_inputs=4, bin_size=40)
   
                  
# Training
history = {"episode": [], "reward" : []}
for episode in tqdm(range(EPISODES)):
    obs = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = player.policy(obs)
        new_obs, reward, done, _ = env.step(action)
        new_obs = new_obs
        episode_reward += reward
        
        player.update_table(reward, action, obs, new_obs)        
        obs = new_obs
        
        if ((episode + 1) % SHOW_EVERY == 0):
            env.render()
    
    history["episode"].append(episode+1)
    history["reward"].append(episode_reward)


# Plot results
mov_avg = [np.average(history["reward"][:i+1]) for i in range(len(history["reward"]))]
mov_avg_100 = mov_avg[:100] + [np.average(history["reward"][i:i+100]) for i in range(len(history["reward"])-100)]

plt.scatter(history["episode"], history["reward"], s=1, alpha=0.5)
plt.plot(history["episode"], mov_avg_100, color="red", alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
