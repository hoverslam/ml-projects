import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Q-learning agent
class MountainCarDriver():
    
    def __init__(self, alpha: float, gamma: float, epsilon: float, 
                 num_actions: int, num_inputs: int, bin_size: int):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.bin_size = bin_size
        
        self.Q = np.random.uniform(-1.0, 1.0, size=([bin_size] * num_inputs + [num_actions]))
        self.bins = self.create_bins()        
    
    def update_table(self, reward: float, action: int, obs: list[float], new_obs: list[float]):
        td_target = reward + self.gamma * np.max(self.Q[self.get_state(new_obs)])
        td_error = td_target - self.Q[self.get_state(obs)][action]
        self.Q[self.get_state(obs)][action] += self.alpha * td_error
        
    def create_bins(self):
        return [
            np.linspace(-1.2, 0.6, self.bin_size),      # Position
            np.linspace(-0.07, 0.07, self.bin_size)     # Velocity
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
env = gym.make("MountainCar-v0")
player = MountainCarDriver(alpha=0.1, gamma=0.99, epsilon=0.1, 
                           num_actions=3, num_inputs=2, bin_size=30)
   
                  
# Training
history = {"episode": [], "reward" : [], "average": []}
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
w = int(EPISODES / 100)
mov_avg = np.convolve(history["reward"], np.ones(w)/w, mode="valid")
plt.scatter(history["episode"], history["reward"], s=1, alpha=0.5)
plt.plot(history["episode"][:(-w+1)], mov_avg, color="red", alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()