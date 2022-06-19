import json
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class QPlayer():
    
    def __init__(self, num_actions: int, num_inputs: int, bins: np.ndarray):
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        
        self.bins = bins
        self.bin_size = bins[0].shape[0]
        self.Q = np.zeros(([self.bin_size] * num_inputs + [num_actions]), dtype=np.float32)              
    
    def update_table(self, reward: float, action: int, obs: list[float], new_obs: list[float],
                     gamma: float, alpha: float):
        td_target = reward + gamma * np.max(self.Q[self.get_state(new_obs)])
        td_error = td_target - self.Q[self.get_state(obs)][action]
        self.Q[self.get_state(obs)][action] += alpha * td_error

    def get_state(self, obs: list[int]) -> tuple[int]:
        states = []
        for i in range(len(obs)):
            states.append(np.digitize(obs[i], self.bins[i]))
            
        return tuple(states)
        
    def policy(self, obs: list[int], epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[self.get_state(obs)])
        
    def save_table(self, filepath: str, filename: str):
        np.save("{}/{}_model".format(filepath, filename), self.Q)
        
    def load_table(self, filepath:str, filename: str):
        self.Q = np.load("{}/{}_model.npy".format(filepath, filename))
    

class QLearning():
    
    def __init__(self, env_name: str, parameter: tuple[float], bins: np.ndarray) -> None:
        # Environment
        self.env = gym.make(env_name)
        self.num_actions = self.env.action_space.n
        self.num_inputs = len(self.env.observation_space.high)              
        
        # Agent
        self.gamma, self.alpha, self.epsilon = parameter
        self.agent = QPlayer(self.num_actions, self.num_inputs, bins)
        
        # Stats
        self.history = {"episode": [], "reward": []}

    def train_agent(self, num_episodes: int, show_every: int) -> None:
        epsilons = self.decay_schedule(self.epsilon[0], self.epsilon[1], 0.5, num_episodes)
        alphas = self.decay_schedule(self.alpha[0], self.alpha[1], 0.5, num_episodes)
        
        for episode in tqdm(range(num_episodes)):
            obs = self.env.reset()
            done = False
            episode_reward = 0
                
            while not done:
                action = self.agent.policy(obs, epsilons[episode])
                new_obs, reward, done, _ = self.env.step(action)
                new_obs = new_obs
                episode_reward += reward
                
                self.agent.update_table(reward, action, obs, new_obs, self.gamma, alphas[episode])        
                obs = new_obs
                
                if ((episode + 1) % show_every == 0):
                    self.env.render()
                    
            self.history["episode"].append(episode+1)
            self.history["reward"].append(episode_reward)
            
    def play(self, num_episodes, show=True):
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.agent.policy(obs, 0)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                    
                if show:
                    self.env.render()
                    
            print("Reward: {:.2f} at episode {}".format(episode_reward, episode+1))
                    
    def plot_performance(self):
        rewards = self.history["reward"]
        mov_avg = [np.average(rewards[:i+1]) for i in range(len(rewards))]
        mov_avg_100 = mov_avg[:100] + [np.average(rewards[i:i+100]) for i in range(len(rewards)-100)]

        plt.scatter(self.history["episode"], rewards, s=1, alpha=0.5)
        plt.plot(self.history["episode"], mov_avg_100, color="red", alpha=0.7)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()

    def save(self, filepath, filename, stats=True):             
        self.agent.save_table(filepath, filename)  
              
        if stats:
            src = "{}/{}".format(filepath, filename)               
            with open("{}_stats.json".format(src), "w") as f:
                f.write(json.dumps(self.history))
                f.close()
        
    def load(self, filepath, filename, stats=True):                     
        self.agent.load_table(filepath, filename)
        
        if stats:
            src = "{}/{}".format(filepath, filename)
            self.history = json.load(open("{}_stats.json".format(src)))
            
    def decay_schedule(self, init_value, min_value, decay_ratio, max_steps):
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        
        values = np.logspace(-2, 0, decay_steps, base=10, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), "edge")
        
        return values