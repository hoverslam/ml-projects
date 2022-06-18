import json
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
        
    def save_table(self, filepath: str, filename: str):
        np.save("{}/{}".format(filepath, filename), self.Q)
        
    def load_table(self, filepath:str, filename: str):
        self.Q = np.load("{}/{}.npy".format(filepath, filename))    


class GymCartPole():
    
    def __init__(self) -> None:
        # Initialize environment
        self.env = gym.make("CartPole-v1")
        self.num_actions = 2
        self.num_inputs = 4
        
        
        # Initialize agent
        self.agent = CartPolePlayer(alpha=0.1, gamma=0.99, epsilon=0.1, bin_size=30, 
                                    num_actions=self.num_actions, num_inputs=self.num_inputs)
        
        # Stats
        self.history = {"episode": [], "reward": []}

    def train_agent(self, num_episodes: int, show_every: int) -> None:
        for episode in tqdm(range(num_episodes)):
            obs = self.env.reset()
            done = False
            episode_reward = 0
                
            while not done:
                action = self.agent.policy(obs)
                new_obs, reward, done, _ = self.env.step(action)
                new_obs = new_obs
                episode_reward += reward
                
                self.agent.update_table(reward, action, obs, new_obs)        
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
                action = self.agent.policy(obs)
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

    def save_agent(self, filepath, filename, stats=True):
        self.agent.save_table(filepath, filename)
        if stats:
            src = "{}/{}".format(filepath, filename)
            with open("{}_stats.json".format(src), "w") as f:
                f.write(json.dumps(self.history))
                f.close()
        
    def load_agent(self, filepath, filename, stats=True):        
        self.agent.load_table(filepath, filename)
        if stats:
            src = "{}/{}".format(filepath, filename)
            self.history = json.load(open("{}_stats.json".format(src)))


if __name__ == "__main__":
    g = GymCartPole()
    g.train_agent(5000, 1000)
    g.save_agent("cart_pole", "q_temp")
    g.plot_performance()
    