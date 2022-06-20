import json
import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tqdm import tqdm


class TD0ActorCritic():
    
    def __init__(self, env_name, actor, critic, gamma=0.99) -> None:
        # Environment
        self.env = gym.make(env_name)
        self.num_actions = self.env.action_space.n
        self.num_inputs = len(self.env.observation_space.high)
        
        # Agent
        self.actor_l1, self.actor_l2, self.actor_lr = actor
        self.critic_l1, self.critic_l2, self.critic_lr = critic
        self.gamma = gamma
        self.actor, self.critic = self.build_agent()
        self.optimizer_actor = keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.optimizer_critic = keras.optimizers.Adam(learning_rate=self.critic_lr)
        
        # Stats
        self.history = {"episode": [], "reward": []}
        
    def build_agent(self):
        actor = keras.Sequential([
            keras.layers.Dense(self.actor_l1, activation="relu"),
            keras.layers.Dense(self.actor_l2, activation="relu"),
            keras.layers.Dense(self.num_actions, activation="softmax")
        ])
        
        critic = keras.Sequential([
            keras.layers.Dense(self.critic_l1, activation="relu"),
            keras.layers.Dense(self.critic_l2, activation="relu"),
            keras.layers.Dense(1)
        ])
        
        return actor, critic

    def train_agent(self, num_episodes: int, show_every: int) -> None:
        for episode in tqdm(range(num_episodes)):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
                    obs = tf.convert_to_tensor([obs])

                    action_probs = self.actor(obs)
                    state_value  = self.critic(obs)
                    
                    action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
                    action_prob = action_probs[0, action]

                    next_obs, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    
                    next_state_value = self.critic(tf.convert_to_tensor([next_obs]))
                    error = (reward + self.gamma * next_state_value - state_value) * (not done)

                    actor_loss = error * tf.math.log(action_prob)
                    critic_loss = error * state_value
                    
                    actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
                    critic_grads = tape_critic.gradient(-critic_loss, self.critic.trainable_variables)
        
                    self.optimizer_actor.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                    self.optimizer_critic.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                    
                    obs = next_obs
                
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
                obs = tf.convert_to_tensor([obs])
                
                action_probs = self.actor(obs)                    
                action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))

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
        plt.title("{}: TD(0) Actor-Critic".format(self.env.unwrapped.spec.id))
        plt.show()

    def save(self, filepath, filename, stats=True):
        src = "{}/{}".format(filepath, filename)
        self.actor.save("{}_model.h5".format(src))
        
        if stats:
            with open("{}_stats.json".format(src), "w") as f:
                f.write(json.dumps(self.history))
                f.close()
        
    def load(self, filepath, filename, stats=True):
        src = "{}/{}".format(filepath, filename)
        self.actor = keras.models.load_model("{}_model.h5".format(src))
        
        if stats:
            self.history = json.load(open("{}_stats.json".format(src)))
