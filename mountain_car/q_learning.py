import numpy as np
import matplotlib.pyplot as plt
import gym

from tqdm import tqdm


# Helper functions
def get_state(observation):
    states = []
    for i in range(n_states):
        states.append(np.digitize(observation[i], state_bins[i]))
        
    return tuple(states)

def policy(observation, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[get_state(observation)])


# Settings
BINS = 20
ALPHA = 0.1  # learning rate
GAMMA = 0.95  # discount rate
EPSILON = 0.1
EPISODES = 5000
SHOW_EVERY = 10000


# Initialize environment
env = gym.make("MountainCar-v0")
state_space_high = env.observation_space.high
state_space_low = env.observation_space.low
n_actions = env.action_space.n
n_states = len(env.observation_space.low)

Q = np.zeros((BINS, BINS, n_actions))
history = {"episode": [], "reward" : []}

state_bins = []
for i in range(n_states):
    state_bins.append(np.linspace(state_space_low[i], state_space_high[i], BINS))


# Training
for episode in tqdm(range(EPISODES)):
    observation = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = policy(observation, EPSILON)
        new_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        
        td_target = reward + GAMMA * np.max(Q[get_state(new_observation)])
        td_error = td_target - Q[get_state(observation)][action]
        Q[get_state(observation)][action] += ALPHA * td_error
        
        observation = new_observation
        
        if ((episode + 1) % SHOW_EVERY == 0):
            env.render()
    
    history["episode"].append(episode+1)
    history["reward"].append(episode_reward)
            

# Plot results
plt.scatter(history["episode"], history["reward"], s=1, alpha=0.5)
plt.show()