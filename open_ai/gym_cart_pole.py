import os
import typer
import numpy as np
from q_learning import QLearning
from actor_critic import TD0ActorCritic


app = typer.Typer()
dirname = os.path.dirname(__file__)

@app.command()
def train(method: str, num_epsiodes: int, show_every: int):
    agent, name = build_agent(method)
    agent.train_agent(num_epsiodes, show_every)
    agent.save_model(os.path.join(dirname, "models"), name)
    agent.save_stats(os.path.join(dirname, "json"), name)

@app.command()    
def play(method: str, num_episodes: int):
    agent, name = build_agent(method)
    agent.load_model(os.path.join(dirname, "models"), name)
    agent.play(num_episodes)
    input("Press ENTER to exit.")

@app.command()
def performance(method: str):
    agent, name = build_agent(method)
    agent.load_stats(os.path.join(dirname, "json"), name)
    agent.plot_performance()
    
def build_agent(method: str):
    match method:
        case "Q-learning":
            parameter = (0.99, [1.0, 0.01], [1.0, 0.01])  # Discount rate, learning rate, epsilon
            bins = [
                np.linspace(-4.8, 4.8, 30),      # Cart Position
                np.linspace(-4.0, 4.0, 30),      # Cart Velocity
                np.linspace(-0.418, 0.418, 30),  # Pole Angle
                np.linspace(-4.0, 4.0, 30)       # Pole Angular Velocity
            ]
            return (QLearning("CartPole-v1", parameter, bins), "Q-CartPole")
        case "Actor-Critic":
            actor = ([32, 32], 0.0002)   # Units per layer, learning rate for actor
            critic = ([32, 32], 0.0004)  # Units per layer, learning rate for critic
            return (TD0ActorCritic("CartPole-v1", actor, critic), "AC-CartPole")
        case _:
            raise ValueError("No such method found!")       


if __name__ == "__main__":
    app()
    