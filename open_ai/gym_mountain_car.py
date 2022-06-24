import os
import typer
import numpy as np
from q_learning import QLearning


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
                np.linspace(-1.2, 0.6, 20),      # Position
                np.linspace(-0.07, 0.07, 20)     # Velocity
            ]
            return (QLearning("MountainCar-v0", parameter, bins), "Q-MountainCar")
        case _:
            raise ValueError("No such method found!")       


if __name__ == "__main__":
    app()
