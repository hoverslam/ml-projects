import typer
import numpy as np
from q_learning import QLearning
from actor_critic import TD0ActorCritic


app = typer.Typer()

@app.command()
def train(method: str, num_epsiodes: int, show_every: int):
    agent, name = build_agent(method)
    agent.train_agent(num_epsiodes, show_every)
    agent.save_model("open_ai/models", name)
    agent.save_stats("open_ai/json", name)

@app.command()    
def play(method: str, num_episodes: int):
    agent, name = build_agent(method)
    agent.load_model("open_ai/models", name)
    agent.play(num_episodes)
    input("Press ENTER to exit.")

@app.command()
def performance(method: str):
    agent, name = build_agent(method)
    agent.load_stats("open_ai/json", name)
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
        case "Actor-Critic":
            # ! Doesn't train => car stays in the valley all the time
            actor = ([64, 32], 0.0001)   # Units per layer, learning rate for actor
            critic = ([64, 32], 0.0005)  # Units per layer, learning rate for critic
            return (TD0ActorCritic("MountainCar-v0", actor, critic), "AC-MountainCar")
        case _:
            raise ValueError("No such method found!")       


if __name__ == "__main__":
    app()
