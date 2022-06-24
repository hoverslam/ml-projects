import os
import typer
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
        case "Actor-Critic":
            actor = ([1028, 512], 0.00003)   # Units per layer, learning rate for actor
            critic = ([1028, 512], 0.00005)  # Units per layer, learning rate for critic
            return (TD0ActorCritic("LunarLander-v2", actor, critic), "AC-LunarLander")
        case _:
            raise ValueError("No such method found!")       


if __name__ == "__main__":
    app()
