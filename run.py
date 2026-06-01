from agents import (
    UncertainAgent,
)
from model import Wizard
import pyglet

from lab4 import (MDPAgent, MDP)

from game import MDPGame
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--map", type=str, default="open", help="Map name")

parser.add_argument("--speed", type=float, default=10, help="Moves per second")
parser.add_argument(
    "--value_iteration_steps", type=int, default=100,
    help="Number of steps to run value iteration"
)
parser.add_argument(
    "--escape_reward", type=float, default=100,
    help="Reward given to agent if it can escape through the portal"
)
parser.add_argument(
    "--living_reward", type=float, default=0,
    help="Reward given to agent at each step it is alive"
)
parser.add_argument(
    "--death_reward", type=float, default=-50,
    help="Reward given to agent at its death"
)

parser.add_argument(
    "--discount", type=float, default=0.9, help="Future discount factor gamma"
)

parser.add_argument(
    "--timeout", type=int, default=60, help="Maximum time (seconds) to run"
)

parser.add_argument(
    "--no_render", action="store_true",
    help="Whether to not render search nodes"
)
parser.add_argument("--debug", action="store_true", help="Enable debug output")
args = parser.parse_args()

if __name__ == "__main__":
    game = MDPGame(
        path=f"maps/{args.map}",
        game_tick_interval=1.0 / args.speed,
        no_render=args.no_render,
        debug=args.debug,
        timeout=args.timeout,
    )

    mdp = MDP(initial_state=game.state,
              escape_reward=args.escape_reward,
              living_reward=args.living_reward,
              death_reward=args.death_reward,
              discount=args.discount)

    for _ in game.state.get_all_entity_locations(Wizard):
        agent = MDPAgent(mdp=mdp,
                         value_iteration_steps=args.value_iteration_steps)

        print("\n=== Value Iteration Results ===")
        print(f"Grid Size: {mdp.game_state.grid_size}")
        print("\nValue Grid (higher values = more valuable):")
        for row in range(mdp.game_state.grid_size[0]):
            print("  ".join(
                f"{mdp.game_state.tile_grid[row][col].__class__.__name__:>2}"
                for col in range(mdp.game_state.grid_size[1])))

        print("\nTile Values:")
        for row in range(mdp.game_state.grid_size[0]):
            rowValues = []
            for col in range(mdp.game_state.grid_size[1]):
                value = agent.values.value_grid[row][col]
                rowValues.append(f"{value:7.2f}")
            print("  ".join(rowValues))

        game.register_entity_agent(agent, 1)

    game.run()

    if not args.no_render:
        pyglet.app.run()
