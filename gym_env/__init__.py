import gym
from gym.envs.registration import register


register(
    id='GridEnv-v0',
    entry_point='gym_env.grid_env.grid_env:GridEnv',
    kwargs={},
    max_episode_steps=10000
    )


def make_env(args, manager):
    env = gym.make(
        "GridEnv-v0", rows=5, cols=5,
        x_rooms=2, y_rooms=2, n_action=4,
        state_dim=args.state_dim, start=args.start,
        target=args.target, exits=args.exits, manager=manager)
    return env
