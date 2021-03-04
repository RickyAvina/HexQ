import gym
from gym.envs.registration import register


register(
    id='GridEnv-v0',
    entry_point='gym_env.grid_env.grid_env:GridEnv',
    kwargs={}
    )


def make_env(args, gui):
    env = gym.make(
        env="GridEnv-v0", rows=args.rows, cols=args.cols,
        x_rooms=args.x_rooms, y_rooms=args.y_rooms, n_action=4,
        state_dim=args.state_dim, start=args.start,
        target=args.target, exits=args.exits, gui=gui)
    return env
