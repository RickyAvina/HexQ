import gym
from gym.envs.registration import register


register(
    id='GridEnv-v0',
    entry_point='gym_env.grid_env.grid_env:GridEnv',
    kwargs={}
    )

register(
    id='Taxi-v4',
    entry_point='gym_env.taxi_env.taxi_env:TaxiEnv',
    kwargs={}
    )

def make_env(args, gui):
    if args.env == "GridEnv-v0":
        env = gym.make("GridEnv-v0",
            rows=args.rows, cols=args.cols,
            x_rooms=args.x_rooms, y_rooms=args.y_rooms, n_action=4,
            state_dim=args.state_dim, start=args.start,
            target=args.target, exits=args.exits, gui=gui)
    elif args.env == "Taxi-v4":
        env = gym.make("Taxi-v4")
    else:
        raise ValueError("Environment {} not recognized!".format(args.env))
    return env
