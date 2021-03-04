from hexq.hexQ import HexQ
from gym_env import make_env


def train(args, gui):
    # Make environment
    env = make_env(args, gui)

    # Initialize HexQ algorithm
    hq = HexQ(env=env, args=args)
