import gym
import numpy as np


def make_env(args):
    import gym_env
    env = gym.make("GridEnv-v0", rows=5, cols=5,
                    x_rooms=2, y_rooms=2, n_action=4)
    return env

m = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN"}

def main(args):
    env = make_env(args)
    s = env.reset()
    while True:
        a = np.random.randint(4)
        s_p, r, d, _ = env.step(a)
        print("{} -> {} -> {}".format(s, m[a], s_p))
        s = s_p

if __name__ == "__main__":
    main(None)
