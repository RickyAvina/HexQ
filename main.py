import gym
import numpy as np
import multiprocessing
from multiprocessing import Manager
import time


render = True

def make_env(args, manager):
    import gym_env

    env = gym.make("GridEnv-v0", rows=5, cols=5,
                    x_rooms=2, y_rooms=2, n_action=4,
                    start=(0, 0), target_loc=(2, 15), manager=manager)
    return env

def main(args):
    if render:
        multiprocessing.set_start_method("spawn")
        with Manager() as manager:
            main_loop(args, manager)
    else:
        main_loop(args, None)


def main_loop(args, manager):
    env = make_env(args, manager)
    s = env.reset()

    while True:
        a = np.random.randint(4)
        s_p, r, d, _ = env.step(a)
        s = s_p
        time.sleep(0.05)


if __name__ == "__main__":
    main(None)
