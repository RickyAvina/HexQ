import gym
import numpy as np
import multiprocessing
from multiprocessing import Manager
from hexQ import HexQ

render = True

def make_env(args, manager):
    # TODO Put make_env inside gym_env.__init__.py
    # from gym_env import make_env

    env = gym.make(
        "GridEnv-v0", rows=5, cols=5,
        x_rooms=2, y_rooms=2, n_action=4,
        start=(0, 0), target_loc=(2, 15), manager=manager)
    return env


def main(args):
    hexQ = HexQ((0, 0))

    if render:
        multiprocessing.set_start_method("spawn")
        with Manager() as manager:
            main_loop(args, manager, hexQ)
    else:
        main_loop(args, None, hexQ)


def main_loop(args, manager, hexQ):
    # TODO Minor: I would rename as train func
    env = make_env(args, manager)
    s = env.reset()

    while True:
        a = np.random.randint(4)
        s_p, r, d, _ = env.step(a)
        s = s_p

        if not hexQ.freq_discovered:
            hexQ.explore(s)

        #time.sleep(0.05)


if __name__ == "__main__":
    # TODO Use argparse for hyperparameter
    # TODO requirements.txt and python virtualenv
    main(None)
