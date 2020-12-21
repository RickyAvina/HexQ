import numpy as np
import multiprocessing
from multiprocessing import Manager
from gym_env import make_env
from hexq.hexQ import HexQ


render = True


def main(args):
    hQ = HexQ((0, 0))

    if render:
        multiprocessing.set_start_method("spawn")
        with Manager() as manager:
            main_loop(args, manager, hQ)
    else:
        main_loop(args, None, hQ)


def main_loop(args, manager, hQ):
    # TODO Minor: I would rename as train func
    env = make_env(args, manager)
    s = env.reset()

    while True:
        a = np.random.randint(4)
        s_p, r, d, _ = env.step(a)
        s = s_p

        if not hQ.freq_discovered:
            hQ.explore(s)
            print("Exploring {}".format(s))

        #time.sleep(0.05)


if __name__ == "__main__":
    # TODO Use argparse for hyperparameter
    # TODO requirements.txt and python virtualenv
    main(None)
