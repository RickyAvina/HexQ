import numpy as np
import multiprocessing
from multiprocessing import Manager
from gym_env import make_env
from hexq.hexQ import HexQ
import time

render = True 


def main(args):
    if render:
        multiprocessing.set_start_method("spawn")
        with Manager() as manager:
            main_loop(args, manager)
    else:
        main_loop(args, None)


def main_loop(args, manager):
    # TODO Minor: I would rename as train func
    env = make_env(args, manager)
    s = env.reset()

    while True:
        a = np.random.randint(4)
        s_p, r, d, _ = env.step(a)
        #print("{}->{}->{}".format(s, a, s_p))
        s = s_p
    
    #hQ = HexQ((0, 0), env, 2)
    #hQ.alg()


if __name__ == "__main__":
    # TODO Use argparse for hyperparameter
    # TODO requirements.txt and python virtualenv
    main(None)
