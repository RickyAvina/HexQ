import argparse
import numpy as np
import multiprocessing
from multiprocessing import Manager
from gym_env import make_env
from hexq.hexQ import HexQ
import os
from misc.utils import set_log
from tensorboardX import SummaryWriter


render = False
start = (0, 0)
target = (15, 2)

def main(args):
    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    
    # set logging 
    log = set_log(args)
    tb_writer = SummaryWriter('./logs/tb_{}'.format(args.log_name))

    if render:
        multiprocessing.set_start_method("spawn")
        with Manager() as manager:
            train(args, manager)
    else:
        train(args, None)


def train(args, manager):
    env = make_env(args, manager)

    hq = HexQ(env=env, start=start, target=target)
    hq.alg()

    '''
    s = env.reset()

    while True:
        a = np.random.randint(4)
        s_p, r, d, _ = env.step(a)
        #print("{}->{}->{}".format(s, a, s_p))
        s = s_p
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Algorithm args

    # Misc
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--seed", default=0, type=int,
        help="Sets Gym, PyTorch, and Numpy seeds")
    parser.add_argument(
        "--mode", default="train", type=str,
        help="Choose between training and testing")
    parser.add_argument(
        "--test_model", default="", type=str,
        help="Specify model to test")
   
    args = parser.parse_args()
    args.log_name = "env:GridWorld-v0-s_prefix::%s" % (args.prefix)
    
    main(args=args)
