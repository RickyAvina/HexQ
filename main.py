import os
import argparse
import multiprocessing
from multiprocessing import Manager
from gym_env import make_env
from hexq.hexQ import HexQ
from misc.utils import set_log
from tensorboardX import SummaryWriter


render = False  # TODO Please use argparser instead of global variable


def main(args):
    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logging
    # TODO log and tb_writer variables are not used.
    # Please use them accordingly
    log = set_log(args)
    tb_writer = SummaryWriter('./logs/tb_{}'.format(args.log_name))

    if render:
        multiprocessing.set_start_method("spawn")
        with Manager() as manager:
            train(args, manager)
    else:
        train(args, None)


# TODO Please move train function into a new file (e.g., trainer.py)
def train(args, manager):
    # Make environment
    env = make_env(args, manager)

    # Initialize HexQ algorithm
    hq = HexQ(env=env, start=(0, 0), target=(15, 2))
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
    parser = argparse.ArgumentParser(description="HeXQ")

    # Algorithm args

    # Environment args
    # TODO I would put environemnt specific arguments in here, such as the ones in
    # gym_env.__init__.py.

    # Misc
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--seed", default=0, type=int,
        help="Sets Gym, PyTorch, and Numpy seeds")
    # TODO For binary variable, please consider the following argparse example:
    # parser.add_argument(
    #     "--use-lstm", action="store_true",
    #     help="If True, include LSTM in network architecture")
    # In the _train.sh, if you specify --use-lstm argument, then
    # the variable is set to True
    parser.add_argument(
        "--mode", default="train", type=str,
        help="Choose between training and testing")
    parser.add_argument(
        "--test_model", default="", type=str,
        help="Specify model to test")

    args = parser.parse_args()
    args.log_name = "env:GridWorld-v0-s_prefix::%s" % (args.prefix)

    main(args=args)
