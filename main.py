import os
import argparse
import multiprocessing
from multiprocessing import Manager
from gym_env import make_env
from hexq.hexQ import HexQ
from misc.utils import set_log
from tensorboardX import SummaryWriter
from render.gui import GUI
import sys


# Should move this to a Constants file:
exits = {(14, 0), (22, 0),
         (10, 1), (22, 1),
         (2, 2), (14, 2), (2, 3),
         (10, 3)}

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

    if args.start is not None:
        args.start = tuple(args.start)

    args.target = tuple(args.target)
    args.exits = exits

    gui = None
    if args.render:
        multiprocessing.set_start_method("spawn")
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        gui = GUI(args.gui_width, args.gui_height, args.rows, args.cols, args.x_rooms,
                  args.y_rooms, args.target, args.exits, queue)
    train(args, gui)

    if gui:
        gui.process.join()
        sys.exit()

# TODO Please move train function into a new file (e.g., trainer.py)
def train(args, gui):
    # Make environment
    env = make_env(args, gui)

    # Initialize HexQ algorithm
    hq = HexQ(env=env, state_dim=args.state_dim, start=args.start, target=args.target)
    #hq.alg()

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

    parser.add_argument('--state_dim', default=2, type=int,
        help="Number of dimensions in the environment")
    parser.add_argument('--rows', default=5, type=int,
        help="Number of rows in every room")
    parser.add_argument('--cols', default=5, type=int,
        help="Number of cols in every room")
    parser.add_argument('--x_rooms', default=2, type=int,
        help="Number of rooms in every row of a floor")
    parser.add_argument('--y_rooms', default=2, type=int,
        help="Number of rooms in every col of a floor")
    parser.add_argument('--gui_width', default=800, type=int,
        help="Width of GUI in pixels")
    parser.add_argument('--gui_height', default=600, type=int,
        help="Height of GUI in pixels")
    parser.add_argument('--start', nargs='*', type=int,
        help="Starting location")
    parser.add_argument('--target', nargs='*', type=int,
        help="Target position, use 2 variables to specify a position in room,\
              and 1 variable to specify a whole room")
    parser.add_argument('--render', action='store_true',
        help="If True, render GUI")
    parser.add_argument(
        "--test_model", default="", type=str,
        help="Specify model to test")

    args = parser.parse_args()
    args.log_name = "env:GridWorld-v0-s_prefix::%s" % (args.prefix)

    main(args=args)
