from hexq.hexQ import HexQ
from gym_env import make_env


def train(args, gui, log, tb_writer):
    """
    Run the HexQ algorithm

    Arguments:
    args      (argparse.Namespace)         command-line arguments
    gui       (render.Gui)                 GUI for rendering
    log       (logging.Log)                log for info and errors
    tb_writer (tensorboardX.SummaryWriter) TensorboardX log
    """

    # Make environment
    env = make_env(args, gui)

    # Initialize HexQ algorithm
    hq = HexQ(env=env, args=args, log=log, tb_writer=tb_writer)
