import gym
import numpy as np
import multiprocessing
from multiprocessing import Manager
import time

def make_env(args, manager):
    import gym_env

    env = gym.make("GridEnv-v0", rows=5, cols=5,
                    x_rooms=2, y_rooms=2, n_action=4,
                    display=True, manager=manager)
    return env


#actions = [2, 0, 1, 1, 3, 3, 3, 3, 3, 1, 3, 3, 1, 1, 1, 1, 2, 2, 2, 3, 2, 2, 2, 0, 0, 0, 0] 

m = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN"}

def main(args):
    multiprocessing.set_start_method("spawn")

    with Manager() as manager:
        env = make_env(args, manager)
        s = env.reset()

        while True:
            a = np.random.randint(4)
            s_p, r, d, _ = env.step(a)
            print("{}->{}->{}".format(s, m[a], s_p))
            s = s_p
            time.sleep(0.2)

'''
               if len(actions) > 0:
                a = actions.pop(0)
                s_p, r, d, _ = env.step(a)
                print("{}->{}->{}".format(s, m[a], s_p))
                s = s_p
                time.sleep(0.2)
'''

if __name__ == "__main__":
    main(None)
