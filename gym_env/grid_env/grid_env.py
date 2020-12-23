import gym
import numpy as np
from gym import spaces
import render.gui as gui


class GridEnv(gym.Env):
    def __init__(self, rows, cols, x_rooms, y_rooms, n_action, start, target_loc, manager=None):
        super(GridEnv, self).__init__()
        self.rows = rows
        self.cols = cols
        self.x_rooms = x_rooms
        self.y_rooms = y_rooms
        self.start = start
        self.target_loc = target_loc
        self.target_reached = False
        # agent loc is (x, y, room_x, room_y)
        self.observation_space = spaces.Tuple((spaces.Discrete(cols),
                                               spaces.Discrete(rows),
                                               spaces.Discrete(x_rooms),
                                               spaces.Discrete(y_rooms)))
        self.action_space = spaces.Discrete(n_action)
        self.exits = {(14, 0), (22, 0),
                      (10, 1), (22, 1),
                      (2, 2), (14, 2), (2, 3),
                      (10, 3)}


        self.manager = manager
        if manager is not None:
            self.pos_queue = manager.list()
            gui.setup(width=600, height=600, rows=rows, cols=cols,
                      x_rooms=x_rooms, y_rooms=y_rooms, target_loc=target_loc,
                      exits=self.exits, action_queue=self.pos_queue)

    def step(self, action):
        self._take_action(action)

        if self.manager is not None:
            # add pos to gui pos queue
            self.pos_queue.append(self.agent_loc)

        next_observation = self.agent_loc
        if np.array_equal(self.agent_loc, self.target_loc):
            reward = 0
            self.target_reached = True
            self.reset()
        else:
            reward = -1

        return (next_observation, reward, self.target_reached, {})

    def reset(self):
        self._init_env()
        return self.agent_loc

    def _init_env(self):
        self.agent_loc = self.start

    def _assert_valid_pos(self, loc, action=None):
        assert loc[0] >= 0 and loc[0] < self.cols*self.rows, \
            "pos {} out of bounds".format(loc[0]) + "from action {}".format(action) if action is not None else ""
        assert loc[1] >= 0 and loc[1] < self.x_rooms * self.y_rooms, \
            "room {} out of boudns".format(loc[1]) + "from action {}".format(action) if action is not None else ""

    def _assert_valid_exits(self):
        raise NotImplementedError()

    def _take_action(self, action):
        if action == 0:  # left
            if self.agent_loc[0] % self.cols != 0:  # left edge
                self.agent_loc = (self.agent_loc[0]-1, self.agent_loc[1])
            elif self.agent_loc in self.exits:
                self.agent_loc = (self.agent_loc[0] + self.cols-1, self.agent_loc[1] - 1)
        elif action == 1:  # right
            if self.agent_loc[0] % self.cols != self.cols-1:  # right edge
                self.agent_loc = (self.agent_loc[0]+1, self.agent_loc[1])
            elif self.agent_loc in self.exits:
                self.agent_loc = (self.agent_loc[0] - (self.cols-1), self.agent_loc[1] + 1)
        elif action == 2:  # up
            if self.agent_loc[0] // self.cols != 0:  # top edge
                self.agent_loc = (self.agent_loc[0]-self.cols, self.agent_loc[1])
            elif self.agent_loc in self.exits:
                self.agent_loc = (self.agent_loc[0] + (self.rows-1)*self.cols, self.agent_loc[1]-self.x_rooms)
        elif action == 3:   # down
            if self.agent_loc[0] // self.cols != self.rows-1:  # bottom edge
                self.agent_loc = (self.agent_loc[0]+self.cols, self.agent_loc[1])
            elif self.agent_loc in self.exits:
                self.agent_loc = (self.agent_loc[0] - ((self.rows-1)*self.cols), self.agent_loc[1]+self.x_rooms)
        else:
            raise ValueError("Incorrect Action")

        self._assert_valid_pos(self.agent_loc, action)
