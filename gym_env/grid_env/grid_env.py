import gym
import numpy as np
from gym import spaces
import render.gui as gui
import random
from misc.utils import random_exclude


class GridEnv(gym.Env):
    def __init__(self, rows, cols, x_rooms, y_rooms, n_action, state_dim, target, exits, start=None, manager=None):
        '''
        target loc: tuple representing target. (1,2,3) would mean Floor 3, Room 2, pos 1
                    (2, 3) would mean room 2, floor 3

        exits: set of tuples used for graphical rendering 
        '''
        super(GridEnv, self).__init__()
        self.rows = rows
        self.cols = cols
        self.x_rooms = x_rooms
        self.y_rooms = y_rooms
        self.state_dim = state_dim
        if start is None:
            # pick random starting point that isn't in target
            rand_pos = random.randint(0, rows*cols-1)
            if state_dim == 1:
                rand_room = random_exclude({target[0]}, 0, x_rooms*y_rooms-1)
            else:
                rand_room = random.randint(0, x_rooms*y_rooms-1)
            self.start = (rand_pos, rand_room)
        else:
            self.start = start
        self.target = target
        self.observation_space = spaces.Tuple((spaces.Discrete(cols),
                                               spaces.Discrete(rows),
                                               spaces.Discrete(x_rooms),
                                               spaces.Discrete(y_rooms)))
        self.action_space = spaces.Discrete(n_action)
        self.exits = exits
        self.primitive_exits = set()  # exits should be discovered

        self.manager = manager
        if manager is not None:
            self.pos_queue = manager.list()
            gui.setup(width=600, height=600, rows=rows, cols=cols,
                      x_rooms=x_rooms, y_rooms=y_rooms, target=target,
                      exits=self.exits, action_queue=self.pos_queue)

    def target_reached(self):
        if self.state_dim == 1:
            return self.agent_loc[1:] == self.target
        elif self.state_dim == 2:
            return self.agent_loc == self.target
        else:
            raise ValueError("state dim: {} not supported!".format(self.state_dim))

    def step(self, action):
        assert self.agent_loc is not None, "agent loc is None!"
        self._take_action(action)

        if self.manager is not None:
            self.pos_queue.append(self.agent_loc)  # Add pos to gui pos queue

        next_observation = self.agent_loc
        target_reached = False
        if self.target_reached():
            print("target reached!")
            reward = 0
            target_reached = True
            self.reset()
        else:
            reward = -1

        return (next_observation, reward, target_reached, {})

    def reset(self):
        self._init_env()
        return self.agent_loc

    def reset_in(self, states):
        self._init_env(states)
        return self.agent_loc

    def _init_env(self, states=None):
        if states is not None:
            self.agent_loc = random.choice(tuple(states))
        else:
            self.agent_loc = self.start

    def _assert_valid_pos(self, loc, action=None):
        assert loc[0] >= 0 and loc[0] < self.cols * self.rows, \
            "pos {} out of bounds".format(loc[0]) + "from action {}".format(action) if action is not None else ""
        assert loc[1] >= 0 and loc[1] < self.x_rooms * self.y_rooms, \
            "room {} out of boudns".format(loc[1]) + "from action {}".format(action) if action is not None else ""

    def _assert_valid_exits(self):
        raise NotImplementedError()

    def _take_action(self, action):
        if action == 0:  # Left
            if self.agent_loc[0] % self.cols != 0:  # Left edge
                self.agent_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
            elif self.agent_loc in self.exits:
                self.agent_loc = (self.agent_loc[0] + self.cols - 1, self.agent_loc[1] - 1)
        elif action == 1:  # Right
            if self.agent_loc[0] % self.cols != self.cols - 1:  # Right edge
                self.agent_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
            elif self.agent_loc in self.exits:
                self.agent_loc = (self.agent_loc[0] - (self.cols - 1), self.agent_loc[1] + 1)
        elif action == 2:  # Up
            if self.agent_loc[0] // self.cols != 0:  # Top edge
                self.agent_loc = (self.agent_loc[0] - self.cols, self.agent_loc[1])
            elif self.agent_loc in self.exits:
                self.agent_loc = (self.agent_loc[0] + (self.rows - 1) * self.cols, self.agent_loc[1] - self.x_rooms)
        elif action == 3:   # Down
            if self.agent_loc[0] // self.cols != self.rows - 1:  # Bottom edge
                self.agent_loc = (self.agent_loc[0] + self.cols, self.agent_loc[1])
            elif self.agent_loc in self.exits:
                self.agent_loc = (self.agent_loc[0] - ((self.rows - 1) * self.cols), self.agent_loc[1] + self.x_rooms)
        else:
            raise ValueError("Incorrect Action")

        self._assert_valid_pos(self.agent_loc, action)
