import gym
from gym import spaces
import random
from misc.utils import random_exclude


class GridEnv(gym.Env):
    def __init__(self, rows, cols, x_rooms, y_rooms, n_action, state_dim, target, exits, start=None, gui=None):
        '''
        target loc: tuple representing target. (1,2,3) would mean Floor 3, Room 2, pos 1
                    (2, 3) would mean room 2, floor 3

        exits: set of tuples that define primitive exits
        '''

        super(GridEnv, self).__init__()
        self.rows = rows
        self.cols = cols
        self.x_rooms = x_rooms
        self.y_rooms = y_rooms
        self.state_dim = state_dim
        self.target = target
        if start is None:
            self.start = self.get_random_start()
        else:
            self.start = start
        self.observation_space = spaces.Tuple((spaces.Discrete(cols),
                                               spaces.Discrete(rows),
                                               spaces.Discrete(x_rooms),
                                               spaces.Discrete(y_rooms)))
        self.action_space = spaces.Discrete(n_action)
        self.exits = exits
        self.gui = gui

    def target_reached(self):
        if len(self.target) == 1:
            return self.agent_loc[1:] == self.target
        elif len(self.target) == 2:
            return self.agent_loc == self.target
        else:
            raise ValueError("state dim: {} not supported!".format(self.state_dim))

    def step(self, action):
        assert self.agent_loc is not None, "agent loc is None!"
        if not self.target_reached():
            self._take_action(action)

        next_observation = self.agent_loc
        target_reached = self.target_reached()

        if target_reached:
            reward = 10
        else:
            reward = -1

        return next_observation, reward, target_reached, dict()

    def get_random_start(self, states=None):
        # pick random starting point that isn't in target
        if states is not None:
            return random.choice(tuple(states))

        if len(self.target) == 1:
            rand_room = random_exclude({self.target[0]}, 0, self.x_rooms*self.y_rooms-1)
            rand_pos = random.randint(0, self.rows*self.cols-1)
        else:
            rand_room = random.randint(0, self.x_rooms*self.y_rooms-1)
            rand_pos = random_exclude({self.target[0]}, 0, self.rows*self.cols-1)
        return (rand_pos, rand_room)

    def reset(self):
        self.agent_loc = self.get_random_start()
        return self.agent_loc

    def reset_in(self, states):
        self.agent_loc = self.get_random_start(states)
        return self.agent_loc

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
