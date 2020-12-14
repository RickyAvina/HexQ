import gym
import numpy as np
from gym import spaces


class GridEnv(gym.Env):
    def __init__(self, rows, cols, x_rooms, y_rooms, n_action):
        super(GridEnv, self).__init__()

        self.rows = rows
        self.cols = cols
        self.x_rooms = x_rooms
        self.y_rooms = y_rooms
        self.target_loc = (0, 0, 1, 0)

        self.observation_space = spaces.Tuple((spaces.Discrete(cols),
                                               spaces.Discrete(rows),
                                               spaces.Discrete(x_rooms),
                                               spaces.Discrete(y_rooms)))
        self.action_space = spaces.Discrete(n_action)
        self.exits = {(self.cols//2, self.rows//2, 0, 0)}

    def step(self, action):
        self._take_action(action)

        next_observation = self._get_loc()

       if np.array_equal(self.agent_loc, self.target_loc):
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        return (next_observation, reward, done, {});

    def reset(self):
        self._init_env()
        return self.get_pos()

    def render(self):
        raise NotImplentedError()

    def _init_env(self):
        self.agent_loc = (0,0,0,0)

    def assert_valid_pos(loc):
        assert(loc[0] >= 0 and loc[0] < self.cols)
        assert(loc[1] >= 0 and loc[1] < self.rows)
        assert(loc[2] >= 0 and loc[2] < self.x_rooms)
        assert(loc[3] >= 0 and loc[3] < self.y_rooms)

    def _assert_valid_exits(self):
        # exits can't be on global edges

        for exit in self.exits:
            assert_valid_pos(exit)
            if exit[0] == 0:
                assert(exit[2] > 0)
            elif exit[0] == self.cols-1:
                assert(exit[1] < self.x_rooms-1)
            if exit[1] == 0:
                assert(exit[3] > 0)
            elif exit[1] == self.rows-1:
                assert(exit[3] < self.y_rooms-1)

    def _take_action(self, action):
        if action==0: # left
            if self.agent_loc in self.exits:
                self.agent_loc = (self.cols-1, self.agent_loc[1], self.agent_loc[2]-1, self.agent_loc[3] 
            elif self.agent_loc[0] > 0:
                self.agent_loc = (self.agent_loc[0]-1, self.agent_loc[1], self.agent_loc[2], self.agent_loc[3])
        elif action==1: # right
            if self.agent_loc in self.exits:
                self.agent_loc = (0, self.agent_loc[1], self.agent_loc[2]+1, self.agent_loc[3])
            elif self.agent_loc[0] < self.cols-1:
                self.agent_loc = (self.agent_loc[0]+1, self.agent_loc[1], self.agent_loc[2], self.agent_loc[3])
        elif action==2: # up
            if self.agent_loc in self.exits:
                self.agent_loc = (self.agent_loc[0], self.rows-1, self.agent_loc[2], self.agent_loc[3]-1)
            elif self.agent_loc[1] > 0:
                self.agent_loc = (self.agent_loc[0], self.agent_loc[1]-1, self.agent_loc[2], self.agent_loc[3])
        elif action==3: # down
            if self.action_loc in self.exits:
                self.agent_loc = (self.agent_loc[0], 0, self.agent_loc[2], self.agent_loc[3]+1)
            elif self.action_loc[1] < self.rows-1:
                self.agent_loc = (self.agent_loc[0], self.agent_loc[1]+1, self.agent_loc[2], self.agent_loc[3])
        else:
            raise ValueError("Incorrect Action")

        assertValidPos(self.agent_loc)

