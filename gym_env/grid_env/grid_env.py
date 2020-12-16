import gym
import numpy as np
from gym import spaces
import render.gui as gui


class GridEnv(gym.Env):
    def __init__(self, rows, cols, x_rooms, y_rooms, n_action, display=True):
        super(GridEnv, self).__init__()

        self.rows = rows
        self.cols = cols
        self.x_rooms = x_rooms
        self.y_rooms = y_rooms
        self.target_loc = (0, 0, 1, 0)
        # agent loc is (x, y, room_x, room_y)
        self.observation_space = spaces.Tuple((spaces.Discrete(cols),
                                               spaces.Discrete(rows),
                                               spaces.Discrete(x_rooms),
                                               spaces.Discrete(y_rooms)))
        self.action_space = spaces.Discrete(n_action)
        coor_exits = {(0, 14), (0, 22), (1, 10), (1, 22), (2, 2), (2, 14), (3, 2), (3, 10)}
        self.exits = {self.render_coor_to_loc(exit) for exit in coor_exits}

        self.display = display
        if display:
            gui.setup(600, 600, 5, 5, 2, 2, coor_exits)

    def render_coor_to_loc(self, render_coor):
        col = render_coor[1] % self.cols
        row = render_coor[1] // self.cols
        x_room = render_coor[0] % self.x_rooms
        y_room = render_coor[1] % self.y_rooms
        return (col, row, x_room, y_room)

    def loc_to_render_coor(self, loc):
        ''' 4 tuple -> 2 tuple '''
        room = loc[2]+loc[3]*self.x_rooms
        pos = loc[0]+loc[1]*self.cols
        return (room, pos)

    def step(self, action):
        self._take_action(action)
        
        if self.display:
            self.render()
        
        next_observation = self.agent_loc

        if np.array_equal(self.agent_loc, self.target_loc):
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        return (next_observation, reward, done, {})

    def reset(self):
        self._init_env()
        return self.agent_loc

    def render(self):
        gui.render(self.loc_to_render_coor(self.agent_loc))

    def _init_env(self):
        self.agent_loc = (0, 0, 0, 0)

    def _assert_valid_pos(self, loc, action=None):
        assert loc[0] >= 0 and loc[0] < self.cols, "{}, {}".format(self.agent_loc, action)
        assert loc[1] >= 0 and loc[1] < self.rows, "{}, {}".format(self.agent_loc, action) 
        assert loc[2] >= 0 and loc[2] < self.x_rooms, "{}, {}".format(self.agent_loc, action)
        assert loc[3] >= 0 and loc[3] < self.y_rooms, "{}, {}".format(self.agent_loc, action)

    def _assert_valid_exits(self):
        # exits can't be on global edges

        for exit in self.exits:
            self._assert_valid_pos(exit)
            if exit[0] == 0:
                assert(exit[2] > 0)
            elif exit[0] == self.cols-1:
                assert(exit[1] < self.x_rooms-1)
            if exit[1] == 0:
                assert(exit[3] > 0)
            elif exit[1] == self.rows-1:
                assert(exit[3] < self.y_rooms-1)

    def _take_action(self, action):
        if action == 0:  # left
            if self.agent_loc in self.exits:
                self.agent_loc = (self.cols-1, self.agent_loc[1], self.agent_loc[2]-1, self.agent_loc[3])
            elif self.agent_loc[0] > 0:
                self.agent_loc = (self.agent_loc[0]-1, self.agent_loc[1], self.agent_loc[2], self.agent_loc[3])
        elif action == 1:  # right
            if self.agent_loc in self.exits:
                self.agent_loc = (0, self.agent_loc[1], self.agent_loc[2]+1, self.agent_loc[3])
            elif self.agent_loc[0] < self.cols-1:
                self.agent_loc = (self.agent_loc[0]+1, self.agent_loc[1], self.agent_loc[2], self.agent_loc[3])
        elif action == 2:  # up
            if self.agent_loc in self.exits:
                self.agent_loc = (self.agent_loc[0], self.rows-1, self.agent_loc[2], self.agent_loc[3]-1)
            elif self.agent_loc[1] > 0:
                self.agent_loc = (self.agent_loc[0], self.agent_loc[1]-1, self.agent_loc[2], self.agent_loc[3])
        elif action == 3:  # down
            if self.agent_loc in self.exits:
                self.agent_loc = (self.agent_loc[0], 0, self.agent_loc[2], self.agent_loc[3]+1)
            elif self.agent_loc[1] < self.rows-1:
                self.agent_loc = (self.agent_loc[0], self.agent_loc[1]+1, self.agent_loc[2], self.agent_loc[3])
        else:
            raise ValueError("Incorrect Action")

        self._assert_valid_pos(self.agent_loc, action)

