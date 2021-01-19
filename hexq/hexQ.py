import random
import policy.QLearn
import numpy as np
from hexq.mdp import MDP, Exit
from misc.utils import exec_action, get_mdp, fill_mdp_properties, aggregate_mdp_properties
from render.gui import change_title


class HexQ(object):
    def __init__(self, env, state_dim, start, target):
        self.env = env
        self.start = start
        self.target = target  # TODO This varaible seems to be not used
        self.exploration_steps = 10000  # TODO In the algorithm argparaser!
                                        # TODO In HexQ page 81, they used 2000
        self.mdps = {}  # level => [MDP,.. ]
        self.state_dim = state_dim

        self._init_mdps()

    def _init_mdps(self):
        MDP.env = self.env
        self.mdps[0] = []

    def find_freq(self):
        """Agent randomly explores env randomly for period of time.
        After exploration, agent sorts variables based on their frequency of change.

        Returns:
            sorted_order (np.ndarray): Sorted order of variables

        References:
            Page 81 in the HexQ paper
        """
        state = self.env.reset()
        seq = [state]

        for _ in range(self.exploration_steps):
            action = np.random.randint(4)  # TODO Use env.action_space instead of hard-coding
            next_state, reward, done, _ = self.env.step(action)
            seq.append(next_state)
            if done:
                state = self.env.reset()
            else:
                state = next_state

        states = set(seq)
        for state in states:
            primitive_mdp = MDP(level=0, state_var=state)
            primitive_mdp.actions = {0, 1, 2, 3}  # TODO Use env.action_space instead of hard-coding
            primitive_mdp.mer = {state}
            primitive_mdp.primitive_states = {state}
            self.mdps[0].append(primitive_mdp)

        freq = [set() for _ in range(self.state_dim)]
        for state in seq:
            for i in range(self.state_dim):
                freq[i].add(state[i])
        sorted_order = np.argsort([len(arr) for arr in freq])
        return sorted_order

    def alg(self):
        # Find freq ordering of vars and initialize lowest level mdps
        # TODO Variable seq is not used
        # TODO that maybe sorting order needs to be opposite such that
        # most frequent variable is in the first order
        freq = self.find_freq()

        # TODO I guess line 2, 3, 4 in Table 5.1 are missing?! Woot woot?

        # level zero (primitive actions)
        # TODO Looking at the paper, it seems like they refer to the most bottom level to be
        # level 1 instead of level 0
        #change_title("explore")

        self.explore(level=0, exploration_steps=2000)
        assert len(self.mdps[0]) == 77, "there should be {} mdps instead of {}".format(77, len(self.mdps[0]))
        # find Markov Equivalent Reigons
        self.create_sub_mdps(1)

        ''' train sub_mdps '''
        #change_title("train sub_mdps")
        self.train_sub_mdps(self.mdps[1])

        # level one (rooms)
        #self.explore(level=1)

    def train_sub_mdps(self, mdps):
        for mdp in mdps:
            policy.QLearn.qlearn(env=self.env, mdps=self.mdps, mdp=mdp)

    def explore(self, level, exploration_steps=None):
        s = self.env.reset()

        for _ in range(self.exploration_steps if exploration_steps is None else exploration_steps):
            mdp = get_mdp(self.mdps, level, s)
            a = mdp.select_random_action()
            s_p, r, d = exec_action(self.env, self.mdps, mdp, s, a)
            fill_mdp_properties(self.mdps, mdp, s, a, s_p)
            if d:
                s = self.env.reset()
            else:
                s = s_p

        aggregate_mdp_properties(self.mdps[level])

    def create_sub_mdps(self, level):
        mdps_copy = set(self.mdps[level-1].copy())
        mdps = []

        while len(mdps_copy) > 0:
            curr_mdp = random.choice(tuple(mdps_copy))
            mer, exits = self.bfs(mdps_copy, curr_mdp, level)

            state_var = next(iter(mer)).state_var[1:]
            mdp = MDP(level=level, state_var=state_var)
            mdp.mer = mer
            mdp.exits = exits
            mdp.actions = exits
            for _mdp in mer:
                mdp.primitive_states.update(_mdp.primitive_states)
            mdps.append(mdp)

        self.mdps[level] = mdps

    def bfs(self, mdp_list, mdp, level, mer=None, exits=None):
        if mer is None:
            mer = set()
        if exits is None:
            exits = set()

        if mdp in mdp_list:
            mdp_list.remove(mdp)
        mer.add(mdp)

        for neighbor in mdp.adj:
            if neighbor.state_var[level:] == mdp.state_var[level:]:
                if neighbor in mdp_list:
                    self.bfs(mdp_list=mdp_list, mdp=neighbor, level=level, mer=mer, exits=exits)
            else:
                for exit in mdp.exits:
                    if neighbor == exit.next_mdp:  # found exit
                        new_exit = Exit(mdp, exit.action, neighbor)
                        exits.add(new_exit)
                        break
        return mer, exits
