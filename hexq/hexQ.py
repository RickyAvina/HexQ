import random
import numpy as np
from hexq.mdp import MDP, Exit
import policy.QLearn
from misc.utils import exec_action, get_mdp, fill_mdp_properties,\
                       aggregate_mdp_properties


class HexQ:
    def __init__(self, env, start, target):
        self.env = env
        self.start = start
        self.target = target
        self.exploration_steps = 10000
        self.mdps = {}  # level => [MDP,.. ]
        self.state_dim = len(start)
        self._init_mdps()

    def _init_mdps(self):
        MDP.env = self.env
        self.mdps[0] = []

    def find_freq(self):
        s = self.env.reset()
        seq = []
        states = set()

        for _ in range(self.exploration_steps):
            seq.append(s)
            states.add(s)
            a = np.random.randint(4)
            s_p, r, d, _ = self.env.step(a)
            s = s_p

        for state in states:
            primitive_mdp = MDP(level=0, state_var=state)
            primitive_mdp.actions = {0, 1, 2, 3}
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
        # find freq ordering of vars and initialize lowest level mdps
        freq = self.find_freq()

        # level zero (primitive actions)
        self.explore(level=0, exploration_steps=10000)

        # find Markov Equivalent Reigons
        self.create_sub_mdps(1)

        #for mdp in self.mdps[0]:
        #    input("mdp: {} \nexits: {} \nactions: {}".format(mdp, len(mdp.exits), mdp.actions))

        # level one (rooms)
        self.explore(level=1)

        for mdp in self.mdps[1]:
            input("mdp: {} \nexits: {} \nactions: {}".format(mdp, len(mdp.exits), mdp.actions))

    def train_sub_MDPs(self, mdps):
        for mdp in mdps:
            policy.QLearn.qlearn(env=self.env, mdps=self.mdps, mdp=mdp)

    def explore(self, level, exploration_steps=None):
        s = self.env.reset()

        for _ in range(self.exploration_steps if exploration_steps is None else exploration_steps):
            mdp = get_mdp(self.mdps, level, s)
            a = mdp.select_random_action()
            #if level > 0:
            #    input("mdp: {} action: {}".format(mdp, a))
            s_p, r = exec_action(self.env, self.mdps, mdp, s, a, 0)
            fill_mdp_properties(self.mdps, mdp, s, a, s_p)
            s = s_p
            #self.mdps[level].append(mdp)

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
                        new_exit = Exit(mdp, exit, neighbor)
                        exits.add(new_exit)
                        break
        return mer, exits
