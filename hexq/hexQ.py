import random
import numpy as np
from hexq.mdp import MDP
import policy.QLearn
from misc.utils import exec_action, get_mdp, fill_mdp_properties, aggregate_mdp_properties 


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
            self.mdps[0].append(primitive_mdp)

        freq = [set() for _ in range(self.state_dim)]

        for state in seq:
            for i in range(self.state_dim):
                freq[i].add(state[i])
        sorted_order = np.argsort([len(arr) for arr in freq])
        return sorted_order

    def alg(self):
        freq = self.find_freq()

        # level zero (primitive actions)
        self.explore(level=0)

        # find Markov Equivelant Reigons
        mers = self.find_MERs(1)

        # from MERS, create sub-mpds
#        sub_mdps = self.create_sub_MDPs(mdp=mdp, mers=mers, level=1)
#        self.mdps[1] = sub_mdps

        # train each sub-mdp
#        self.train_sub_MDPs(self.mdps[1])

        
        #transition_probs, exits, entries = self.explore(level=1)
        # train each sub-mdp, should yield a policy to reach each resp exit
        # an additional sub-mdp should be trained to navigate from exits to goal

        # create abstract actions and initation set for each sub-mdp
        # this means that each sub-mdp should be numbered according to their room

        # find value function for s, a pairs in sub_mdp
        # Actions of next level = exits of current level

    def train_sub_MDPs(self, mdps):
        for mdp in mdps:
            policy.QLearn.qlearn(env=self.env, mdps=self.mdps,
                    mdp=mdp)

    def explore(self, level):
        s = self.env.reset()

        for _ in range(self.exploration_steps):
            mdp = get_mdp(self.mdps, level, s)
            a = mdp.select_random_action()
            s_p, r = exec_action(self.env, self.mdps, mdp, s, a, 0)
            fill_mdp_properties(self.mdps, mdp, s, a, s_p)
            s = s_p
            self.mdps[level].append(mdp)

        aggregate_mdp_properties(self.mdps[level])

    def find_MERs(self, level):
        mdps_copy = set(self.mdps[level-1].copy())
        mers = []

        while len(mdps_copy) > 0:
            curr_mdp = random.choice(tuple(mdps_copy))
            mer = self.bfs(mdps_copy, curr_mdp, level)
            mers.append(mer)
        return mers

    def bfs(self, mdp_list, mdp, level, mer=None):
        if mer is None:
            mer = set()
       
        #input("mdp: {}".format(mdp))
        if mdp in mdp_list:
            mdp_list.remove(mdp)
        mer.add(mdp)

        # input("{} mdps,  mer: {}".format(len(mdp_list), mer))
        
        for neighbor in mdp.adj:
            #input("curr: {} neighbor: {}".format(mdp.state_var, neighbor.state_var))
            
            if neighbor in mdp_list and neighbor.state_var[level:] == mdp.state_var[level:]:
                self.bfs(mdp_list=mdp_list, mdp=neighbor, level=level, mer=mer)

        return mer
    '''
    def bfs(self, mdp, states, s, mer=None):
        if mer is None:
            mer = set()

        if s in states:
            states.remove(s)
        mer.add(s)

        for neighbor in mdp.adj[s]:
            if neighbor in states and (s, neighbor) not in mdp.exit_pairs:
                self.bfs(mdp=mdp, states=states, s=neighbor, mer=mer)

        return mer
    '''

    def create_sub_MDPs(self, mdp, mers, level):
        # Three architectures:
        # 1) One MDP per level, initiation set and different policies per exit
        # 2) One MDP per MER/exit pair. This is the one used in the HEXQ alg.
        # 3) One MDP per MER (with various exits), each action will be a diff exit policy
        sub_mdps = []
        # Picking arch 3
        for mer in mers:
            random_state = random.choice(tuple(mer))
            sub_mdp = MDP(level=level, state_var=random_state[level])
            sub_mdp.mer = mer

            # find exits that correspond to this MER
            for state, action in mdp.exits:
                if state in mer:
                    sub_mdp.exits.add((state, action))
                    sub_mdp.actions.add(state)

            sub_mdps.append(sub_mdp)
        return sub_mdps
