import random
import numpy as np
from hexq.mdp import MDP
import policy.QLearn


class HexQ:
    def __init__(self, env, start, target):
        self.env = env
        self.start = start
        self.target = target
        self.exploration_steps = 10000
        self.freq_discovered = False
        self.mdps = {}  # level => [MDP,.. ]
        self.state_dim = len(start)
        self._init_mdps()

    def _init_mdps(self):
        MDP.env = self.env
        root_mdp = MDP(0, None)

        # initialize base mdp with primitive actions
        root_mdp.actions = {0, 1, 2, 3}
        self.mdps[0] = root_mdp

    def find_freq(self):
        s = self.env.reset()
        seq = []

        for _ in range(self.exploration_steps):
            seq.append(s)
            a = np.random.randint(4)
            s_p, r, d, _ = self.env.step(action=a, reset=False)
            s = s_p

        freq = [set() for _ in range(self.state_dim)]

        for state in seq:
            for i in range(self.state_dim):
                freq[i].add(state[i])

        sorted_order = np.argsort([len(arr) for arr in freq])
        return sorted_order

    def alg(self):
        #freq = self.find_freq()

        # level zero (primitive actions)
        mdp = self.mdps[0]
        transition_probs, exits, entries = self.explore(level=0)

        mers = self.find_MERs(mdp)

        # from MERS, create sub-mpds
        sub_mdps = self.create_sub_MDPs(mdp=mdp, mers=mers, level=1)
        self.mdps[1] = sub_mdps

        print(self.mdps)
        # train each sub-mdp
        self.train_sub_MDPs(self.mdps[1])

        
        #transition_probs, exits, entries = self.explore(level=1)
        # train each sub-mdp, should yield a policy to reach each resp exit
        # an additional sub-mdp should be trained to navigate from exits to goal

        # create abstract actions and initation set for each sub-mdp
        # this means that each sub-mdp should be numbered according to their room

        # find value function for s, a pairs in sub_mdp
        # Actions of next level = exits of current level

    def train_sub_MDPs(self, mdps):
        for mdp in mdps:
            policy.QLearn.qlearn(self.env, mdp, {0, 1, 2, 3})

    def explore(self, level):
        s = self.env.reset()

        for _ in range(self.exploration_steps):
            # select actions from action set
            if level == 0:
                mdp = self.mdps[0]
                a = mdp.select_random_action()
                s_p, r, d, _ = self.primitive_trans(mdp, s, a)
            else:
                # multiple steps
                # figure out which MDP you're in
                sub_mdp = None

                for mdp in self.mdps[level]:
                    if s in mdp.mer:
                        sub_mdp = mdp
                        break

                assert sub_mdp is not None, "state {} does not belong to any sub MDP".format(s)
                # pick action
                a = sub_mdp.select_random_action()
                
                s_p, r, d, _ = self.multistep_trans(mdp=sub_mdp, s=s, a=a)

            s = s_p

        trans = mdp._count_to_probs()
        return trans, mdp.exits, mdp.entries

    def primitive_trans(self, mdp, s, a):
        s_p, r, d, _ = self.env.step(action=a)

        # add to entire state space
        MDP.states.add(s)
        mdp.mer.add(s)

        # fill in root adj to find MERs
        if s not in mdp.adj:
            mdp.adj[s] = set()
        mdp.adj[s].add(s_p)

        # fill in transition probs
        if (s, a) not in mdp.trans_count:
            mdp.trans_count[(s, a)] = {s_p: 1}
        elif s_p not in mdp.trans_count[(s, a)]:
            mdp.trans_count[(s, a)][s_p] = 1
        else:
            mdp.trans_count[(s, a)][s_p] += 1

        # fill in exit/entries
        if s[1:] != s_p[1:]:
            mdp.exit_pairs.add((s, s_p))
            mdp.exits.add((s, a))
            mdp.entries.add(s_p)

        return s_p, r, d, None

    def multistep_trans(self, mdp, s, a):
        # multiple steps
        # This MDP should select random acttions which are policies to exits of
        # previous level. There should be an initation set (states where you can
        # execute a certain policy/pick actions from.
        # An action should take you to the exit, and then you take the exit action

        # sample random action from actions
        
        # take action a which means
        '''
        while (s != a):
            # selet actions from level-1 (in this case primitives)

        '''
        
        s_p = a
        return s_p, 0, False, None

    def find_MERs(self, mdp):
        ''' MERs are just states with deterministic intra-region transitions '''

        states = mdp.mer.copy()  # for level 0, this is all the states
        mers = []

        while len(states) > 0:
            s = random.choice(tuple(states))
            mer = self.bfs(mdp, states, s)
            mers.append(mer)
        return mers

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
