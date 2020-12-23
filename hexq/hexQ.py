import random
import numpy as np


class HexQ:
    def __init__(self, start, env, state_dim):
        self.env = env
        self.start = start
        self.exploration_steps = 1000
        self.freq_discovered = False
        self.mdps = {}
        self.state_dim = state_dim
        self._init_mdps()

    def _init_mdps(self):
        for i in range(self.state_dim):
            self.mdps[i] = MDP()

        # initialize base mdp with primitive actions
        self.mdps[0].set_actions((0, 1, 2, 3))

    def find_freq(self):
        s = self.env.reset()
        seq = []

        for _ in range(self.exploration_steps):
            seq.append(s)
            a = np.random.randint(4)
            s_p, r, d, _ = self.env.step(a)
            s = s_p

        freq = [set() for _ in range(self.state_dim)]

        for state in seq:
            for i in range(self.state_dim):
                freq[i].add(state[i])

        sorted_order = np.argsort([len(arr) for arr in freq])
        return sorted_order

    def alg(self):
        #freq = self.find_freq()
    
        import sys
        for e in range(self.state_dim):
            transition_probs, exits, entries = self.explore(self.mdps[e], e)
            sys.exit()

    def explore(self, mdp, e):
        s = self.env.reset()

        for _ in range(self.exploration_steps):
            # select actions from action set
            a = mdp.select_random_action()
            s_p, r, d, _ = self.env.step(a)
            print("{}->{}->{}".format(s, a, s_p))
            mdp.add_trans(s, a, s_p)
            s = s_p
        
        #print("mdp counts: {}".format(mdp.trans_count))
        #trans = mdp._count_to_probs()
        #print("\nmdp trans: {}".format(trans.items()))
        return None, None, None


class MDP:
    '''
    states are a set
    actions are a tuple
    '''

    def __init__(self):
        self.states = set()
        self.actions = ()
        self.trans_count = {}  # (s, a) -> {s_p: count, s_p': count'}
        self.trans_probs = None

    def add_state(self, state):
        self.states.add(state)

    def set_actions(self, actions):
        self.actions = actions

    def add_action(self, action):
        self.actions = self.actions + (action, )

    def select_random_action(self):
        return random.choice(self.actions)

    def add_trans(self, s, a, s_p):
        #print("receiving s: {}, a: {}, s_p: {}".format(s, a, s_p))

        if (s, a) not in self.trans_count:
            self.trans_count[(s, a)] = {s_p: 1}
        elif s_p not in self.trans_count[(s, a)]:
            self.trans_count[(s, a)][s_p] = 1
        else:
            self.trans_count[(s, a)][s_p] += 1

    def _count_to_probs(self):
        trans_probs = {}

        for s_a in self.trans_count:
            if s_a not in trans_probs:
                trans_probs[s_a] = {}

            total_count = sum(self.trans_count[s_a].values)
            for s_p in self.trans_count[s_a]:
                trans_probs[s_a][s_p] = self.trans_count[s_a][s_p] / total_count

        return trans_probs
