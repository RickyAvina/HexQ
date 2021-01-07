import random


class MDP:
    '''
    states are a set
    actions are a tuple
    '''

    # states are shared across all MDPs
    states = set()
    env = None

    def __init__(self, level, state_var):
        self.state_var = state_var
        self.level = level

        self.mer = set()
        self.actions = set()   # R => exits (for primitives, key=value)

        self.trans_count = {}  # (s, a) -> {s_p: count, s_p': count'}
        # In future, could be a frozen dict {s: a: {s_p: count, s_p': count'}, a': {}}
        self.adj = {}
        self.trans_probs = None

        self.exit_pairs = set()  # {(s, s_p), ...}
        self.exits = set()  # {(s, a), ...}
        self.entries = set()  # {s', ...}

    def __repr__(self):
        return "(MDP) level {} var {} actions {}".format(self.level, self.state_var, self.actions)

    def select_random_action(self):
        return random.choice(tuple(self.actions))

    def add_trans(self, s, a, s_p):
        # transitions are deterministic after primitive level
        # states are also known, however, it is kept for convenience

        # record states
        self.states.add(s)

        # fill in adj dictionary
        if s not in self.adj:
            self.adj[s] = set()
        self.adj[s].add(s_p)

        # fill in transition probabilities and entries/exits
        if s != self.target:
            if (s, a) not in self.trans_count:
                self.trans_count[(s, a)] = {s_p: 1}
            elif s_p not in self.trans_count[(s, a)]:
                self.trans_count[(s, a)][s_p] = 1
            else:
                self.trans_count[(s, a)][s_p] += 1

            if s[1] != s_p[1]:  # exit/entry
                self.exit_pairs.add((s, s_p))
                self.exits.add((s, a))
                self.entries.add(s_p)

    def _count_to_probs(self):
        trans_probs = {}

        for s_a in self.trans_count:
            if s_a not in trans_probs:
                trans_probs[s_a] = {}

            total_count = sum(self.trans_count[s_a].values())
            for s_p in self.trans_count[s_a]:
                trans_probs[s_a][s_p] = self.trans_count[s_a][s_p] / total_count

        return trans_probs

    def find_MERs(self):
        ''' MERs are just states with deterministic intra-region transitions '''
        states = self.states.copy()
        mers = []

        while len(states) > 0:
            s = random.choice(tuple(states))
            mer = {s}
            self.bfs(states, s, mer)
            mers.append(mer)

        return mers

    def bfs(self, states, s, mer):
        if s in states:
            states.remove(s)
            mer.add(s)
            for neighbor in self.adj[s]:
                if neighbor in states and (s, neighbor) not in self.exit_pairs:
                    self.bfs(states=states, s=neighbor, mer=mer)
