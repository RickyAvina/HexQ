import random


class Exit(object):
    def __init__(self, mdp, action, next_mdp):
        self.mdp = mdp
        self.action = action
        self.next_mdp = next_mdp

    def __repr__(self):
        return "mdp: {} -> (action: {}) -> next_mdp: {}".format(self.mdp, self.action, self.next_mdp)

    def __eq__(self, other):
        return self.mdp == other.mdp and self.next_mdp == other.next_mdp

    def __hash__(self):
        return hash(self.mdp) + hash(self.next_mdp)


class MDP(object):
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

        self.mer = set()  # mdps one level under
        self.primitive_states = set()
        self.actions = set()   # R => exits (for primitives, key=value)

        self.trans_count = {}  # (s, a) -> {s_p: count, s_p': count'}
        # In future, could be a frozen dict {s: a: {s_p: count, s_p': count'}, a': {}}
        self.adj = set()
        self.trans_probs = None

        self.exit_pairs = set()  # {(s, s_p), ...}
        self.exits = set()  # {Exit}
        self.entries = set()  # {s', ...}

        self.policies = dict()  # {exit: Q-val dict}

    def __repr__(self):
        return "level {} var {}".format(self.level, self.state_var)

    def __eq__(self, other):
        return self.level == other.level and self.state_var == other.state_var

    def __hash__(self):
        return hash((self.level,) + self.state_var)

    def __lt__(self, other):
        return self.state_var < other.state_var

    def select_random_action(self):
        assert len(self.actions) > 0, "actions are empty"
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
