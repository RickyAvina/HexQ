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

    def __init__(self, level, state_var):
        self.state_var = state_var
        self.level = level

        self.mer = set()  # mdps one level under
        self.primitive_states = set()
        self.actions = set()   # R => exits (for primitives, key=value)
        self.exit_actions = {}

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

    def count_to_probs(self):
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


""" MDP Utility Methods """
def fill_mdp_properties(mdps, mdp, s, a, s_p):
    # fill in MDPs adjacency set

    if s != s_p:
        adj_mdp = get_mdp(mdps, mdp.level, s_p)
        mdp.adj.add(adj_mdp)
        adj_mdp.adj.add(mdp)

    # fill in MDPs transition count
    if (s, a) not in mdp.trans_count:
        mdp.trans_count[(s, a)] = {s_p: 1}
    elif s_p not in mdp.trans_count[(s, a)]:
        mdp.trans_count[(s, a)][s_p] = 1
    else:
        mdp.trans_count[(s, a)][s_p] += 1

    # fill in exit/entries if primitive
    if mdp.level == 0:
        if s != s_p:
            exit = Exit(mdp, a, adj_mdp)
            mdp.exits.add(exit)

def aggregate_mdp_properties(mdps):
    for mdp in mdps:
        mdp.trans_probs = mdp.count_to_probs()

def get_mdp(mdps, level, s):
    sub_mdp = None

    for _sub_mdp in mdps[level]:
        if s in _sub_mdp.primitive_states:
            sub_mdp = _sub_mdp
            break

    assert sub_mdp is not None, "state {} does not belong to any sub MDP at level {}".format(s, level)
    return sub_mdp

def exec_action(env, mdps, mdp, state, action, rs=None):
    '''
    action is {0, 1, 2, 3} if primitive and (state, action) if not
    '''
    if rs is None:
        rs = 0

    if mdp.level == 0:
        s_p, r, d, info  = env.step(action)
        return s_p, r, d, info

    sub_mdp = get_mdp(mdps, mdp.level-1, state)
    exit_mdp = action.mdp  # mdp(l0) -> action -> next_mdp(l0)

    while sub_mdp != exit_mdp:
        s_p, r, d, info = exec_action(action, mdps, sub_mdp, state, action, rs)
        rs += r
        sub_mdp = get_mdp(mdps, sub_mdp.level, s_p)

    return s_p, rs, d, info
