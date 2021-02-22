import random
import sys
sys.path.append('.')


class Exit(object):
    def __init__(self, mdp, action, next_mdp):
        self.mdp = mdp
        self.action = action
        self.next_mdp = next_mdp

    def __repr__(self):
        return "mdp: {} -> (action: {}) -> next_mdp: {}".format(self.mdp.simple_rep(), self.action, self.next_mdp.simple_rep())

    '''
    def __eq__(self, other):
        return self.mdp == other.mdp and self.next_mdp == other.next_mdp

    def __hash__(self):
        return hash(self.mdp) + hash(self.next_mdp)
    '''

class MDP(object):
    '''
    states are a set
    actions are a tuple
    '''
    def __init__(self, level, state_var):
        self.state_var = state_var
        self.level = level

        self.mer = frozenset()  # mdps one level under
        self.primitive_states = set()

        self.trans_history = {}  # (s, a) -> {'states':  {s_p: count, s_p': count'}}
        #                                     'rewards': {r, r', ...}
        #                                     'dones':   {d, d', ...}

        # In future, could be a frozen dict {s: a: {s_p: count, s_p': count'}, a': {}}
        self.adj = set()
        self.trans_probs = None

        self.exit_pairs = set()  # {(s, s_p), ...}
        self.exits = set()  # {Exit}
        self.entries = set()  # {s', ...}

        self.policies = dict()  # {exit: Q-val dict}

    def __repr__(self):
        return "level {} var {}".format(self.level, self.state_var)

    def simple_rep(self):
        return "level {} var {}".format(self.level, self.state_var)

    '''
    def __eq__(self, other):
        if isinstance(other, MDP):
            return (self.level == other.level and self.state_var == other.state_var and self.mer==other.mer)
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__())
    '''

    def __lt__(self, other):
        if self.level < other.level:
            return True
        elif self.level > other.level:
            return False
        else:
            return self.state_var < other.state_var

    def select_random_action(self):
        #if self.level == 0:
        #    return random.choice(tuple(self.actions))
        assert len(self.exits) > 0, "actions are empty, mdp: {}".format(self)
        return random.choice(tuple(self.exits))

    def fill_properties(self, a, next_mdp, r, d):
        # fill adjacency set
        if self != next_mdp:
            self.adj.add(next_mdp)
            #next_mdp.adj.add(self)
        
        # fill in trans history
        if a not in self.trans_history:
            self.trans_history[a] = {'states': {next_mdp: 1}}
        elif next_mdp not in self.trans_history[a]['states']:
            self.trans_history[a]['states'][next_mdp] = 1
        else:
            self.trans_history[a]['states'][next_mdp] += 1
        
        if 'rewards' not in self.trans_history[a]:
            self.trans_history[a]['rewards'] = []
        self.trans_history[a]['rewards'].append(r)

        if 'dones' not in self.trans_history[a]:
            self.trans_history[a]['dones'] = []
        self.trans_history[a]['dones'].append(d)

    def get_upper_mdp(self, mdps):
        # get mdp at level l+1
        for _mdp in mdps[self.level+1]:
            if self in _mdp.mer:
                return _mdp

        raise ValueError("MDP {} is not a sub mdp of an mdp at level {}".format(self, self.level+1))
        return None

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
def fill_mdp_properties(mdps, mdp, s, a, s_p, r, d):
    # fill in MDPs adjacency set

    if s != s_p:
        adj_mdp = get_mdp(mdps, mdp.level, s_p)
        mdp.adj.add(adj_mdp)
        adj_mdp.adj.add(mdp)

    # USE MDP INSTEAD OF S

    # fill in MDPs transition count
    if (s, a) not in mdp.trans_history:
        mdp.trans_history[(s, a)] = {'states': {s_p: 1}}
    elif s_p not in mdp.trans_history[(s, a)]['states']:
        mdp.trans_history[(s, a)]['states'][s_p] = 1
    else:
        mdp.trans_history[(s, a)]['states'][s_p] += 1

    if 'rewards' not in mdp.trans_history[(s, a)]:
        mdp.trans_history[(s, a)]['rewards'] = []
    mdp.trans_history[(s, a)]['rewards'].append(r)

    if 'dones' not in mdp.trans_history[(s, a)]:
        mdp.trans_history[(s, a)]['dones'] = []
    mdp.trans_history[(s, a)]['dones'].append(d)

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

def exec_action(env, mdps, mdp, state, exit, render=False):
    '''
    action is {0, 1, 2, 3} if primitive and (state, action) if not
    '''

    if mdp.level == 0:
        s_p, r, d, info = env.step(exit)  # at primitive level, exit is action
        if render:
            env.gui.render_agent(s_p)
        return s_p, r, d, info

    s_p, d, info = state, False, dict()
    cumm_reward = 0

    while mdp != exit.next_mdp:
        sub_mdp = get_mdp(mdps, mdp.level-1, state)
        # get best action according to q-values
        best_action = max_q(mdp.policies[exit][state])
        s_p, r, d, info = exec_action(env, mdps, sub_mdp, state, best_action, render)
        cumm_reward += r
        state = s_p
        mdp = get_mdp(mdps, mdp.level, state)

    return s_p, cumm_reward, d, info


def max_q(exit_qvals):
    return max(exit_qvals, key=lambda k: exit_qvals.get(k))
