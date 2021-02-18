import random
import policy.QLearn
import numpy as np
from hexq.mdp import MDP, Exit, get_mdp, fill_mdp_properties, aggregate_mdp_properties, exec_action
import pickle
import os


class HexQ(object):
    def __init__(self, env, args):
        self.env = env
        self.start = args.start
        self.target = args.target
        self.args = args
        self.mdps = {}  # level => [MDP,.. ]
        self.state_dim = args.state_dim
        if self.args.test:
            self.test_policy()
        else:
            self.alg()

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

        for _ in range(self.args.exploration_iterations*10):
            action = np.random.randint(4)  # TODO Use env.action_space instead of hard-coding
            next_state, reward, done, info = self.env.step(action)
            seq.append(next_state)
            if done:
                state = self.env.reset()
            else:
                state = next_state

        states = set(seq)
        for state in states:
            primitive_mdp = MDP(level=0, state_var=state)
            primitive_mdp.exits = {0, 1, 2, 3}  # TODO Use env.action_space instead of hard-coding
            primitive_mdp.mer = frozenset({state})
            primitive_mdp.primitive_states = {state}
            self.mdps[0].add(primitive_mdp)

        freq = [set() for _ in range(self.state_dim)]
        for state in seq:
            for i in range(self.state_dim):
                freq[i].add(state[i])
        sorted_order = np.argsort([len(arr) for arr in freq])
        return sorted_order

    def test_policy(self):
        assert os.path.exists(self.args.binary_file), "file {} doesn't exist!".format(self.args.binary_file)
        mdp_binary = open(self.args.binary_file)
        self.mdps = pickle.load(mdp_binary)
        input(self.mdps)
        ''' 
        mdp1 = Hashable_MDP(0, (0, 1))
        mdp2 = Hashable_MDP(1, (0, ))
        mdp1.adj.add(mdp2)
        mdp2.adj.add(mdp1)
        
        u_mdp1 = Unhashable_MDP.from_hashable(mdp1)
        u_mdp2 = Unhashable_MDP.from_hashable(mdp2)
        input(u_mdp1)
        input(u_mdp2)
        # convert Hashable to unhashable MDPs
        p_on = open("test.pickle", "wb")
        pickle.dump({u_mdp1, u_mdp2}, p_on)
        p_on.close()
        
        p_off = open("test.pickle", "rb")
        pickled1, pickled2 = pickle.load(p_off)
        r_mdp1 = Hashable_MDP.from_unhashable(pickled1)
        r_mdp2 = Hashable_MDP.from_unhashable(pickled2)
        input(r_mdp1)
        input(r_mdp2)
        # try to fix this problem
        pickle_dict = open(self.args.binary_file, 'rb')
        self.mdps = pickle.load(pickle_dict)
        input(self.mdps)
        '''

    def alg(self):
        # Find freq ordering of vars and initialize lowest level mdps
        # TODO that maybe sorting order needs to be opposite such that
        # most frequent variable is in the first order
        self.mdps[0] = set()

        freq = self.find_freq()

        # TODO I guess line 2, 3, 4 in Table 5.1 are missing?! Woot woot?

        # level zero (primitive actions)
        # TODO Looking at the paper, it seems like they refer to the most bottom level to be
        # level 1 instead of level 0

        self.explore(level=0, exploration_iterations=self.args.exploration_iterations)

        # find Markov Equivalent Reigons
        self.create_sub_mdps(1)

        ''' train sub_mdps '''
        self.train_sub_mdps(self.mdps[1])

        # level one (rooms)
        self.explore(level=1)

        # find Markov Equivelant Regions (which should be one)
        self.create_sub_mdps(2)

        self.train_sub_mdps(self.mdps[2])

        with open(self.args.binary_file, 'wb') as handle:
            pickle.dump(self.mdps, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        if self.args.verbose:
            print("Finished pickling MDPs, saved at {}!".format(self.args.binary_file))

    def train_sub_mdps(self, mdps):
        arrow_list = []

        for mdp in mdps:
            arrows = policy.QLearn.qlearn(env=self.env, mdps=self.mdps, mdp=mdp, args=self.args)
            arrow_list.append(arrows)

        if self.env.gui:
            self.env.gui.render_q_values(arrow_list)

    def explore(self, level, exploration_iterations=None):
        s = self.env.reset()

        for _ in range(self.args.exploration_iterations if exploration_iterations is None else exploration_iterations):
            mdp = get_mdp(self.mdps, level, s)
            a = mdp.select_random_action()
            s_p, r, d, info = exec_action(self.env, self.mdps, mdp, s, a)
            next_mdp = get_mdp(self.mdps, level, s_p)
            mdp.adj.add(next_mdp)
            mdp.fill_properties(a, next_mdp, r, d)

            if d:
                s = self.env.reset()
            else:
                s = s_p

        #aggregate_mdp_properties(self.mdps[level])

    def create_sub_mdps(self, level):
        mdps_copy = set(self.mdps[level-1].copy())
        mdps = set()
        upper_level_exits = {}

        while len(mdps_copy) > 0:
            curr_mdp = random.choice(tuple(mdps_copy))
            mer, exits = set(), set()
            self.dfs(mdps_copy, curr_mdp, level, mer, exits)
            state_var = next(iter(mer)).state_var[1:]
            mdp = MDP(level=level, state_var=state_var)
            mdp.mer = frozenset(mer)
            upper_level_exits[mdp] = exits
            for _mdp in mer:
                mdp.primitive_states.update(_mdp.primitive_states)
            mdps.add(mdp)

        self.mdps[level] = mdps

        # Add MDP Exits/Actions
        for mdp in self.mdps[level]:
            mdp.exits = set()
            for s_mdp, exit, n_mdp in upper_level_exits[mdp]:
                neighbor_mdp = n_mdp.get_upper_mdp(self.mdps) 
                mdp.exits.add(Exit(mdp, Exit(s_mdp, exit, n_mdp), neighbor_mdp))

    def is_exit(self, mdp, neighbor, level):
        # an exit is a transition that
        # 1: causes the MDP to terminate
        # 2: causes context to change
        # 3: has a non-stationary trans function
        # 4: has a non-stationary reward function
        # 5: transitions between MERs

        for action in mdp.trans_history:
            if neighbor in mdp.trans_history[action]['states']:
                # Condition 2
                if neighbor.state_var[level:] != mdp.state_var[level:]:
                    return True, action, 2

                # Condition 1/5
                if True in mdp.trans_history[action]['dones']:
                    return True, action, 1

                # Condition 4
                if mdp.level < 1:
                    if len(set(mdp.trans_history[action]['rewards'])) > 1:
                        return True, action, 4

        return False, None, None

    def dfs(self, mdp_list, mdp, level, mer, exits):
        if mdp in mdp_list:
            mdp_list.remove(mdp)
        mer.add(mdp)
        for neighbor in mdp.adj:
            found_exit, action, condition = self.is_exit(mdp, neighbor, level)
            if found_exit:
                exits.add((mdp, action, neighbor))
            else:
                if neighbor in mdp_list:
                    self.dfs(mdp_list, neighbor, level, mer, exits)
