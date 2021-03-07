"""
The HexQ object is the functioning code of the HexQ algorithm.
"""

import random
import policy.QLearn
import numpy as np
from hexq.mdp import MDP, Exit, get_mdp, aggregate_mdp_properties, exec_action
import pickle
import os


class HexQ(object):
    def __init__(self, env, args):
        self.env = env
        self.start = args.start
        self.target = args.target
        self.args = args
        self.mdps = {}                      # level => {MDP,.. }
        if self.args.test:
            self.test_policy()
        else:
            self.alg()

    def find_freq(self):
        """Agent randomly explores env randomly for period of time.
        After exploration, agent sorts variables based on their frequency of change.

        Returns:
            state_dim    int          The number of dimensions of the state          
            sorted_order (np.ndarray) Sorted order of variables

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
        return sorted_order[::-1]

    def test_policy(self):
        assert os.path.exists(self.args.binary_file), "file {} doesn't exist!".format(self.args.binary_file)
        pickle_dict = open(self.args.binary_file, 'rb')
        mdps = pickle.load(pickle_dict)
        
        while True:
            # select greedy actions and reset if necessary
            s = self.env.reset()
            mdp = get_mdp(mdps, self.args.state_dim, s)
            
            # select best top level policy
            max_q_val = self.args.init_q
            best_exit = None
            for exit in mdp.policies:
                max_q = max(mdp.policies[exit][s].values())
                if max_q > max_q_val:
                    max_q_val = max_q
                    best_exit = exit
            s_p, r, d, info = exec_action(self.env, mdps, mdp, s, best_exit, True)
            
    def alg(self):
        # Find freq ordering of vars and initialize lowest level mdps
        self.mdps[0] = set()
        self.freq = list(self.find_freq())

        for level in range(self.args.state_dim):  # TODO remove state_dim
            self.explore(level=level, exploration_iterations=self.args.exploration_iterations)
            self.create_sub_mdps(level+1)
            self.train_sub_mdps(self.mdps[level+1])
        
        with open(self.args.binary_file, 'wb') as handle:
            pickle.dump(self.mdps, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        pickle_dict = open(self.args.binary_file, 'rb')
        r_mdps = pickle.load(pickle_dict)
        
        if self.args.verbose:
            print("Finished pickling MDPs, saved at {}!".format(self.args.binary_file))

    def train_sub_mdps(self, mdps):
        arrow_list = []

        for mdp in mdps:
            arrows = policy.QLearn.qlearn(env=self.env, mdps=self.mdps, mdp=mdp, args=self.args)
            arrow_list.extend(arrows)
        if self.env.gui:
            self.env.gui.render_q_values(arrow_list)

    def explore(self, level, exploration_iterations=None):
        s = self.env.reset()

        for _ in range(self.args.exploration_iterations if exploration_iterations is None else exploration_iterations):
            mdp = get_mdp(self.mdps, level, s)
            a = mdp.select_random_action()
            s_p, r, d, info = exec_action(self.env, self.mdps, mdp, s, a)
            next_mdp = get_mdp(self.mdps, level, s_p)
            mdp.fill_properties(a, next_mdp, r, d)

            if d:
                s = self.env.reset()
            else:
                s = s_p

        # Because grid world is deterministic, this line is commented
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
                if mdp.sv(self.freq)[level:] != neighbor.sv(self.freq)[level:]:
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
