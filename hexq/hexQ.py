"""
The HexQ object is the functioning code of the HexQ algorithm.
Based from Bernhard Hengst's paper: http://www.cse.unsw.edu.au/~bernhardh/BHPhD2003.pdf
"""

import random
import policy.QLearn
import numpy as np
from hexq.mdp import MDP, Exit, get_mdp, aggregate_mdp_properties, exec_action
import pickle
import os


class HexQ(object):
    def __init__(self, env, args, log, tb_writer):
        self.env = env
        self.start = args.start
        self.target = args.target
        self.args = args
        self.log = log
        self.tb_writer = tb_writer
        self.mdps = {}  # level => {MDP,.. }
        if self.args.test:
            self.test_policy()
        else:
            self.alg()

    def test_policy(self):
        """
        Test your behavior by deserializing the MDP dictionary and executing greedy
        actions at the highest level (solving the overall MDP), click to advance to the next
        primitive action.
        """

        assert os.path.exists(self.args.binary_file), "file {} doesn't exist!".format(self.args.binary_file)
        pickle_dict = open(self.args.binary_file, 'rb')
        mdps = pickle.load(pickle_dict)

        while True:
            # select greedy actions and reset if necessary
            s = self.env.reset()
            mdp = get_mdp(mdps, len(s), s)

            # select best top level policy
            max_q_val = self.args.init_q
            best_exit = None
            for exit in mdp.policies:
                max_q = max(mdp.policies[exit][s].values())
                if max_q > max_q_val:
                    max_q_val = max_q
                    best_exit = exit
            s_p, r, d, info = exec_action(self.env, mdps, mdp, s, best_exit, True)

    def find_freq(self):
        """Agent randomly explores env randomly for period of time.
        After exploration, agent sorts variables based on their frequency of change.

        Returns:
            state_dim    (int)          The number of dimensions of the state
            sorted_order (np.ndarray) Sorted order of variables, if our state is (pos, room, floor)
                                      [1, 0, 2] means room changes the most frequently and floor change
                                      the least frequently

        References:
            Page 81 in the HexQ paper
        """
        state = self.env.reset()
        state_dim = len(state)
        seq = [state]

        for _ in range(self.args.exploration_iterations*10):
            action = np.random.randint(self.env.action_space.n)
            next_state, reward, done, info = self.env.step(action)
            seq.append(next_state)
            if done:
                state = self.env.reset()
            else:
                state = next_state

        # Create a primitive MDP for every unique state explored
        states = set(seq)
        for state in states:
            primitive_mdp = MDP(level=0, state_var=state)
            primitive_mdp.exits = {x for x in range(self.env.action_space.n)}
            primitive_mdp.mer = frozenset({state})
            primitive_mdp.primitive_states = {state}
            self.mdps[0].add(primitive_mdp)

        freq = [{'sv': i, 'last': None, 'changes': 0} for i in range(state_dim)]
        for state in seq:
            for i in range(state_dim):
                if freq[i]['last'] is None:
                    freq[i]['last'] = state[i]
                else:
                    if state[i] != freq[i]['last']:
                        freq[i]['changes'] += 1
                        freq[i]['last'] = state[i]

        sorted_freq = sorted(freq, key=lambda x: x['changes'], reverse=True)
        return [d['sv'] for d in sorted_freq], state_dim

    def alg(self):
        """
        The entry point for the HexQ algorithm. Constructs a task hierarchy by creating a dictionary
        {level: {MDPs}}, where each MDP has actions it can take and a policy assigning Q-Values to those
        actions.
        """

        # Find freq ordering of vars and initialize lowest level mdps
        self.mdps[0] = set()
        self.freq, self.state_dim = self.find_freq()

        # Build the task hierarchy
        for level in range(self.state_dim):
            # Randomly execute actions from level and fill MDP properties (trans prob, adj, etc)
            self.explore(level=level, exploration_iterations=self.args.exploration_iterations)
            # Using MDP properties, find exits, MERs, and form MDPs at level+1
            self.create_sub_mdps(level+1)
            ''' 
            for mdp in self.mdps[level+1]:
                if len(mdp.exits) == 0:
                    input("mdp: {} mer: {}".format(mdp, mdp.mer))
                    for s_mdp in mdp.mer:
                        input("sub mdp: {} actions: {}".format(s_mdp, s_mdp.exits))
            '''
            # Train a policy to reach every exit in the MDPs at level+1
            self.train_sub_mdps(self.mdps[level+1])

        # Serialize the MDPs
        with open(self.args.binary_file, 'wb') as handle:
            pickle.dump(self.mdps, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        self.log[self.args.log_name].info("Finished pickling MDPs, saved at {}!\n".format(self.args.binary_file))

    def explore(self, level, exploration_iterations=None):
        """
        Explore transitions at a particular level

        Arguments:
        level                  (Int) level to explore (0 is lowest)
        exploration_iterations (Int) the amount of transitions to sample
        """

        s = self.env.reset()

        for _ in range(self.args.exploration_iterations if exploration_iterations is None else exploration_iterations):
            # Given a state, get the MDP at the specified level
            mdp = get_mdp(self.mdps, level, s)
            a = mdp.select_random_action()
            # Execute a hierarchical action
            s_p, r, d, info = exec_action(self.env, self.mdps, mdp, s, a)
            next_mdp = get_mdp(self.mdps, level, s_p)
            # Store information about transitions for finding exits
            mdp.fill_properties(a, next_mdp, r, d)

            if d:
                s = self.env.reset()
            else:
                s = s_p

        # Because grid world is deterministic, this line is commented
        # Convert transition counts to probabilities
        #aggregate_mdp_properties(self.mdps[level])

    def create_sub_mdps(self, level):
        """
        Find MERs and exits and form MDPs at the next level

        Arguments:
        level (Int) level of created MDPs
        """

        mdps_copy = set(self.mdps[level-1].copy())
        mdps = set()
        upper_level_exits = {}

        # Full depth-first search to group MDPs into MERs
        while len(mdps_copy) > 0:
            curr_mdp = random.choice(tuple(mdps_copy))
            mer, exits = set(), set()
            # Group curr_mdp with neighbors to form a MER and find exits
            self.dfs(mdps_copy, curr_mdp, level, mer, exits)

            # Choose a state var that is representative of the new MER
            state_var = next(iter(mer)).state_var[1:]
            # Create a new upper level MDP and set its properties
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
            # Generate new exits (mdp at level, Exit at level-1, target mdp at level)
            for s_mdp, exit, n_mdp in upper_level_exits[mdp]:
                neighbor_mdp = n_mdp.get_upper_mdp(self.mdps)
                mdp.exits.add(Exit(mdp, Exit(s_mdp, exit, n_mdp), neighbor_mdp))

    def is_exit(self, mdp, neighbor, level):
        """
        Determine if there is an exit between mdp and neighbor. Not all of the conditions below
        are considered.

        Arguments:
        mdp      (MDP)  a mdp at level `level`
        neighbor (MDP)  a different mdp at level `level`
        level    (Int)  the level of mdp and neighbor

        Returns:
            _      (bool)   True if an exit exits
            action (Exit)   Exit at level-1 if an exit exits, None otherwise
            _      (Int)    Condition that triggered an exit

        An exit is a transition that
        1: causes the MDP to terminate
        2: causes context to change
        3: has a non-stationary trans function
        4: has a non-stationary reward function
        5: transitions between MERs

        References:
        Page 84
        """

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
        """
        Using a Depth-First Search, create a MER, which is a set of MDPs that can reach eachother without Exit actions
        Populates `mer` and `exits`

        Arguments:
        mdp_list (set{MDP})  set of MDPs to consider
        mdp      (MDP)       MDP to consider
        level    (Int)       level of MDP
        mer      (set{MDP})  Markov Equivalent Reigion
        exits    (set{Exit}) exits associated with mer
        """

        # Only consider an MDP once in DFS
        if mdp in mdp_list:
            mdp_list.remove(mdp)
        mer.add(mdp)

        for neighbor in mdp.adj:
            # Determine if neighbor is an exit or part of the MER
            found_exit, action, condition = self.is_exit(mdp, neighbor, level)
            if found_exit:
                exits.add((mdp, action, neighbor))
            else:
                if neighbor in mdp_list:
                    # Recursively consider other MDPs in `mdp_list`
                    self.dfs(mdp_list, neighbor, level, mer, exits)

    def train_sub_mdps(self, mdps):
        """
        Train a policy to reach every exit for each mdp in mdps. Render policy if applicable

        Arguments:
        mdps (set{MDP}) a set of MDPs to find exits for
        """

        arrow_list = []

        for mdp in mdps:
            # generate an arrow for every state in an MDP
            arrows = policy.QLearn.qlearn(env=self.env, mdps=self.mdps, mdp=mdp,
                                          args=self.args, log=self.log, tb_writer=self.tb_writer)
            arrow_list.extend(arrows)
        if self.env.gui:
            self.env.gui.render_q_values(arrow_list)
