import random
from misc.utils import get_mdp, get_prim_action, exec_action
import numpy as np
from tqdm import tqdm


lr = 0.05
exploration_steps = 500
gamma = 0.9
init_val = 0


def qlearn(env, mdps,  mdp):
    '''
    env: env
    mdps: all mdps in hierarcy
    mdp: sub-mdp to train
    '''

    '''
    mdp.policies: {exit_mdp: {mdp:
                                {a: q, a': q'}
                              mdp':
                                {a: q, a': q'}},
                    exit_mdp': {...}}
    '''
    for exit in mdp.exits:
        print("finding policy for exit: {}".format(exit))
        mdp.policies[exit.mdp] = dict()

        for sub_mdp in mdp.mer:
            mdp.policies[exit.mdp][sub_mdp] = dict()
            for action in sub_mdp.actions:
                mdp.policies[exit.mdp][sub_mdp][action] = init_val  # Q(s, a) = 0

        decay_count = 0
        for step in tqdm(range(exploration_steps)):

            # initialize pos in MER
            s = env.reset_in(mdp.primitive_states)
            sub_mdp = get_mdp(mdps, mdp.level-1, s)
            cum_reward = 0
            decay_count += 1
            history = []

            while sub_mdp != exit.mdp:
                a = get_action(sub_mdp, decay_count/exploration_steps, mdp.policies[exit.mdp], False)

                s_p, r, d = exec_action(env, mdps, sub_mdp, s, a)
                next_sub_mdp = get_mdp(mdps, mdp.level-1, s_p)
                if next_sub_mdp == exit.mdp:
                    r = 0
                cum_reward += r
                history.append((sub_mdp, a, next_sub_mdp, r, d))
                sub_mdp = next_sub_mdp
                s = s_p
            update_q_vals(mdp.policies[exit.mdp], history)

        if env.gui:
            # convert to ({state var: arrow}
            env.gui.render_q_values(get_arrows(mdp.policies))
            print("hereee")

def get_arrows(qvals):
    res = {}
    for exit_mdp in qvals:
        res[exit_mdp.state_var] = {}
        for mdp in qvals[exit_mdp]:
            max_action = max(qvals[exit_mdp][mdp], key=lambda k: qvals[exit_mdp][mdp].get(k))
            res[exit_mdp.state_var][mdp.state_var] = max_action
    return res

def max_q(exit_qvals, mdp):
    #add_qval(exit_qvals, mdp)
    return max(exit_qvals[mdp], key=lambda k: exit_qvals[mdp].get(k))

def get_non_exit_action(mdp, a):
    for exit in mdp.exits:
        if exit.mdp == mdp and a == exit.action:
            return get_non_exit_action(mdp, random.choice(tuple(mdp.actions)))

def get_action(mdp, p, exit_qvals, exit_actions_allowed=True):
    if random.random() > p:
        a = random.choice(tuple(mdp.actions))
        return a
    else:
        # return choice with highest q val
        action = max_q(exit_qvals, mdp)
        return action

def add_qval(exit_qvals, mdp, a=None):
    if mdp not in exit_qvals:
        exit_qvals[mdp] = dict()
        for action in mdp.actions:
            exit_qvals[mdp][action] = init_val
    if a:
        exit_qvals[mdp][a] = init_val

def update_q_vals(exit_qvals, history):
    local_history = history.copy()
    local_history.reverse()

    for s_mdp, a, sp_mdp, r, d in local_history:
        #add_qval(exit_qvals, s_mdp, a)
        #add_qval(exit_qvals, sp_mdp)
        if d:
            exit_qvals[s_mdp][a] += lr*(r-exit_qvals[s_mdp][a])
        else:
            max_future_q = exit_qvals[sp_mdp][max_q(exit_qvals, sp_mdp)]
            exit_qvals[s_mdp][a] += lr*((r+max_future_q)-exit_qvals[s_mdp][a])
            #input("updating ({}, {} max: {})".format(s_mdp, a, max_future_q))
