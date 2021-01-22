import random
from misc.utils import get_mdp, get_prim_action, exec_action
from render.gui import render_q_values
import numpy as np


lr = 0.05
exploration_steps = 2000
gamma = 0.9


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
        mdp.policies[exit.mdp] = dict()

        for sub_mdp in mdp.mer:
            mdp.policies[exit.mdp][sub_mdp] = dict()
            for action in sub_mdp.actions:
                mdp.policies[exit.mdp][sub_mdp][action] = 0  # Q(s, a) = 0

        decay_count = 0
        #input("exit: {}".format(exit))
        for step in range(exploration_steps):

            # initialize pos in MER
            s = env.reset_in(mdp.primitive_states)
            sub_mdp = get_mdp(mdps, mdp.level-1, s)
            cum_reward = 0
            decay_count += 1
            history = []

            while sub_mdp != exit.mdp:
                try:
                    a = get_action(sub_mdp, decay_count/exploration_steps, mdp.policies[exit.mdp], False)  # right now hard coded probability, should decay

                    s_p, r, d = exec_action(env, mdps, sub_mdp, s, a)
                    next_sub_mdp = get_mdp(mdps, mdp.level-1, s_p)
                    if next_sub_mdp == exit.mdp:
                        r = 0
                    #input("{}->{}->{}: {}".format(s, a, s_p, r))
                    cum_reward += r
                    history.append((sub_mdp, a, next_sub_mdp, r, d))
                    #update_q_vals(mdp.policies[exit.mdp], sub_mdp, a, next_sub_mdp, r, d)
                    sub_mdp = next_sub_mdp
                    s = s_p
                except Exception as e:
                    raise ValueError("[exit: {}] {}".format(exit, str(e)))
            update_q_vals(mdp.policies[exit.mdp], history)
            #input("exit: {}\nqvals: {}".format(exit, sorted(mdp.policies[exit.mdp].items())))
        render_q_values(mdp.policies, exit)


def max_q(exit_qvals, mdp):
    if mdp not in exit_qvals:
        raise ValueError("{} not in qvals[exit]!".format(mdp))
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
        try:
            action = max_q(exit_qvals, mdp)
            return action
        except Exception as e:
            raise Exception("[get_action] " + str(e))

def update_q_vals(exit_qvals, history):
    local_history = history.copy()
    local_history.reverse()

    for s_mdp, a, sp_mdp, r, d in local_history:
        try:
            if d:
                exit_qvals[s_mdp][a] += lr*(r-exit_qvals[s_mdp][a])
            else:
                max_future_q = exit_qvals[sp_mdp][max_q(exit_qvals, sp_mdp)]
                exit_qvals[s_mdp][a] += lr*((r+max_future_q)-exit_qvals[s_mdp][a])
                #input("updating ({}, {} max: {})".format(s_mdp, a, max_future_q))
        except:
            raise Exception("[update_q_vals] " + str(e))
