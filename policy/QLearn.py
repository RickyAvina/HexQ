import random
from misc.utils import get_mdp, get_prim_action, exec_action
from render.gui import render_q_values


lr = 1e-5
exploration_steps = 2000
beta = 0.5
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

        # initialize pos in MER
        s = env.reset_in(mdp.primitive_states)

        for step in range(exploration_steps):
            # pick lower level action with eps_greedy
            sub_mdp = get_mdp(mdps, mdp.level-1, s)
            cum_reward = 0

            while sub_mdp != exit.mdp:
                a = get_action(sub_mdp, 0.2, mdp.policies[exit.mdp])  # right now hard coded probability, should decay
                s_p, r, d = exec_action(env, mdps, sub_mdp, s, a)
                assert not d, "should never really be done..."
                cum_reward += r
                s = s_p
                next_sub_mdp = get_mdp(mdps, mdp.level-1, s)
                update_q_vals(mdp.policies[exit.mdp], sub_mdp, a, next_sub_mdp, r, d)
                sub_mdp = next_sub_mdp

            render_q_values(mdp.policies)
            input("cumm reward: {}".format(cum_reward))
            #input("q values: {}".format(mdp.policies[exit.mdp]))

def max_q(exit_qvals, mdp):
    assert mdp in exit_qvals, "{} not in qvals[exit]!".format(mdp)
    return max(exit_qvals[mdp], key=lambda k: exit_qvals[mdp].get(k))


def get_action(mdp, p, exit_qvals):
    if random.random() > p:
        return random.choice(tuple(mdp.actions))
    else:
        # return choice with highest q val
        return max_q(exit_qvals, mdp)

def update_q_vals(exit_qvals, sub_mdp, a, next_sub_mdp, r, d):
    max_future_q = max_q(exit_qvals, next_sub_mdp)
    exit_qvals[sub_mdp][a] = (1-beta)*exit_qvals[sub_mdp][a] + (1-d)*beta*(r + gamma*max_future_q)
