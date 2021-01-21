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

            try:
                while sub_mdp != exit.mdp:
                    a = get_action(sub_mdp, 0.2, mdp.policies[exit.mdp], False)  # right now hard coded probability, should decay
                    s_p, r, d = exec_action(env, mdps, sub_mdp, s, a)
                    cum_reward += r
                    s = s_p
                    next_sub_mdp = get_mdp(mdps, mdp.level-1, s)
                    update_q_vals(mdp.policies[exit.mdp], sub_mdp, a, next_sub_mdp, r, d)
                    sub_mdp = next_sub_mdp

                render_q_values(mdp.policies, exit)
                input("rendered.")
            except Exception as e:
                input("e: {} exit: {}".format(e, exit))
            #input("q values: {}".format(mdp.policies[exit.mdp]))


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

def update_q_vals(exit_qvals, sub_mdp, a, next_sub_mdp, r, d):
    try:
        if not d:
            max_future_q = max_q(exit_qvals, next_sub_mdp)
            exit_qvals[sub_mdp][a] = (1-beta)*exit_qvals[sub_mdp][a] + beta*(r + gamma*max_future_q)
        else:
            exit_qvals[sub_mdp][a] = (1-beta)*exit_qvals[sub_mdp][a]
    except Exception as e:
        raise Exception("[update_q_vals] " + str(e))
