import random
from misc.utils import get_mdp, get_prim_action


lr = 1e-5
exploration_steps = 2000
beta = 0.5
gamma = 0.9

# {exit: {(s,a): q, (s, a'): q'}
# {exit: {s:  {a: q, a': q},
#        {s': {a: q, a': q'}}


def qlearn(env, mdps,  mdp):
    '''
    env: env
    mdps: all mdps in hierarcy
    mdp: sub-mdp to train
    '''

    for exit in mdp.exits:
        mdp.policies[exit] = dict()

        # find sub-mdps that overlap with mdp mer
        for sub_mdp in mdps[mdp.level-1]:
            if sub_mdp.mer.issubset(mdp.mer):
                mdp.policies[exit][sub_mdp.state_var] = dict()
                for action in sub_mdp.actions:
                    mdp.policies[exit][sub_mdp.state_var][action] = 0   # Q(s, a) = 0

        # initialize pos in MER
        s = env.reset_in(mdp.mer)
        for step in range(exploration_steps):
            while s != exit[0]:
                a = get_prim_action(mdps=mdps, mdp=mdp, qvals=mdp.policies[exit],
                                    s=s, a=exit, p=0.2)
                s_p, r, d, _ = env.step(a)
                input("{}->{}->{}".format(s, a, s_p))
                s = s_p

            # update step
            #max_q = max_qval(qvals, s_p)
            #qvals[s][a] = (1-beta)*qvals[s][a] + beta*(r+gamma*max_q)
