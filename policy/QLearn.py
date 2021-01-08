import random

lr = 1e-5
exploration_steps = 2000


# {exit: {(s,a): q, (s, a'): q'}
# {exit: {s:  {a: q, a': q},
#        {s': {a: q, a': q'}}

def qlearn(env, mdp, sub_actions):
    '''
    env: env
    mdp: sub-mdp to train
    sub_actions: actions of level e-1
    '''

    for exit in mdp.exits:
        mdp.policies[exit] = dict()  # empty Q-Val dict
        qvals = mdp.policies[exit]

        for state in mdp.mer:
            qvals[state] = dict()
            for action in sub_actions:
                qvals[state][action] = 0  # initialize q vals

        s = env.reset()

        for step in range(exploration_steps):
            if random.random() > 0.8:  # epsilon-greedy
                action = random.choice(tuple(sub_actions))
            else:  # select action with the highest q-value
                action = max(qvals[state], key=lambda k: qvals[state].get(k))

            input("chose action: {}".format(action))
