import random
from hexq.mdp import get_mdp, exec_action
from tqdm import tqdm


def qlearn(env, mdps,  mdp, args):
    '''
    Implements Q-Learning to help agents find a policy to reach all
    of the exits in an mdp. Fills in an mdp's policy attribute which has the form:
        {exit_mdp:
            {mdp:
                {a: q, a': q'}
            mdp':
                {a: q, a': q'}},
        exit_mdp': {...}}

    Arguments:
        env: Gym env
        mdps: All MDPs in the hierarchy
        mdp: particular sub-mdp to train
        args: command-line args

    Returns:
        A dictionary mapping states to arrows for rendering of the form:
            {state: [0-4], ...}
            Where 0 = left, 1 = right, 2 = up, 3 = right
    '''

    for exit in mdp.exits:
        if args.verbose:
            print("finding policy for exit: {}".format(exit))

        mdp.policies[exit] = dict()

        for sub_mdp in mdp.mer:
            mdp.policies[exit][sub_mdp] = dict()
            for action in sub_mdp.actions:
                mdp.policies[exit][sub_mdp][action] = args.init_q  # Initialize Q-Vals

        decay_count = 0
        for step in tqdm(range(args.exploration_iterations), disable=(not args.verbose)):

            # initialize pos in MER
            s = env.reset_in(mdp.primitive_states)
            sub_mdp = get_mdp(mdps, mdp.level-1, s)
            cum_reward = 0
            decay_count += 1
            history = []
            steps_taken = 0

            # exit is {l1 mdp: action: {l0 mdp -> prim action -> l0 mdp}: l1 mdp}
            while sub_mdp != exit.action.next_mdp:
                if steps_taken > args.max_steps:
                    break

                a = get_action(sub_mdp, 0.5+decay_count/(2*args.exploration_iterations), mdp.policies[exit])
                s_p, r, d, info = exec_action(env, mdps, sub_mdp, s, a)
                next_sub_mdp = get_mdp(mdps, mdp.level-1, s_p)

                if next_sub_mdp not in mdp.mer:
                    if next_sub_mdp == exit.action.next_mdp:
                        r = 0
                        d = True
                    else:
                        r = -100
                        next_sub_mdp = sub_mdp
                        break

                cum_reward += r
                history.append((sub_mdp, a, next_sub_mdp, r, d))
                sub_mdp = next_sub_mdp
                s = s_p
                steps_taken += 1

            if len(history) > 0:
                update_q_vals(args, mdp.policies[exit], history)

    return get_arrows(mdp.policies)

def get_arrows(qvals):
    '''
    Get arrows asosciated with Q-Values

    Arguments:
        qvals: {exit: {mdp: {action: q, ...}}}

    Returns:
        {state: [0-4], ...}
        Where 0 = left, 1 = right, 2 = up, 3 = right
    '''

    res = {}
    for exit in qvals:
        res[exit.action.mdp.state_var] = {}
        for mdp in qvals[exit]:
            max_action = max(qvals[exit][mdp], key=lambda k: qvals[exit][mdp].get(k))
            res[exit.action.mdp.state_var][mdp.state_var] = max_action

    return res

def max_q(exit_qvals, mdp):
    return max(exit_qvals[mdp], key=lambda k: exit_qvals[mdp].get(k))

def get_non_exit_action(mdp, a):
    for exit in mdp.exits:
        if exit.mdp == mdp and a == exit.action:
            return get_non_exit_action(mdp, random.choice(tuple(mdp.actions)))

def get_action(mdp, p, exit_qvals):
    if random.random() > p:
        a = random.choice(tuple(mdp.actions))
        return a
    else:
        # return choice with highest q val
        action = max_q(exit_qvals, mdp)
        return action

def update_q_vals(args, exit_qvals, history):
    local_history = history.copy()
    local_history.reverse()

    for s_mdp, a, sp_mdp, r, d in local_history:
        max_future_q = 0
        best_action = None

        if not d:
            best_action = max_q(exit_qvals, sp_mdp)
            max_future_q = exit_qvals[sp_mdp][best_action]
        exit_qvals[s_mdp][a] = (1-args.lr)*exit_qvals[s_mdp][a] + args.lr*(r+args.gamma*max_future_q)
