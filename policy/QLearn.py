import random
from hexq.mdp import get_mdp, exec_action
from tqdm import tqdm


def qlearn(env, mdps, mdp, args):
    '''
    Implements Q-Learning to help agents find a policy to reach all
    of the exits in an mdp. Fills in an mdp's policy attribute which has the form:
        {exit_mdp:
            {state:
                {a: q, a': q'}
            state':
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
            for primitive_state in sub_mdp.primitive_states:
                #input("prim state: {}".format(primitive_state))
                mdp.policies[exit][primitive_state] = dict()
                for action in sub_mdp.exits:
                    mdp.policies[exit][primitive_state][action] = args.init_q  # Initialize Q-Vals

        decay_count = 0
 
        for step in tqdm(range(args.exploration_iterations//mdp.level), disable=(not args.verbose)):
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
                a = get_action(sub_mdp, 0.5+decay_count/(2*args.exploration_iterations), mdp.policies[exit][s])
                s_p, r, d, info = exec_action(env, mdps, sub_mdp, s, a)
                next_sub_mdp = get_mdp(mdps, mdp.level-1, s_p)
                #input("{} --> {} --> {} r: {} d: {}".format(sub_mdp, "some action", next_sub_mdp, r, d))
                if next_sub_mdp not in mdp.mer:
                    if next_sub_mdp == exit.action.next_mdp:
                        r = 0
                        d = True
                    else:
                        r = -100
                        next_sub_mdp = sub_mdp
                        break

                cum_reward += r
                history.append((s, a, s_p, r, d))
                sub_mdp = next_sub_mdp
                s = s_p
                steps_taken += 1

            if len(history) > 0:
                update_q_vals(args, mdp.policies[exit], history)
    
    if mdp.level == 1:
        return get_arrows(mdp.policies)
    else:
        return [] 

def max_q(exit_qvals):
    return max(exit_qvals, key=lambda k: exit_qvals.get(k))

def get_action(mdp, p, exit_qvals):
    if random.random() > p:
        return mdp.select_random_action()
    else:
        # return choice with highest q val
        action = max_q(exit_qvals)
        return action

def update_q_vals(args, exit_qvals, history):
    local_history = history.copy()
    local_history.reverse()

    for s, a, s_p, r, d in local_history:
        max_future_q = 0
        best_action = None

        if not d:
            best_action = max_q(exit_qvals[s_p])
            max_future_q = exit_qvals[s_p][best_action]
        exit_qvals[s][a] = (1-args.lr)*exit_qvals[s][a] + args.lr*(r+args.gamma*max_future_q)

def get_arrows(qvals):
    '''
    Get arrows asosciated with Q-Values

    Arguments:
        qvals: {exit: {state: {action: q, ...}}}

    Returns:
        {state: [0-4], ...}
        Where 0 = left, 1 = right, 2 = up, 3 = right
    '''

    res = [] 
    for exit in qvals:
        arrow_dict = dict()
        arrow_dict['exit'] = exit.action.mdp.state_var
        arrow_dict['states'] = dict()
        for state in qvals[exit]:
            arrow_dict['states'][state] = max_q(qvals[exit][state])
        res.append(arrow_dict) 
    return res

