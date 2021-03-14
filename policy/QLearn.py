import random
from hexq.mdp import get_mdp, exec_action
from tqdm import tqdm


def qlearn(env, mdps, mdp, args, log, tb_writer):
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
        env      : Gym env
        mdps     : All MDPs in the hierarchy
        mdp      : particular sub-mdp to train
        args     : command-line args
        log      : debug log
        tb_writer: Tensorboard graph

    Returns:
        A dictionary mapping states to arrows for rendering of the form:
            {state: [0-4], ...}
            Where 0 = left, 1 = right, 2 = up, 3 = right
    '''

    for exit in mdp.exits:
        if args.verbose:
            log[args.log_name].info("finding policy for exit: {}".format(exit))

        # Derive a policy for each exit
        mdp.policies[exit] = dict()
        for sub_mdp in mdp.mer:
            # Store a Q-Value for every state in the MER
            for primitive_state in sub_mdp.primitive_states:
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
            epsilon = args.epsilon

            # pick actions from level-1 until mdp transitions reaches the exit
            while sub_mdp != exit.action.next_mdp:
                if steps_taken > args.max_steps:
                    break

                # pick an action from sub_mdp using an eps greedy strategy
                a = get_action(sub_mdp, epsilon, mdp.policies[exit][s])
                epsilon = max(args.min_epsilon, epsilon*args.epsilon_decay)

                # execute hierarchical action
                s_p, r, d, info = exec_action(env, mdps, sub_mdp, s, a)
                next_sub_mdp = get_mdp(mdps, mdp.level-1, s_p)
                if next_sub_mdp not in mdp.mer:
                    # reward agent for reaching the exit
                    if next_sub_mdp == exit.action.next_mdp:
                        r = 0
                        d = True
                    else:
                        # penalize agent for exiting to the wrong MDP
                        r = -100
                        next_sub_mdp = sub_mdp
                        break

                cum_reward += r
                history.append((s, a, s_p, r, d))
                sub_mdp = next_sub_mdp
                s = s_p
                steps_taken += 1

            if len(history) > 0:
                tb_writer.add_scalar('cum_reward_{}'.format(name_replace(exit)), cum_reward, step)
                update_q_vals(args, mdp.policies[exit], history, tb_writer)

    if mdp.level == 1:
        return get_arrows(mdp.policies)
    else:
        return []

def name_replace(exit):
    s = str(exit)
    for r in ((" ", ""), ("(", "_"), (")", "_"), (">", "-"), (":", "-"), (",", "_")):
        s = s.replace(*r)
    return s

def max_q(exit_qvals):
    """
    Get the best action from Q-Values

    Arguments:
    exit_qvals {action: q}

    Returns
    action (Exit) the best action to take
    """
    return max(exit_qvals, key=lambda k: exit_qvals.get(k))

def get_action(mdp, p, exit_qvals):
    """
    Select a eps-greedy action

    Arguments
    p          (Float [0-1]) the probability of selecting a random action
    exit_qvals (dict {a: p}) q-values for a state and exit

    Returns
    action (Exit) best action or random action
    """

    if random.random() < p:
        return mdp.select_random_action()
    else:
        # return choice with highest q val
        action = max_q(exit_qvals)
        return action

def update_q_vals(args, exit_qvals, history, tb_writer):
    """
    Update Q-values for `exit_qvals`

    Arguments
    exit_qvals (dict {a: p})               q-values for a state and exit
    history    (Array)                     array of (s, a, next state, r, d) tuples
    tb_writer  (Tensorboard.SummaryWriter) tensorboard log
    """

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
        qvals (dictionary) {exit: {state: {action: q, ...}}}

    Returns:
        res  (dictionary) {state: [0-4], ...}
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

