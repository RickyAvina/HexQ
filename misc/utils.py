import git
import logging
import random
from hexq.mdp import Exit
import argparse


def random_exclude(exclude, low, high):
    randInt = random.randint(low, high)
    if randInt in exclude:
        return random_exclude(exclude, low, high)
    else:
        return randInt

def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args, path="."):
    log = {}

    set_logger(
        logger_name=args.log_name,
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    repo = git.Repo(path)
    log[args.log_name].info("Branch: {}".format(repo.active_branch))
    log[args.log_name].info("Commit: {}".format(repo.head.commit))

    return log


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def fill_mdp_properties(mdps, mdp, s, a, s_p):
    # fill in MDPs adjacency set

    if s != s_p:
        adj_mdp = get_mdp(mdps, mdp.level, s_p)
        mdp.adj.add(adj_mdp)
        adj_mdp.adj.add(mdp)

    # fill in MDPs transition count
    if (s, a) not in mdp.trans_count:
        mdp.trans_count[(s, a)] = {s_p: 1}
    elif s_p not in mdp.trans_count[(s, a)]:
        mdp.trans_count[(s, a)][s_p] = 1
    else:
        mdp.trans_count[(s, a)][s_p] += 1

    # fill in exit/entries if primitive
    if mdp.level == 0:
        if s != s_p:
            exit = Exit(mdp, a, adj_mdp)
            mdp.exits.add(exit)

def aggregate_mdp_properties(mdps):
    for mdp in mdps:
        mdp.trans_probs = count_to_probs(mdp)

def count_to_probs(mdp):
    trans_probs = {}

    for s_a in mdp.trans_count:
        if s_a not in trans_probs:
            trans_probs[s_a] = {}

            total_count = sum(mdp.trans_count[s_a].values())
            for s_p in mdp.trans_count[s_a]:
                trans_probs[s_a][s_p] = mdp.trans_count[s_a][s_p] / total_count
    return trans_probs

def get_mdp(mdps, level, s):
    sub_mdp = None

    for _sub_mdp in mdps[level]:
        if s in _sub_mdp.primitive_states:
            sub_mdp = _sub_mdp
            break

    assert sub_mdp is not None, "state {} does not belong to any sub MDP at level {}".format(s, level)
    return sub_mdp

def max_qval(qvals, state):
    '''
    qvals: {state: {action: val}}
    '''
    return max(qvals[state], key=lambda k: qvals[state].get(k))

def eps_greedy_action(state, qvals, sub_mdp, prob):
    '''
    qvals: {state: {action: val}...}
    '''
    if random.random() > prob:
        a = random.choice(tuple(sub_mdp.actions))
    else:
        a = max_qval(qvals, state)

    return a

def get_prim_action(mdps, mdp, s, a, p):
    qvals = mdp.policies[a]

    if mdp.level > 0:
        # look for sub_mdp
        sub_mdp = get_mdp(mdps=mdps, level=mdp.level, s=s)
        # look at q-values for actions at this level
        action = eps_greedy_action(s, qvals, sub_mdp, p)
        # changing probability < 1 would allow training at multiple levels
        return get_prim_action(mdps, sub_mdp, qvals, s, action, 1)  # deterministic lower level
    else:
        return a

def select_random_action(mdps, mdp, s):
    if mdp.level == 0:
        return random.choice(tuple(mdp.actions))
    else:
        sub_mdp = get_mdp(mdps=mdps, level=mdp.level-1, s=s)
        return random.choice(tuple(sub_mdp.actions))

def get_sub_action(mdps, mdp, s, a, p):
    if mdp.level == 0:
        return a, mdp
    else:
        qvals = mdp.policies[a]
        sub_mdp = get_mdp(mdps=mdps, level=mdp.level, s=s)
        action = eps_greedy_action(s, qvals, sub_mdp, p)
        return action, sub_mdp

def exec_action(env, mdps, mdp, state, action, s_ps=None, rs=None, ds=None):
    '''
    action is {0, 1, 2, 3} if primitive and (state, action) if not
    '''
    if rs is None:
        rs = 0

    if mdp.level == 0:
        s_p, r, d, _ = env.step(action)
        return s_p, r, d

    sub_mdp = get_mdp(mdps, mdp.level-1, state)
    exit_mdp = action.mdp  # mdp(l0) -> action -> next_mdp(l0)

    while sub_mdp != exit_mdp:
        s_p, r, d, _ = exec_action(action, mdps, sub_mdp, state, action, s_ps, rs, ds)
        rs += r
        s = s_p
        sub_mdp = get_mdp(mdps, sub_mdp.level, s)

    return s_ps, rs, ds
