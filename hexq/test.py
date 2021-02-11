import pickle

# Make PICKLE-EABLE MDP and non-pickleable MDP
class MDP(object):
    def __init__(self, level, state_var):
        self.state_var = state_var
        self.level = level
        self.adj = frozenset()
        self.mer = frozenset()
        ...

    def __repr__(self):
         return "level {} var {} mer {}".format(self.level, self.state_var, self.mer)

    # def __eq__(self, other):
    #      if isinstance(other, MDP):
    #          return (self.level == other.level and self.state_var == other.state_var and self.mer == other.mer)
    #      else:
    #          return False
    #
    # def __hash__(self):
    #     return hash(self.__repr__())

    # def __lt__(self, other):
    #     return self.state_var < other.state_var


def main():
    mdp1 = MDP(0, (0, 1))
    mdp2 = MDP(0, (0, 2))
    mdp1.adj = frozenset({mdp2})
    mdp2.adj = frozenset({mdp1})
    mdps = {mdp1, mdp2}
    dumped = pickle.dumps(mdps)
    reloaded = pickle.loads(dumped)
    print(reloaded)
    print(mdp1 == mdp2)
    print(mdp1 == MDP(0, (0, 2)))
    # p_on = open("test.pickle", "wb")
    # pickle.dump(mdps, p_on)
    # p_on.close()
    #
    #
    # p_off = open("test.pickle", "rb")
    # emp = pickle.load(p_off)
    # print(emp)


main()