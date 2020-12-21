import numpy as np


class HexQ:
    def __init__(self, start):
        self.seq = []
        self.exploration_steps = 1000
        self.freq_discovered = False
        self.curr_steps_explored = 0

    def explore(self, s):
        self.seq.append(s)
        self.curr_steps_explored += 1
        if self.curr_steps_explored > self.exploration_steps:
            freq = []  # [{}, {}]
            for _ in range(len(self.seq[0])):
                freq.append(set())

            for state in self.seq:
                for i in range(len(state)):
                    freq[i].add(state[i])
            
            print("freq: {}".format(freq))
            
            self.freq_discovered = True
            discovered_freq = np.argsort([len(arr) for arr in freq])
            print("discovered freq: {}".format(discovered_freq))
