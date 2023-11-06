import numpy as np


class Automaton1():
    def __init__(self):
        self.max_state = 3

    def onehot2index(self, onehot):
        index = np.where(onehot == 1)[0][0]
        return index

    def index2onehot(self, index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    def reset(self):
        index = 0
        self.q = self.index2onehot(index, self.max_state)
        return self.q

    def step(self, label):
        index = self.onehot2index(self.q)
        label = self.onehot2index(label)
        if index == 0: #Initial state
            if label in [0, 7]:
                index = 2
            elif label == 5:
                index = 1 #accepting state
            else:
                index = 0
        elif index == 1:
            if label in [0, 7]:
                index = 2 #rejecting state
            else:
                index = 1
        else:
            index = 2
        self.q = self.index2onehot(index, self.max_state)
        return self.q

# The subsystem for the room temp. The way
class Subsystem1():
    def __init__(self):
        self.Te = -1
        self.Th = 50
        self.alpha = 0.45
        self.beta = 0.045
        self.gamma = 0.09


        self.Ul = 0  # Lower bound for input
        self.Uu = 1  # Upper bounds for input

        # == == == == == == == == == == == == = Marginals for Uniform Grid == == == == == == == == == == == ===

        self.delta_u = 0.2  # Input set discretization parameter

        self.n_u = int(np.fix((self.Uu - self.Ul) / self.delta_u))  # Number of partitions for input

    def onehot2index(self, onehot):
        index = np.where(onehot == 1)[0][0]
        return index

    def index2onehot(self, index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    def index2input(self, index):
        state = self.Ul + self.delta_u*index + self.delta_u/2
        return state

    def input2index(self, state):
        index = np.fix((state - self.Ul) / self.delta_u)
        return index

    def getActionSet(self, state):
        if state[0] > 22:
            action_set = np.arange(1)
        elif state[0] < 16:
            action_set = np.array([4])
        elif state[0] >= 16 and state[0] < 19:
            action_set = np.array([2, 3, 4])
        else:
            action_set = np.arange(self.n_u) #need to be fixed at the moment we consider all of actions we need to do action masking
        return action_set

    def reset(self):
        self.s = [np.random.rand()*2+18]#+0.249995+0.000006*np.random.rand()
        return self.s, self.getActionSet(self.s)

    def step(self, action, temp_of_others):
        action = self.index2input(action)
        Ti = self.s[0]
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        Tip1 = temp_of_others
        Te = self.Te
        Th = self.Th
        x_next = Ti + alpha*(2*Tip1-2*Ti)+beta*(Te-Ti)+gamma*(Th-Ti)*action
        self.s = [x_next]
        return self.s, self.getActionSet(self.s)

class Env():
    def __init__(self):
        self.automaton1 = Automaton1()
        self.automaton2 = Automaton1()
        #self.automaton3 = Automaton1()
        #self.automaton4 = Automaton1()
        self.subsystem1 = Subsystem1()
        self.subsystem2 = Subsystem1()
        #self.subsystem3 = Subsystem1()
        #self.subsystem4 = Subsystem1()
        self.max_action_set1 = None
        self.max_action_set2 = None
        #self.max_action_set3 = None
        #self.max_action_set4 = None
        self.T = 40  # Horizon
        self.n_l = 8
        self.n_s = 1
        self.n_a_1 = 8
        self.n_a_2 = 5
        self.n_a_3 = 17
        self.n_s_1 = 7
        self.n_s_2 = 15
        self.n_s_3 = 20
    def onehot2index(self, onehot):
        index = np.where(onehot == 1)[0][0]
        return index

    def index2onehot(self, index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    def allLabels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7]

    def getLabel(self, state):
        # if state[1]:
        #    return 5
        # x = self.onehot2index(state)
        if state <= 17:
            l = 0
            return self.index2onehot(l, self.n_l)
        elif state > 17 and state <= 18:
            l = 1
            return self.index2onehot(l, self.n_l)
        elif state > 18 and state <= 19:
            l = 2
            return self.index2onehot(l, self.n_l)
        elif state > 19 and state <= 20:
            l = 3
            return self.index2onehot(l, self.n_l)
        elif state > 20 and state <= 21:
            l = 4
            return self.index2onehot(l, self.n_l)
        elif state > 21 and state <= 22:
            l = 5
            return self.index2onehot(l, self.n_l)
        elif state > 22 and state <= 23:
            l = 6
            return self.index2onehot(l, self.n_l)
        else:
            l = 7
            return self.index2onehot(l, self.n_l)

    def getStates(self, label):
        if label == 0:
            a = [0, 1, 2]
            return [x for x in a]
        elif label == 1:
            a = [2, 3, 4]
            return [x for x in a]
        elif label == 2:
            a = [4, 5, 6]
            return [x for x in a]
        elif label == 3:
            a = [6, 7, 8]
            return [x for x in a]
        elif label == 4:
            a = [8, 9, 10]
            return [x for x in a]
        elif label == 5:
            a = [10, 11, 12]
            return [x for x in a]
        elif label == 6:
            a = [12, 13, 14]
            return [x for x in a]
        elif label == 7:
            a = [14, 15, 16]
            return [x for x in a]

    def reset(self):
        self.t = 0
        self.player = 1  # 0 for maximizing player, 1 for minimizing
        sub_state1, max_action_set1 = self.subsystem1.reset()
        sub_state2, max_action_set2 = self.subsystem2.reset()
        #sub_state3, max_action_set3 = self.subsystem3.reset()
        #sub_state4, max_action_set4 = self.subsystem4.reset()
        self.max_action_set1 = max_action_set1
        self.max_action_set2 = max_action_set2
        #self.max_action_set3 = max_action_set3
        #self.max_action_set4 = max_action_set4
        auto_state1 = self.automaton1.reset()
        auto_state2 = self.automaton2.reset()
        #auto_state3 = self.automaton3.reset()
        #auto_state4 = self.automaton4.reset()
        self.state1 = np.concatenate((sub_state1, auto_state1), axis=0)
        self.state2 = np.concatenate((sub_state2, auto_state2), axis=0)
        #self.state3 = np.concatenate((sub_state3, auto_state3), axis=0)
        #self.state4 = np.concatenate((sub_state4, auto_state4), axis=0)
        self.state = np.concatenate([[self.state1], [self.state2]])#, [self.state3], [self.state4]])
        self.action_set = [[max_action_set1], [max_action_set2]]#, [max_action_set3], [max_action_set4]]
        #action_set = self.allLabels()  # Labels
        return self.state, self.action_set, self.player

    def disturbance_other_players(self, label):
        states = self.getStates(label)
        w = np.random.rand() * 0.0001
        return w


    def step(self, action):
        action_set = None
        acc1 = None
        acc2 = None
        #acc3 = None
        #acc4 = None

        self.t = self.t + 1

        temp_1 = self.state1[0:1]
        auto_state1 = self.automaton1.step(self.getLabel(temp_1))
        temp_2 = self.state2[0:1]
        auto_state2 = self.automaton2.step(self.getLabel(temp_2))
        #temp_3 = self.state3[0:1]
        #auto_state3 = self.automaton3.step(self.getLabel(temp_3))
        #temp_4 = self.state4[0:1]
        #auto_state4 = self.automaton4.step(self.getLabel(temp_4))

        acc1 = self.onehot2index(auto_state1) == 1
        acc2 = self.onehot2index(auto_state2) == 1
        #acc3 = self.onehot2index(auto_state3) == 1
        #acc4 = self.onehot2index(auto_state4) == 1

        action1 = action[0][0]
        action2 = action[0][1]
        #action3 = action[0][2]
        #action4 = action[0][3]

        sub_state1, max_action_set1 = \
            self.subsystem1.step(action1, temp_2)
        sub_state2, max_action_set2 = \
            self.subsystem2.step(action2, temp_1)
        #sub_state3, max_action_set3 = \
        #    self.subsystem3.step(action3)
        #sub_state4, max_action_set4 = \
        #    self.subsystem4.step(action4)
        #wd = self.disturbance_other_players(0)
        #sub_state1[0] = sub_state1[0] + self.disturbance_other_players(0)
        #sub_state2[0] = sub_state2[0] + self.disturbance_other_players(0)
        #sub_state3[0] = sub_state3[0] + self.disturbance_other_players(0)
        #sub_state4[0] = sub_state4[0] + self.disturbance_other_players(0)

        self.max_action_set1 = max_action_set1
        self.max_action_set2 = max_action_set2
        #self.max_action_set3 = max_action_set3
        #self.max_action_set4 = max_action_set4
        self.state1 = np.concatenate((sub_state1[0], auto_state1), axis=0)
        self.state2 = np.concatenate((sub_state2[0], auto_state2), axis=0)
        #self.state3 = np.concatenate((sub_state3, auto_state3), axis=0)
        #self.state4 = np.concatenate((sub_state4, auto_state4), axis=0)
        self.state = np.concatenate([[self.state1], [self.state2]])#, [self.state3], [self.state4]])
        self.action_set = [[max_action_set1], [max_action_set2]]#, [max_action_set3], [max_action_set4]]
        reward1 = 0
        reward2 = 0
        #reward3 = 0
        #reward4 = 0

        done = False




        if self.t >= self.T:
            if acc1:# or not acc2:
                reward1 = 1
            else:
                reward1 = -1
            if acc2:# or not acc2:
                reward2 = 1
            else:
                reward2 = -1
            #if acc3:# or not acc2:
            #    reward3 = 1
            #else:
            #    reward3 = -1
            #if acc4:# or not acc2:
            #    reward4 = 1
            #else:
            #    reward4 = -1
            done = True

        reward = np.array([reward1, reward2])#, reward3, reward4])

        return self.state, reward, done, self.action_set, self.player


#if __name__ == '__main__':
#    env = Env()
#    s, a, p = env.reset()
#    s, r, d, a, p = env.step(2)
#    s, r, d, a, p = env.step(2)
#    s, r, d, a, p = env.step(2)
#    s, r, d, a, p = env.step(2)
#    print(s)
#   print(a)