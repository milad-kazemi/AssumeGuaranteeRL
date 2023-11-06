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
        self.subsystem1 = Subsystem1()
        self.max_action_set = None
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
        #if state[1]:
        #    return 5
        #x = self.onehot2index(state)
        if state <= 17:
            l=0
            return self.index2onehot(l, self.n_l)
        elif state > 17 and state <= 18:
            l=1
            return self.index2onehot(l, self.n_l)
        elif state > 18 and state <= 19:
            l=2
            return self.index2onehot(l, self.n_l)
        elif state > 19 and state <= 20:
            l=3
            return self.index2onehot(l, self.n_l)
        elif state > 20 and state <= 21:
            l=4
            return self.index2onehot(l, self.n_l)
        elif state > 21 and state <= 22:
            l=5
            return self.index2onehot(l, self.n_l)
        elif state > 22 and state <= 23:
            l=6
            return self.index2onehot(l, self.n_l)
        else:
            l=7
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
        sub_state, max_action_set = self.subsystem1.reset()
        self.max_action_set = max_action_set
        auto_state1 = self.automaton1.reset()
        auto_state2 = self.automaton2.reset() #auto_state2, _
        self.state = np.concatenate((sub_state, auto_state1, auto_state2), axis=0)
        action_set = self.allLabels()  # Labels
        return self.state, action_set, self.player

    def disturbance_other_players(self, label):
        states = self.getStates(label)
        w = np.random.rand() * 0.0001
        return w


    def step(self, action):
        action_set = None
        acc1 = None
        acc2 = None
        reward = 0
        done = 0
        if len(self.state) == self.n_s_1:  # min label selected
            self.player = 0
            action = self.index2onehot(action, self.n_a_1)
            self.state = np.concatenate((self.state, action), axis=0)#self.state + (action,)  # Append label to state
            action_set = self.max_action_set
        elif len(self.state) == self.n_s_2:  # max action selected
            self.player = 1
            action_set = self.getStates(self.onehot2index(self.state[7:]))  # Get states in label
            action = self.index2onehot(action, self.n_a_2)
            self.state = np.concatenate((self.state, action), axis=0)#self.state + (action,)  # Append action to state
        elif len(self.state) == self.n_s_3:  # min state selected
            self.t = self.t + 1
            temp_1 = self.state[0:1]
            auto_state1 = self.automaton1.step(self.getLabel(temp_1))
            temp_2 = self.state[4:7]
            auto_state2 = self.automaton2.step(temp_2)
            acc1 = self.onehot2index(auto_state1) == 1
            acc2 = self.onehot2index(auto_state2) == 1
            temp_3 = self.state[15:20]
            temp_for_others = action*.5+15
            sub_state, max_action_set = \
                self.subsystem1.step(self.onehot2index(temp_3), temp_for_others)
            temp_4 = self.state[7:17]
            self.max_action_set = max_action_set
            self.state = np.concatenate((sub_state, auto_state1, auto_state2), axis=0)
            action_set = self.allLabels()
            label_reward = self.onehot2index(self.getLabel(temp_1))
            if self.onehot2index(auto_state1) == 0:
                reward = -.05/40
            elif self.onehot2index(auto_state1) == 1:
                reward = 1/40
            elif self.onehot2index(auto_state1) == 2:
                reward = -.20/40

            #if label_reward in [0, 7]:  # or not acc2:
            #    reward = -20
            #elif label_reward in [3, 4, 6]:
            #    reward = -5
            #elif label_reward in [5]:
            #    reward = 100
            #elif label_reward in [1, 2]:
            #    reward = -10
            #else:
            #    reward = -1
            #done = False
        else:
            raise RuntimeError("unreachable")

        #reward = 0





        if self.t >= self.T:
            #if acc1:# or not acc2:
            #    reward = 1
            #else:
            #    reward = 0
            done = True



        return self.state, reward, done, action_set, self.player


#if __name__ == '__main__':
#    env = Env()
#    s, a, p = env.reset()
#    s, r, d, a, p = env.step(2)
#    s, r, d, a, p = env.step(2)
#    s, r, d, a, p = env.step(2)
#    s, r, d, a, p = env.step(2)
#    print(s)
#   print(a)