import numpy as np
import time
from utils.utils import *
from utils.random_search import grid_choices_random
from utils.grid_search import grid_choices, get_num_grid_choices
import sys
import os
from utils.utils import *
from environments.intersections import IntersectionsEnv, PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE

class Automaton1():
    def __init__(self):
        self.max_state = 2

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
            if label in [0,1]:
                index = 0
            elif label in [2]: #or l1 in [2] or l2 in [2] or l3 in [2] or l4 in [2]:
                index = 1 #rejecting state
        elif index == 1:
            index = 1
        self.q = self.index2onehot(index, self.max_state)
        return self.q

# The subsystem for the room temp. The way
class subsystem1():
    def __init__(self):
        device = torch.device('cpu')
        constants = load_constants('constants/constants.json')

        id = 'eval_0'
        self.env = IntersectionsEnv(constants, device, id, False, get_net_path(constants))
        self.n_u = 2

    def onehot2index(self, onehot):
        index = np.where(onehot == 1)[0][0]
        return index

    def index2onehot(self, index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    def getActionSet(self, state):
        action_set = np.arange(self.n_u) #need to be fixed at the moment we consider all of actions we need to do action masking
        return action_set

    def reset(self):
        self.s = self.env.reset()
        return self.s, self.getActionSet(self.s)

    def step(self, action):
        action = action[0]
        next_state, reward, done = self.env.step(np.array(action, dtype=np.int64), 0, get_global_reward=False)
        self.s = next_state#[next_state[0]]
        return self.s, self.getActionSet(self.s), reward, done


class Env():
    def __init__(self):
        self.automaton1 = Automaton1()
        self.automaton2 = Automaton1()
        self.automaton3 = Automaton1()
        self.automaton4 = Automaton1()
        self.automaton5 = Automaton1()
        self.automaton6 = Automaton1()
        self.automaton7 = Automaton1()
        self.automaton8 = Automaton1()
        self.automaton9 = Automaton1()
        self.subsystem1 = subsystem1()
        self.max_action_set = None
        self.T = 100  # Horizon
        self.n_l = 4
        self.n_s = 1
        self.n_a_1 = 4
        self.n_a_2 = 2
        self.n_a_3 = 9
        self.n_s_1 = 13
        self.n_s_2 = 17
        self.n_s_3 = 19
        self.changing_action = None
        self.action_for_others_value = [np.array([0,0,0])]

    def onehot2index(self, onehot):
        index = np.where(onehot == 1)[0][0]
        return index

    def index2onehot(self, index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    def allLabels(self):
        return [0, 1]#[0, 1, 2, 3]



    def getLabel(self, state):
        #if state[1]:
        #    return 5
        #x = self.onehot2index(state)
        nl = 3
        sub_state_ns = state[1: 3]
        sub_state_we = [state[0], state[3]]
        #waiting_time = state[5]

        jam_ns = sum(sub_state_ns)
        jam_we = sum(sub_state_we)

        jam = jam_ns + jam_we
        #print(state)

        if jam < 10:
            l=0
            return self.index2onehot(l, nl)
        elif 10 <= jam < 20:
            l = 1
            return self.index2onehot(l, nl)
        else:
            l = 2
            return self.index2onehot(l, nl)

    def getStates(self, label):
        if label == 0:
            a = [0, 4, 8]
            return [x for x in a]
        elif label == 1:
            a = [10, 14, 18]
            return [x for x in a]
        else:
            a = [20, 24, 28]
            return [x for x in a]


    def reset(self):
        self.t = 0
        self.player = 0  # 0 for maximizing player, 1 for minimizing
        sub_state, max_action_set = self.subsystem1.reset()
        self.max_action_set = max_action_set
        auto_state1 = self.automaton1.reset()
        auto_state2 = self.automaton2.reset() #auto_state2, _
        auto_state3 = self.automaton3.reset()
        auto_state4 = self.automaton4.reset()
        auto_state5 = self.automaton5.reset()
        auto_state6 = self.automaton6.reset()  # auto_state2, _
        auto_state7 = self.automaton7.reset()
        auto_state8 = self.automaton8.reset()
        auto_state9 = self.automaton9.reset()

        sub_state1 = sub_state[0]
        sub_state2 = sub_state[1]
        sub_state3 = sub_state[2]
        sub_state4 = sub_state[3]
        sub_state5 = sub_state[4]
        sub_state6 = sub_state[5]
        sub_state7 = sub_state[6]
        sub_state8 = sub_state[7]
        sub_state9 = sub_state[8]
        self.state1 = np.concatenate((sub_state1, auto_state1), axis=0)
        self.state2 = np.concatenate((sub_state2, auto_state2), axis=0)
        self.state3 = np.concatenate((sub_state3, auto_state3), axis=0)
        self.state4 = np.concatenate((sub_state4, auto_state4), axis=0)
        self.state5 = np.concatenate((sub_state5, auto_state5), axis=0)
        self.state6 = np.concatenate((sub_state6, auto_state6), axis=0)
        self.state7 = np.concatenate((sub_state7, auto_state7), axis=0)
        self.state8 = np.concatenate((sub_state8, auto_state8), axis=0)
        self.state9 = np.concatenate((sub_state9, auto_state9), axis=0)

        action_set = self.allLabels()  # Labels
        self.changing_action = np.zeros(4)
        self.state = np.concatenate([[self.state1], [self.state2], [self.state3], [self.state4],
                                     [self.state5], [self.state6], [self.state7], [self.state8],
                                     [self.state9]])
        return self.state, action_set, self.player

    def jam_length(self, state):
        l=6
        state = state[0]
        j0 = state[0]
        j0 = sum(j0[0:4])
        j1 = state[1]
        j1 = sum(j1[0:4])
        j2 = state[2]
        j2 = sum(j2[0:4])

        return not(j0<l or j1<l or j2<l)

    def elapsed_time(self, state: object) -> object:
        #elapsed0 = state[0]
        #elapsed0 = elapsed0[6]
        elapsed1 = state[1]
        elapsed1 = elapsed1[6]
        elapsed2 = state[2]
        elapsed2 = elapsed2[6]
        elapsed3 = state[3]
        elapsed3 = elapsed3[6]
        return [np.array([elapsed1, elapsed2, elapsed3])]

    def action_for_others(self, eps, action):
        eps =eps[0]
        tmin = 3*2
        tmax = 10
        action = action[0]
        for i in range(len(eps)):
            if eps[i] <= tmin:
                action[i] = action[i]
            elif tmin < eps[i] < tmax:
                if bool(random.randint(0, 1)):
                    action[i] = self.change_the_action(action[i])
                else:
                    action[i] = action[i]
            else:
                action[i] = self.change_the_action(action[i])
        return [np.array(action)]


    def change_the_action(self, action):
        if action == 0:
            action = 1
        else:
            action =0

    def step(self, action):
        action_set = None
        acc1 = None
        rej1 = None
        acc2 = None
        reward1 = 0
        reward2 = 0
        reward3 = 0
        reward4 = 0
        reward5 = 0
        reward6 = 0
        reward7 = 0
        reward8 = 0
        reward9 = 0
        done = False
        self.player = 0
        self.t = self.t + 1
        #print(self.t)
        temp_1 = self.state1[0:4]
        auto_state1 = self.automaton1.step(self.getLabel(temp_1))
        temp_2 = self.state2[0:4]
        auto_state2 = self.automaton2.step(self.getLabel(temp_2))
        temp_3 = self.state3[0:4]
        auto_state3 = self.automaton3.step(self.getLabel(temp_3))
        temp_4 = self.state4[0:4]
        auto_state4 = self.automaton4.step(self.getLabel(temp_4))
        temp_5 = self.state5[0:4]
        auto_state5 = self.automaton5.step(self.getLabel(temp_5))
        temp_6 = self.state6[0:4]
        auto_state6 = self.automaton6.step(self.getLabel(temp_6))
        temp_7 = self.state7[0:4]
        auto_state7 = self.automaton7.step(self.getLabel(temp_7))
        temp_8 = self.state8[0:4]
        auto_state8 = self.automaton8.step(self.getLabel(temp_8))
        temp_9 = self.state9[0:4]
        auto_state9 = self.automaton9.step(self.getLabel(temp_9))

        acc1 = self.onehot2index(auto_state1) == 0
        acc2 = self.onehot2index(auto_state2) == 0
        acc3 = self.onehot2index(auto_state3) == 0
        acc4 = self.onehot2index(auto_state4) == 0
        acc5 = self.onehot2index(auto_state5) == 0
        acc6 = self.onehot2index(auto_state6) == 0
        acc7 = self.onehot2index(auto_state7) == 0
        acc8 = self.onehot2index(auto_state8) == 0
        acc9 = self.onehot2index(auto_state9) == 0

        rej1 = self.onehot2index(auto_state1) == 1 or\
                self.onehot2index(auto_state2) == 1 or\
                self.onehot2index(auto_state3) == 1 or \
                self.onehot2index(auto_state4) == 1 or \
                self.onehot2index(auto_state5) == 1 or \
                self.onehot2index(auto_state6) == 1 or \
                self.onehot2index(auto_state7) == 1 or \
                self.onehot2index(auto_state8) == 1 or \
                self.onehot2index(auto_state9) == 1

        action_to_implement = action#np.array(action, dtype=np.int64)
        sub_state, max_action_set, reward, done = \
                self.subsystem1.step(action_to_implement)
        sub_state1 = sub_state[0]
        sub_state2 = sub_state[1]
        sub_state3 = sub_state[2]
        sub_state4 = sub_state[3]
        sub_state5 = sub_state[4]
        sub_state6 = sub_state[5]
        sub_state7 = sub_state[6]
        sub_state8 = sub_state[7]
        sub_state9 = sub_state[8]

        self.state1 = np.concatenate((sub_state1, auto_state1), axis=0)
        self.state2 = np.concatenate((sub_state2, auto_state2), axis=0)
        self.state3 = np.concatenate((sub_state3, auto_state3), axis=0)
        self.state4 = np.concatenate((sub_state4, auto_state4), axis=0)
        self.state5 = np.concatenate((sub_state5, auto_state5), axis=0)
        self.state6 = np.concatenate((sub_state6, auto_state6), axis=0)
        self.state7 = np.concatenate((sub_state7, auto_state7), axis=0)
        self.state8 = np.concatenate((sub_state8, auto_state8), axis=0)
        self.state9 = np.concatenate((sub_state9, auto_state9), axis=0)

        #elapsed = self.elapsed_time(state)
        #self.action_for_others_value = self.action_for_others(elapsed,self.action_for_others_value)
        #sub_state = [state[0]]
        #jam_length = self.jam_length([state])
        #if rej1:
        #    done = True
        #temp_4 = self.state[7:17]
        #self.max_action_set = max_action_set
        #sub_state = sub_state[0]
        if acc1:
            reward1=1
        else:
            reward1=0

        if acc2:
            reward2 = 1
        else:
            reward2 = 0

        if acc3:
            reward3 = 1
        else:
            reward3 = 0

        if acc4:
            reward4 = 1
        else:
            reward4 = 0

        if acc5:
            reward5 = 1
        else:
            reward5 = 0

        if acc6:
            reward6 = 1
        else:
            reward6 = 0

        if acc7:
            reward7 = 1
        else:
            reward7 = 0

        if acc8:
            reward8 = 1
        else:
            reward8 = 0

        if acc9:
            reward9 = 1
        else:
            reward9 = 0
            #reward = reward[0]
        #self.state = np.concatenate((sub_state, auto_state1, auto_state2, auto_state3), axis=0)
        #action_set = self.allLabels()

        #reward = 0
        #done = False




        if self.t >= self.T:
        #    if acc1:# or not acc2:
        #        reward = 1
        #    elif rej1:
        #        reward = -1
        #    else:
        #        reward = 0
            done = True
        self.state = np.concatenate([[self.state1], [self.state2], [self.state3], [self.state4],
                                     [self.state5], [self.state6], [self.state7], [self.state8],
                                     [self.state9]])
        reward = np.array([reward1, reward2, reward3, reward4,
                           reward5, reward6, reward7, reward8,
                           reward9])
        self.max_action_set = max_action_set
        action_set = max_action_set

        return self.state, reward, done, action_set, self.player