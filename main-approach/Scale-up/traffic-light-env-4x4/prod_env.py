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

    def step(self, label, l1, l2, l3, l4):
        index = self.onehot2index(self.q)
        label = self.onehot2index(label)
        if index == 0: #Initial state
            if label in [0,1] and l1 in [0,1] and l2 in [0,1] and l3 in [0,1] and l4 in [0,1]:
                index = 0
            elif label in [2]: #or l1 in [2] or l2 in [2] or l3 in [2] or l4 in [2]:
                index = 1 #rejecting state
        elif index == 1:
            index = 1
        self.q = self.index2onehot(index, self.max_state)
        return self.q

# The subsystem for the room temp. The way
class Subsystem1():
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
        self.subsystem1 = Subsystem1()
        self.max_action_set = None
        self.T = 200  # Horizon
        self.n_l = 81
        self.n_l_ind = 3
        self.n_s = 1
        self.n_a_1 = 81
        self.n_a_2 = 2
        self.n_a_3 = 81
        self.n_s_1 = 9
        self.n_s_2 = 21
        self.n_s_3 = 23
        self.changing_action = None
        #self.action_for_others_value = [np.array([0,0,0])]

    def onehot2index(self, onehot):
        index = np.where(onehot == 1)[0][0]
        return index

    def index2onehot(self, index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    def allLabels(self):
        a = np.arange(81)
        return [x for x in a]#[0]#



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
        self.player = 1  # 0 for maximizing player, 1 for minimizing
        sub_state, max_action_set = self.subsystem1.reset()
        self.max_action_set = max_action_set
        auto_state1 = self.automaton1.reset()
        sub_state = sub_state[0]
        self.state = np.concatenate((sub_state, auto_state1), axis=0)
        action_set = self.allLabels()  # Labels
        return self.state, action_set, self.player


    def change_the_action(self, action):
        if action == 0:
            action = 1
        else:
            action =0

    def getting_all_the_neighbors(self, action):
        le = int(action / 27)
        ln = int((action % 27) / 9)
        ls = int(((action % 27) % 9) / 3)
        lw = int(((action % 27) % 9) % 3)
        return [self.index2onehot(le, self.n_l_ind), self.index2onehot(ln, self.n_l_ind)
            , self.index2onehot(ls, self.n_l_ind), self.index2onehot(lw, self.n_l_ind)]


    def step(self, action):
        action_set = None
        acc1 = None
        rej1 = None
        acc2 = None
        reward = 0
        done = False
        if len(self.state) == self.n_s_1:  # min label selected
            self.player = 0
            labels = self.getting_all_the_neighbors(action)
            self.state = np.concatenate((self.state, labels[0], labels[1], labels[2], labels[3]), axis=0)
            action_set = self.max_action_set
        elif len(self.state) == self.n_s_2:  # max action selected
            self.player = 1
            action_set = self.getStates(self.onehot2index(self.state[7:]))  # Get states in label
            action = self.index2onehot(action, self.n_a_2)
            self.state = np.concatenate((self.state, action), axis=0)#self.state + (action,)  # Append action to state
        elif len(self.state) == self.n_s_3:  # min state selected
            self.player = 0
            self.t = self.t + 1
            #print(self.t)
            temp_1 = self.state[0:4]
            labels = self.state[self.n_s_1:self.n_s_2]
            l1 = labels[0:3]
            l2 = labels[3:6]
            l3 = labels[6:9]
            l4 = labels[9:12]

            temp_2_1 = self.onehot2index(l1)
            temp_2_2 = self.onehot2index(l2)
            temp_2_3 = self.onehot2index(l3)
            temp_2_4 = self.onehot2index(l4)
            auto_state1 = self.automaton1.step(self.getLabel(temp_1), temp_2_1, temp_2_2, temp_2_3, temp_2_4)

            acc1 = self.onehot2index(auto_state1) == 0
            rej1 = self.onehot2index(auto_state1) == 1
            #if rej1:
            #    done = True
            temp_3 = self.state[self.n_s_2:self.n_s_3]
            junction_action = [self.onehot2index(temp_3)]

            #action_to_implement = np.concatenate((junction_action), axis=0)
            action_to_implement = np.array([junction_action], dtype=np.int64)

            #np.array([[1, 0, 1, 0]], dtype=np.int64)#
            state, max_action_set, reward, done = \
                self.subsystem1.step(action_to_implement)
            #elapsed = self.elapsed_time(state)
            #self.action_for_others_value = self.action_for_others(elapsed,self.action_for_others_value)
            sub_state = [state[0]]
            #jam_length = self.jam_length([state])
            #if rej1:
            #    done = True
            temp_4 = self.state[7:17]
            self.max_action_set = max_action_set
            sub_state = sub_state[0]
            if acc1:
                reward=1
            else:
                reward=0
            #reward = reward[0]
            self.state = np.concatenate((sub_state, auto_state1), axis=0)
            action_set = self.allLabels()
        else:
            raise RuntimeError("unreachable")

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
        return self.state, reward, done, action_set, self.player