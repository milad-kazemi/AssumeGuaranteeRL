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
            if label in [0]:
                index = 0
            elif label in [1]:
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
        self.automaton2 = Automaton1()
        self.automaton3 = Automaton1()
        self.subsystem1 = Subsystem1()
        self.max_action_set = None
        self.T = 300  # Horizon
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
        return [0]#[0, 1, 2, 3]



    def getLabel(self, state):
        #if state[1]:
        #    return 5
        #x = self.onehot2index(state)
        sub_state = state[0:4]
        jam = sum(sub_state)
        if jam < 20:
            l=0
            return self.index2onehot(l, self.n_l)
        else:
            l = 1
            return self.index2onehot(l, self.n_l)


    def getStates(self, label):
        if label == 0:
            a = [0, 1]
            return [x for x in a]
        elif label == 1:
            a = [2]
            return [x for x in a]

    def reset(self):
        self.t = 0
        self.player = 1  # 0 for maximizing player, 1 for minimizing
        sub_state, max_action_set = self.subsystem1.reset()
        self.max_action_set = max_action_set
        auto_state1 = self.automaton1.reset()
        auto_state2 = self.automaton2.reset() #auto_state2, _
        auto_state3 = self.automaton3.reset()  # auto_state3, _
        sub_state = sub_state[0]
        self.state = np.concatenate((sub_state, auto_state1, auto_state2, auto_state3), axis=0)
        action_set = self.allLabels()  # Labels
        self.changing_action = np.zeros(4)
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
        reward = 0
        done = False
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
            self.player = 0
            self.t = self.t + 1
            #print(self.t)
            temp_1 = self.state[0:4]
            auto_state1 = self.automaton1.step(self.getLabel(temp_1))
            temp_2 = self.state[self.n_s_1:self.n_s_2]
            temp_2 = self.onehot2index(temp_2)
            temp_2_2 = int(temp_2 / 2)
            temp_2_3 = temp_2 % 2
            auto_state2 = self.automaton2.step(self.index2onehot(temp_2_2, 2))
            auto_state3 = self.automaton3.step(self.index2onehot(temp_2_3, 2))
            acc1 = self.onehot2index(auto_state1) == 0

            rej1 = self.onehot2index(auto_state1) == 1 or\
                   self.onehot2index(auto_state2) == 1 or\
                   self.onehot2index(auto_state3) == 1
            #if rej1:
            #    done = True
            temp_3 = self.state[self.n_s_2:self.n_s_3]
            junction_action = [self.onehot2index(temp_3)]

            action_for_others = self.action_for_others_value[0]
            action_to_implement = np.concatenate((junction_action, action_for_others), axis=0)
            action_to_implement = np.array([action_to_implement], dtype=np.int64)

            #np.array([[1, 0, 1, 0]], dtype=np.int64)#
            state, max_action_set, reward, done = \
                self.subsystem1.step(action_to_implement)
            elapsed = self.elapsed_time(state)
            self.action_for_others_value = self.action_for_others(elapsed,self.action_for_others_value)
            sub_state = [state[0]]
            jam_length = self.jam_length([state])
            if rej1:
                done = True
            temp_4 = self.state[7:17]
            self.max_action_set = max_action_set
            sub_state = sub_state[0]
            if acc1:
                reward=1
            else:
                reward=0
            #reward = reward[0]
            self.state = np.concatenate((sub_state, auto_state1, auto_state2, auto_state3), axis=0)
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


