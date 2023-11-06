# import required module
from simple_dqn_torch import Agent
from prod_env import *
import time
from utils.utils import *
from utils.random_search import grid_choices_random
from utils.grid_search import grid_choices, get_num_grid_choices
import matplotlib.pyplot as plt
from utils.utils import *
from environments.intersections import IntersectionsEnv, PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE
import numpy as np
import torch as T
import sys
import os
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs")



if __name__ == '__main__':
    # Simulation
    # Simulation
    def onehot2index(onehot):
        index = np.where(onehot == 1)[0][0]
        return index


    def index2onehot(index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot


    def getLabel(state):
        #if state[1]:
        #    return 5
        #x = self.onehot2index(state)
        sub_state = state[0:4]
        n_l = 4
        jam = sum(sub_state)
        if jam < 20:
            l=0
            return index2onehot(l, n_l)
        else:
            l = 1
            return index2onehot(l, n_l)



    def the_action(agent, observation, player):
        temp = np.zeros((1, 17),dtype=np.float32)
        temp[0] = observation
        state = T.tensor(temp).to(agent.Q_eval.device)
        actions = agent.Q_tgt.forward2(state)
        if player == 0:
            action = T.argmax(actions).item()
        else:
            action = T.argmin(actions).item()
        return action


    # flip = {0: 2, 2: 0, 1: 3, 3: 1}  # Pick mirrored action

    # assign directory
    f = []

    subsystem = Subsystem1()
    automaton0 = Automaton1()
    automaton1 = Automaton1()
    automaton2 = Automaton1()
    automaton3 = Automaton1()
    n = 10
    nt = 200
    SMC = np.zeros(n)


    i_1 = 0
    directory = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\traffic-light-env\\saved-modelll"
    for filename in os.listdir(directory):#662['episode 254']:#
        f = os.path.join(directory, filename)
        # checking if it is a file
        print(f)
        agent = T.load(f)

        # Steps
        stat_avg = 0
        for i in range(0, n):
            #print(i)
            # Reset
            player = 0
            observation, action_set = subsystem.reset()
            auto_state0 = automaton0.reset()
            auto_state1 = automaton1.reset()
            auto_state2 = automaton2.reset()
            auto_state3 = automaton3.reset()
            substate0 = observation[0]
            substate1 = observation[1]
            substate2 = observation[2]
            substate3 = observation[3]

            label0 = getLabel(substate0)
            label1 = getLabel(substate1)
            label2 = getLabel(substate2)
            label3 = getLabel(substate3)
            l0 = index2onehot(onehot2index(label1) * 2 + onehot2index(label2), 4)
            l1 = index2onehot(onehot2index(label0) * 2 + onehot2index(label3), 4)
            l2 = index2onehot(onehot2index(label0) * 2 + onehot2index(label3), 4)
            l3 = index2onehot(onehot2index(label1) * 2 + onehot2index(label2), 4)

            state0 = np.concatenate((substate0, auto_state0, auto_state1, auto_state2, l0), axis=0)
            state1 = np.concatenate((substate1, auto_state1, auto_state0, auto_state3, l1), axis=0)
            state2 = np.concatenate((substate2, auto_state2, auto_state0, auto_state3, l2), axis=0)
            state3 = np.concatenate((substate3, auto_state3, auto_state1, auto_state2, l3), axis=0)

            state_f = []
            ac = np.zeros(100)
            #for t in range(0,100):
                #tt = t/100+17
                #label2[3] = 0
                #label2[4] = 1
                #env_state1 = np.concatenate(([tt], auto_state1, auto_state2, label2), axis=0)
            #    ac[t] = the_action(agent, state0, 1)
            #plt.plot(np.arange(17, 18, 0.01), ac)

            for t in range(0, nt):
                # S = np.zeros((4, 3))

                # S[onehot2index(state1) // 3, onehot2index(state1) % 3] = 1
                # S[onehot2index(state2) // 3, onehot2index(state2) % 3] = 2
                # plt.imshow(S)
                # plt.title('T = {:d}'.format(t))
                # plt.show()
                state0 = np.concatenate((substate0, auto_state0, auto_state1, auto_state2, l0), axis=0)
                state1 = np.concatenate((substate1, auto_state1, auto_state0, auto_state3, l1), axis=0)
                state2 = np.concatenate((substate2, auto_state2, auto_state0, auto_state3, l2), axis=0)
                state3 = np.concatenate((substate3, auto_state3, auto_state1, auto_state2, l3), axis=0)
                #env_state1 = np.concatenate((state1, auto_state1, auto_state2, label2), axis=0)
                # print(state1)
                #state_f = np.concatenate((state_f, state1), axis=0)
                # print(state2)
                #env_state2 = np.concatenate((state2, auto_state2, auto_state1, label1), axis=0)
                # print(Q.Q[env_state])
                action_set1 = np.arange(2)
                act0 = the_action(agent, state0, 0)
                act1 = the_action(agent, state1, 0)
                act2 = the_action(agent, state2, 0)
                act3 = the_action(agent, state3, 0)
                auto_state0 = automaton0.step(getLabel(substate0))
                auto_state1 = automaton1.step(getLabel(substate1))
                auto_state2 = automaton2.step(getLabel(substate2))
                auto_state3 = automaton3.step(getLabel(substate3))
                act = np.array([[act0, act1, act2, act3]], dtype=np.int64)


                # print(auto_state2)
                observation, action_set1, reward, done = subsystem.step(act)
                substate0 = observation[0]
                substate1 = observation[1]
                substate2 = observation[2]
                substate3 = observation[3]

                label0 = getLabel(substate0)
                label1 = getLabel(substate1)
                label2 = getLabel(substate2)
                label3 = getLabel(substate3)
                l0 = index2onehot(onehot2index(label1) * 2 + onehot2index(label2), 4)
                l1 = index2onehot(onehot2index(label0) * 2 + onehot2index(label3), 4)
                l2 = index2onehot(onehot2index(label0) * 2 + onehot2index(label3), 4)
                l3 = index2onehot(onehot2index(label1) * 2 + onehot2index(label2), 4)
                #print(state_f)
            stat_avg += int(onehot2index(auto_state0) == 0)
            #print(stat_avg)
            #print(state_f)
            #plt.plot(range(0, nt), state_f)
            # plt.show()
        stat_avg = stat_avg / n
        #plt.xlabel('time step')
        #plt.ylabel('S')
        #plt.show()
        #plt.savefig(f1, format="png")
        print(stat_avg)
        SMC[i_1] = stat_avg
        writer.add_scalar('SMC200', stat_avg, i_1)
        i_1+=1
        #stat_avg = 0


