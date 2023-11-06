from simple_dqn_torch import Agent
from grid_onehot import Env
from grid_onehot import Subsystem1
from grid_onehot import Subsystem2
from grid_onehot import Automaton1
from grid_onehot import Automaton2
import matplotlib as plt
import os
import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs")


#Simulation
def onehot2index(onehot):
    index = np.where(onehot == 1)[0][0]
    return index


def index2onehot(index, n):
    onehot = np.zeros(n)
    onehot[index] = 1
    return onehot


def getLabel(state):
    # if state[1]:
    #    return 5
    n_l = 6
    x = onehot2index(state)
    if x % 3 == 0:
        l = 0
        return index2onehot(l, n_l)
    elif x % 3 == 2:
        l = 2
        return index2onehot(l, n_l)
    elif x == 1:
        l = 3
        return index2onehot(l, n_l)
    elif x == 10:
        l = 4
        return index2onehot(l, n_l)
    else:
        l = 1
        return index2onehot(l, n_l)


flip = {0: 2, 2: 0, 1: 3, 3: 1}  # Pick mirrored action
if __name__ == '__main__':
    # Reset
    subsystem1 = Subsystem1()
    subsystem2 = Subsystem2()
    automaton1 = Automaton1()
    automaton2 = Automaton2()
    n = 1000
    nt = 50
    SMC = np.zeros(n)
    i_1 = 0
    directory = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\IQL\\Gridworld\\saved-models"
    for filename in os.listdir(directory):  # 662['episode 254']:#
        f = os.path.join(directory, filename)
        # checking if it is a file
        print(f)
        agent = T.load(f)
        stat_avg = 0
        for i in range(0, n):
            state1, action_set1 = subsystem1.reset()
            state2, action_set2 = subsystem2.reset()
            auto_state1 = automaton1.reset()
            auto_state2 = automaton2.reset()
            label1 = getLabel(state1)
            label2 = getLabel(state2)

            # Steps
            for t in range(0, 6):
                S = np.zeros((4, 3))

                S[onehot2index(state1) // 3, onehot2index(state1) % 3] = 1
                S[onehot2index(state2) // 3, onehot2index(state2) % 3] = 2
                # plt.imshow(S)
                # plt.title('T = {:d}'.format(t))
                # plt.show()
                env_state1 = np.concatenate((state1, auto_state1), axis=0)
                env_state2 = np.concatenate((state2, auto_state2), axis=0)
                state = np.concatenate([[env_state1], [env_state2]])
                action_set = [[action_set1], [action_set2]]
                # print(Q.Q[env_state])
                act = agent.get_greedy(state, action_set, 0)
                #act2 = flip[act1]  # The solution is symmetrical in this problem. For other problems, more work required.
                auto_state1 = automaton1.step(getLabel(state1))
                auto_state2 = automaton2.step(getLabel(state2))
                next_state1, action_set1 = subsystem1.step(act[0], state1)
                next_state2, action_set2 = subsystem2.step(act[1], state2)
                state1, state2 = next_state1, next_state2
                label1 = getLabel(state1)
                label2 = getLabel(state2)
            stat_avg += int(onehot2index(auto_state1) == 2)

        stat_avg = stat_avg / n
        print(stat_avg)
        SMC[i_1] = stat_avg
        writer.add_scalar('SMC200', stat_avg, i_1)
        i_1 += 1