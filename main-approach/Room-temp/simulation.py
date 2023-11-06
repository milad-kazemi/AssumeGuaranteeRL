# Imports necessary libraries and classes for the simulation
from simple_dqn_torch import Agent
from room_temp_original import Env
from room_temp_original import Subsystem1
from room_temp_original import Automaton1
import matplotlib as plt
import numpy as np
import torch as T
import os
from torch.utils.tensorboard import SummaryWriter
# Initialize TensorBoard writer for logging
writer = SummaryWriter("runs")


# Helper functions for encoding and decoding states as one-hot vectors
def onehot2index(onehot):
    index = np.where(onehot == 1)[0][0]
    return index


def index2onehot(index, n):
    onehot = np.zeros(n)
    onehot[index] = 1
    return onehot

# Function to assign labels based on the state value
def getLabel(state):
    n_l = 8
    # if state[1]:
    #    return 5
    # x = self.onehot2index(state)
    if state <= 16:
        l = 0
        return index2onehot(l, n_l)
    elif state > 16 and state <= 17:
        l = 1
        return index2onehot(l, n_l)
    elif state > 17 and state <= 18:
        l = 2
        return index2onehot(l, n_l)
    elif state > 18 and state <= 19:
        l = 3
        return index2onehot(l, n_l)
    elif state > 19 and state <= 20:
        l = 4
        return index2onehot(l, n_l)
    elif state > 20 and state <= 21:
        l = 5
        return index2onehot(l, n_l)
    elif state > 21 and state <= 22:
        l = 6
        return index2onehot(l, n_l)
    else:
        l = 7
        return index2onehot(l, n_l)


# Main simulation loop starts here
# flip = {0: 2, 2: 0, 1: 3, 3: 1}  # Pick mirrored action
if __name__ == '__main__':
    # Reset
    subsystem1 = Subsystem1()
    subsystem2 = Subsystem1()
    automaton1 = Automaton1()
    automaton2 = Automaton1()
    n = 1000 # Number of iterations
    nt = 40 # Number of time steps
    SMC = np.zeros(n)
    i_1 = 0
    # Set directory where the saved models are located
    directory = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\Room-temp\\saved-models"
# Loop through each saved model in the directory
    for filename in os.listdir(directory):  # 662['episode 254']:#
        f = os.path.join(directory, filename)
        # checking if it is a file
        print(f)
        agent = T.load(f)
        stat_avg = 0
        stat_avg1 = 0
        stat_avg2 = 0
        for i in range(0, n):
            state1, action_set1 = subsystem1.reset()
            state2, action_set2 = subsystem2.reset()
            auto_state1 = automaton1.reset()
            auto_state2 = automaton2.reset()
            label1 = getLabel(state1[0])
            label2 = getLabel(state2[0])

            # Steps
            for t in range(0, nt):
                # S = np.zeros((4, 3))

                # S[onehot2index(state1) // 3, onehot2index(state1) % 3] = 1
                # S[onehot2index(state2) // 3, onehot2index(state2) % 3] = 2
                # plt.imshow(S)
                # plt.title('T = {:d}'.format(t))
                # plt.show()
                env_state1 = np.concatenate((state1, auto_state1, auto_state2, label2), axis=0)
                #print(state1)
                #print(state2)
                env_state2 = np.concatenate((state2, auto_state2, auto_state1, label1), axis=0)
                # print(Q.Q[env_state])
                act1 = agent.action_masking(env_state1, action_set1, 0)

                act2 = agent.action_masking(env_state2, action_set1,0)
                auto_state1 = automaton1.step(getLabel(state1[0]))
                auto_state2 = automaton2.step(getLabel(state2[0]))
                next_state1, action_set1 = subsystem1.step(act1, state2[0])
                next_state2, action_set2 = subsystem2.step(act2, state1[0])
                #print(state1, state2)
                state1, state2 = next_state1, next_state2
                label1 = getLabel(state1[0])
                label2 = getLabel(state2[0])
                ll1 = onehot2index(label1)
                ll2 = onehot2index(label2)
                #print(ll1, ll2)


            stat_avg1 += int(onehot2index(auto_state1) == 1)
            stat_avg2 += int(onehot2index(auto_state2) == 1)
            stat_avg += int(onehot2index(auto_state1) == 1 and onehot2index(auto_state2) == 1)

        stat_avg = stat_avg / n
        stat_avg1 = stat_avg1 / n
        stat_avg2 = stat_avg2 / n

            # plt.xlabel('time step')
            # plt.ylabel('S')
            # plt.show()
            # plt.savefig(f1, format="png")
        print(stat_avg, stat_avg1, stat_avg2)
        SMC[i_1] = stat_avg1
        writer.add_scalar('SMC200', stat_avg1, i_1)
        i_1 += 1




