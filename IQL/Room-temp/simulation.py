from simple_dqn_torch import Agent
from room_temp import Env
from room_temp import Subsystem1
from room_temp import Automaton1
import matplotlib as plt
import numpy as np
import torch as T
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs")


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
    n_l = 8
    # if state[1]:
    #    return 5
    # x = self.onehot2index(state)
    if state <= 16.75:
        l = 0
        return index2onehot(l, n_l)
    elif state > 16.75 and state <= 17:
        l = 1
        return index2onehot(l, n_l)
    elif state > 17 and state <= 17.25:
        l = 2
        return index2onehot(l, n_l)
    elif state > 17.25 and state <= 17.5:
        l = 3
        return index2onehot(l, n_l)
    elif state > 17.5 and state <= 17.75:
        l = 4
        return index2onehot(l, n_l)
    elif state > 17.75 and state <= 18:
        l = 5
        return index2onehot(l, n_l)
    elif state > 18 and state <= 18.25:
        l = 6
        return index2onehot(l, n_l)
    else:
        l = 7
        return index2onehot(l, n_l)


# flip = {0: 2, 2: 0, 1: 3, 3: 1}  # Pick mirrored action
if __name__ == '__main__':
    # Reset
    env = Env()
    n = 1000
    nt = 40
    SMC = np.zeros(n)
    i_1 = 0
    directory = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\IQL\\Room-temp\\saved-models"
    for filename in os.listdir(directory):  # 662['episode 254']:#
        f = os.path.join(directory, filename)
        # checking if it is a file
        print(f)
        agent = T.load(f)
        stat1_avg = 0
        stat2_avg = 0
        stat_avg_total = 0
        for i in range(0, n):
            observation, action_set, player = env.reset()
            # Steps
            for t in range(0, nt):
                action = agent.get_greedy(observation, action_set, player)
                observation_, reward, done, action_set, player = env.step(action)
                observation = observation_


            stat1_avg += int(onehot2index(observation[0][1:]) == 1)
            stat2_avg += int(onehot2index(observation[1][1:]) == 1)
            stat_avg_total += int(onehot2index(observation[0][1:]) == 1 and onehot2index(observation[1][1:]) == 1)

        stat1_avg = stat1_avg / n
        stat2_avg = stat2_avg / n
        stat_avg_total = stat_avg_total / n
            # plt.xlabel('time step')
            # plt.ylabel('S')
            # plt.show()
            # plt.savefig(f1, format="png")
        print(stat_avg_total, stat1_avg, stat2_avg)
        #SMC[i_1] = stat_avg
        #writer.add_scalar('SMC200', stat_avg, i_1)
        i_1 += 1




