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
env = Env()


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
        # if state[1]:
        #    return 5
        # x = self.onehot2index(state)
        nl = 3
        sub_state_ns = state[1: 3]
        sub_state_we = [state[0], state[3]]

        jam_ns = sum(sub_state_ns)
        jam_we = sum(sub_state_we)

        jam = jam_ns + jam_we
        # print(state)

        if jam < 10:
            l = 0
            return index2onehot(l, nl)
        elif 10 <= jam < 20:
            l = 1
            return index2onehot(l, nl)
        else:
            l = 2
            return index2onehot(l, nl)


    def get_greedy(agent, observation, action_set, player):
        ###############1st player
        temp1 = np.zeros((1, *agent.input_dims1),
                         dtype=np.float32)
        temp1[0] = observation[0]
        state1 = T.tensor(temp1).to(agent.Q_tgt.device)
        actions1 = agent.Q_tgt.forward1(state1)
        action1 = np.random.randint(2)#T.argmax(actions1).item()
        ###############2nd player
        temp2 = np.zeros((1, *agent.input_dims1),
                         dtype=np.float32)
        temp2[0] = observation[1]
        state2 = T.tensor(temp2).to(agent.Q_eval.device)
        actions2 = agent.Q_eval.forward2(state2)
        action2 = np.random.randint(2)#T.argmax(actions2).item()
        ###############3rd player
        temp3 = np.zeros((1, *agent.input_dims1),
                         dtype=np.float32)
        temp3[0] = observation[2]
        state3 = T.tensor(temp3).to(agent.Q_eval.device)
        actions3 = agent.Q_eval.forward3(state3)
        action3 = np.random.randint(2)#T.argmax(actions3).item()
        ###############4th player
        temp4 = np.zeros((1, *agent.input_dims1),
                         dtype=np.float32)
        temp4[0] = observation[3]
        state4 = T.tensor(temp4).to(agent.Q_eval.device)
        actions4 = agent.Q_eval.forward4(state4)
        action4 = np.random.randint(2)#T.argmax(actions4).item()
        ###############5th player
        temp5 = np.zeros((1, *agent.input_dims1),
                         dtype=np.float32)
        temp5[0] = observation[4]
        state5 = T.tensor(temp5).to(agent.Q_eval.device)
        actions5 = agent.Q_eval.forward5(state5)
        action5 = np.random.randint(2)#T.argmax(actions5).item()
        ###############6nd player
        temp6 = np.zeros((1, *agent.input_dims1),
                         dtype=np.float32)
        temp6[0] = observation[5]
        state6 = T.tensor(temp6).to(agent.Q_eval.device)
        actions6 = agent.Q_eval.forward6(state6)
        action6 = np.random.randint(2)#T.argmax(actions6).item()
        ###############7th player
        temp7 = np.zeros((1, *agent.input_dims1),
                         dtype=np.float32)
        temp7[0] = observation[6]
        state7 = T.tensor(temp7).to(agent.Q_eval.device)
        actions7 = agent.Q_eval.forward7(state7)
        action7 = np.random.randint(2)#T.argmax(actions7).item()
        ###############8th player
        temp8 = np.zeros((1, *agent.input_dims1),
                         dtype=np.float32)
        temp8[0] = observation[7]
        state8 = T.tensor(temp8).to(agent.Q_eval.device)
        actions8 = agent.Q_eval.forward8(state8)
        action8 = np.random.randint(2)#T.argmax(actions8).item()
        ###############9th player
        temp9 = np.zeros((1, *agent.input_dims1),
                         dtype=np.float32)
        temp9[0] = observation[8]
        state9 = T.tensor(temp9).to(agent.Q_eval.device)
        actions9 = agent.Q_eval.forward9(state9)
        action9 = np.random.randint(2)#T.argmax(actions9).item()

        action = [
            np.array([action1, action2, action3, action4, action5, action6, action7, action8, action9], dtype=np.int64)]

        return action


    # flip = {0: 2, 2: 0, 1: 3, 3: 1}  # Pick mirrored action

    # assign directory

    n = 10
    nt = 100
    SMC = np.zeros(n)


    i_1 = 0
    directory = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\IQL\\traffic-light-env\\saved-models"
    for filename in os.listdir(directory):#662['episode 254']:#
        f = os.path.join(directory, filename)
        # checking if it is a file
        print(f)
        agent = T.load(f)

        # Steps
        stat_avg = 0
        rew = 0
        for i in range(0, n):
            #print(i)
            # Reset
            player = 0
            observation, action_set, player = env.reset()
            action_set = np.arange(2)
            player = 0
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

                action_set = np.arange(2)
                act = get_greedy(agent, observation, action_set, 0)
                observation_, reward, done, action_set, player = env.step(act)
                if sum(reward)==9:
                    rew = 1
                else:
                    rew = 0

                # print(auto_state2)

                #print(state_f)
            stat_avg += rew
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


