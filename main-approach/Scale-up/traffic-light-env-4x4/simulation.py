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

    def getStates(label):
        if label == 0:
            a = [0, 4, 8]
            return [x for x in a]
        elif label == 1:
            a = [10, 14, 18]
            return [x for x in a]
        else:
            a = [20, 24, 28]
            return [x for x in a]



    def the_action(agent, observation, player):
        temp = np.zeros((1, 21),dtype=np.float32)
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
    automaton4 = Automaton1()
    automaton5 = Automaton1()
    automaton6 = Automaton1()
    automaton7 = Automaton1()
    automaton8 = Automaton1()
    automaton9 = Automaton1()
    automaton10 = Automaton1()
    automaton11 = Automaton1()
    automaton12 = Automaton1()
    automaton13 = Automaton1()
    automaton14 = Automaton1()
    automaton15 = Automaton1()

    n = 100
    nt = 100
    SMC = np.zeros(n)


    i_1 = 0
    f = []
    directory = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\scale-up\\traffic-light-env-4x4\\saved-models"
    for filename in os.listdir(directory):#662['episode 254']:#
        f = os.path.join(directory, filename)
        # checking if it is a file
        print(f)
        agent = T.load(f)

        # Steps
        stat_avg_total = 0
        stat0_avg = 0
        stat1_avg = 0
        stat2_avg = 0
        stat3_avg = 0
        stat4_avg = 0
        stat5_avg = 0
        stat6_avg = 0
        stat7_avg = 0
        stat8_avg = 0
        stat9_avg = 0
        stat10_avg = 0
        stat11_avg = 0
        stat12_avg = 0
        stat13_avg = 0
        stat14_avg = 0
        stat15_avg = 0
        for i in range(0, n):
            #print(i)
            # Reset
            player = 0
            observation, action_set = subsystem.reset()
            auto_state0 = automaton0.reset()
            auto_state1 = automaton1.reset()
            auto_state2 = automaton2.reset()
            auto_state3 = automaton3.reset()
            auto_state4 = automaton4.reset()
            auto_state5 = automaton5.reset()
            auto_state6 = automaton6.reset()
            auto_state7 = automaton7.reset()
            auto_state8 = automaton8.reset()
            auto_state9 = automaton9.reset()
            auto_state10 = automaton10.reset()
            auto_state11 = automaton11.reset()
            auto_state12 = automaton12.reset()
            auto_state13 = automaton13.reset()
            auto_state14 = automaton14.reset()
            auto_state15 = automaton15.reset()

            substate0 = observation[0][0:4]
            substate1 = observation[1][0:4]
            substate2 = observation[2][0:4]
            substate3 = observation[3][0:4]
            substate4 = observation[4][0:4]
            substate5 = observation[5][0:4]
            substate6 = observation[6][0:4]
            substate7 = observation[7][0:4]
            substate8 = observation[8][0:4]
            substate9 = observation[9][0:4]
            substate10 = observation[10][0:4]
            substate11 = observation[11][0:4]
            substate12 = observation[12][0:4]
            substate13 = observation[13][0:4]
            substate14 = observation[14][0:4]
            substate15 = observation[15][0:4]

            label0 = getLabel(substate0)
            label1 = getLabel(substate1)
            label2 = getLabel(substate2)
            label3 = getLabel(substate3)
            label4 = getLabel(substate4)
            label5 = getLabel(substate5)
            label6 = getLabel(substate6)
            label7 = getLabel(substate7)
            label8 = getLabel(substate8)
            label9 = getLabel(substate9)
            label10 = getLabel(substate10)
            label11 = getLabel(substate11)
            label12 = getLabel(substate12)
            label13 = getLabel(substate13)
            label14 = getLabel(substate14)
            label15 = getLabel(substate15)

            state0 = np.concatenate((observation[0], auto_state0, label4, label3, label8, label14),
                                    axis=0)  # c: E: 1  N: 2 S: 5 W: 8
            state1 = np.concatenate((observation[1], auto_state1, label8, label14, label11, label2),
                                    axis=0)  # E: W:8 NE: 3 SE: 6 C: 0
            state2 = np.concatenate((observation[2], auto_state2, label1, label10, label5, label13),
                                    axis=0)  # N: NE: 3 S: 5 C: 0 NW: 4
            state3 = np.concatenate((observation[3], auto_state3, label7, label9, label0, label6),
                                    axis=0)  # NE: NW: 4 SE: 6 E: 1 N: 2
            state4 = np.concatenate((observation[4], auto_state4, label10, label7, label13, label0),
                                    axis=0)  # NW: N
            state5 = np.concatenate((observation[5], auto_state5, label11, label2, label12, label15),
                                    axis=0)  # should be mdified
            state6 = np.concatenate((observation[6], auto_state6, label3, label11, label14, label2),
                                    axis=0)  # should be mdified
            state7 = np.concatenate((observation[7], auto_state7, label12, label15, label4, label3),
                                    axis=0)  # should be mdified
            state8 = np.concatenate((observation[8], auto_state8, label13, label0, label9, label1),
                                    axis=0)  # should be mdified
            state9 = np.concatenate((observation[9], auto_state9, label15, label8, label3, label11),
                                    axis=0)  # should be mdified
            state10 = np.concatenate((observation[10], auto_state10, label14, label12, label2, label4),
                                    axis=0)  # c: E: 1  N: 2 S: 5 W: 8
            state11 = np.concatenate((observation[11], auto_state11, label9, label1, label6, label5),
                                    axis=0)  # E: W:8 NE: 3 SE: 6 C: 0
            state12 = np.concatenate((observation[12], auto_state12, label6, label5, label10, label7),
                                    axis=0)  # N: NE: 3 S: 5 C: 0 NW: 4
            state13 = np.concatenate((observation[13], auto_state13, label2, label4, label15, label8),
                                    axis=0)  # NE: NW: 4 SE: 6 E: 1 N: 2
            state14 = np.concatenate((observation[14], auto_state14, label0, label6, label1, label10),
                                    axis=0)  # NW: N
            state15 = np.concatenate((observation[15], auto_state15, label5, label13, label7, label9),
                                    axis=0)  # should be mdified

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

                state0 = np.concatenate((observation[0], auto_state0, label4, label3, label8, label14),
                                        axis=0)  # c: E: 1  N: 2 S: 5 W: 8
                state1 = np.concatenate((observation[1], auto_state1, label8, label14, label11, label2),
                                        axis=0)  # E: W:8 NE: 3 SE: 6 C: 0
                state2 = np.concatenate((observation[2], auto_state2, label1, label10, label5, label13),
                                        axis=0)  # N: NE: 3 S: 5 C: 0 NW: 4
                state3 = np.concatenate((observation[3], auto_state3, label7, label9, label0, label6),
                                        axis=0)  # NE: NW: 4 SE: 6 E: 1 N: 2
                state4 = np.concatenate((observation[4], auto_state4, label10, label7, label13, label0),
                                        axis=0)  # NW: N
                state5 = np.concatenate((observation[5], auto_state5, label11, label2, label12, label15),
                                        axis=0)  # should be mdified
                state6 = np.concatenate((observation[6], auto_state6, label3, label11, label14, label2),
                                        axis=0)  # should be mdified
                state7 = np.concatenate((observation[7], auto_state7, label12, label15, label4, label3),
                                        axis=0)  # should be mdified
                state8 = np.concatenate((observation[8], auto_state8, label13, label0, label9, label1),
                                        axis=0)  # should be mdified
                state9 = np.concatenate((observation[9], auto_state9, label15, label8, label3, label11),
                                        axis=0)  # should be mdified
                state10 = np.concatenate((observation[10], auto_state10, label14, label12, label2, label4),
                                         axis=0)  # c: E: 1  N: 2 S: 5 W: 8
                state11 = np.concatenate((observation[11], auto_state11, label9, label1, label6, label5),
                                         axis=0)  # E: W:8 NE: 3 SE: 6 C: 0
                state12 = np.concatenate((observation[12], auto_state12, label6, label5, label10, label7),
                                         axis=0)  # N: NE: 3 S: 5 C: 0 NW: 4
                state13 = np.concatenate((observation[13], auto_state13, label2, label4, label15, label8),
                                         axis=0)  # NE: NW: 4 SE: 6 E: 1 N: 2
                state14 = np.concatenate((observation[14], auto_state14, label0, label6, label1, label10),
                                         axis=0)  # NW: N
                state15 = np.concatenate((observation[15], auto_state15, label5, label13, label7, label9),
                                         axis=0)  # should be mdified
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
                act4 = the_action(agent, state4, 0)
                act5 = the_action(agent, state5, 0)
                act6 = the_action(agent, state6, 0)
                act7 = the_action(agent, state7, 0)
                act8 = the_action(agent, state8, 0)
                act9 = the_action(agent, state9, 0)
                act10 = the_action(agent, state10, 0)
                act11 = the_action(agent, state11, 0)
                act12 = the_action(agent, state12, 0)
                act13 = the_action(agent, state13, 0)
                act14 = the_action(agent, state14, 0)
                act15 = the_action(agent, state15, 0)

                l0 = onehot2index(getLabel(substate0))
                l1 = onehot2index(getLabel(substate1))
                l2 = onehot2index(getLabel(substate2))
                l3 = onehot2index(getLabel(substate3))
                l4 = onehot2index(getLabel(substate4))
                l5 = onehot2index(getLabel(substate5))
                l6 = onehot2index(getLabel(substate6))
                l7 = onehot2index(getLabel(substate7))
                l8 = onehot2index(getLabel(substate8))
                l9 = onehot2index(getLabel(substate9))
                l10 = onehot2index(getLabel(substate10))
                l11 = onehot2index(getLabel(substate11))
                l12 = onehot2index(getLabel(substate12))
                l13 = onehot2index(getLabel(substate13))
                l14 = onehot2index(getLabel(substate14))
                l15 = onehot2index(getLabel(substate15))

                auto_state0 = automaton0.step(getLabel(substate0), l4, l3, l8, l14)
                auto_state1 = automaton1.step(getLabel(substate1), l8, l14, l11, l2)
                auto_state2 = automaton2.step(getLabel(substate2), l1, l10, l5, l13)
                auto_state3 = automaton3.step(getLabel(substate3), l7, l9, l0, l6)
                auto_state4 = automaton4.step(getLabel(substate4), l10, l7, l13, l0)
                auto_state5 = automaton5.step(getLabel(substate5), l11, l2, l12, l15)
                auto_state6 = automaton6.step(getLabel(substate6), l3, l11, l14, l2)
                auto_state7 = automaton7.step(getLabel(substate7), l12, l15, l4, l3)
                auto_state8 = automaton8.step(getLabel(substate8), l13, l0, l9, l1)
                auto_state9 = automaton9.step(getLabel(substate9), l15, l8, l3, l11)
                auto_state10 = automaton10.step(getLabel(substate10), l14, l12, l2, l4)
                auto_state11 = automaton11.step(getLabel(substate11), l9, l1, l6, l5)
                auto_state12 = automaton12.step(getLabel(substate12), l6, l5, l10, l7)
                auto_state13 = automaton13.step(getLabel(substate13), l2, l4, l15, l8)
                auto_state14 = automaton14.step(getLabel(substate14), l0, l6, l1, l10)
                auto_state15 = automaton15.step(getLabel(substate15), l5, l13, l7, l9)

                act = np.array([[act0, act1, act2, act3, act4, act5, act6, act7, act8, act9, act10, act11, act12, act13, act14, act15]], dtype=np.int64)


                # print(auto_state2)
                observation, action_set1, reward, done = subsystem.step(act)

                substate0 = observation[0][0:4]
                substate1 = observation[1][0:4]  # should be modified based on neighbors
                substate2 = observation[2][0:4]
                substate3 = observation[3][0:4]
                substate4 = observation[4][0:4]
                substate5 = observation[5][0:4]
                substate6 = observation[6][0:4]  # should be modified based on neighbors
                substate7 = observation[7][0:4]
                substate8 = observation[8][0:4]
                substate9 = observation[9][0:4]
                substate10 = observation[10][0:4]
                substate11 = observation[11][0:4]  # should be modified based on neighbors
                substate12 = observation[12][0:4]
                substate13 = observation[13][0:4]
                substate14 = observation[14][0:4]
                substate15 = observation[15][0:4]

                label0 = getLabel(substate0)
                label1 = getLabel(substate1)
                label2 = getLabel(substate2)
                label3 = getLabel(substate3)
                label4 = getLabel(substate4)
                label5 = getLabel(substate5)
                label6 = getLabel(substate6)
                label7 = getLabel(substate7)
                label8 = getLabel(substate8)
                label9 = getLabel(substate9)
                label10 = getLabel(substate10)
                label11 = getLabel(substate11)
                label12 = getLabel(substate12)
                label13 = getLabel(substate13)
                label14 = getLabel(substate14)
                label15 = getLabel(substate15)

            #print(state_f)
            stat0_avg += int(onehot2index(auto_state0) == 0)
            stat1_avg += int(onehot2index(auto_state1) == 0)
            stat2_avg += int(onehot2index(auto_state2) == 0)
            stat3_avg += int(onehot2index(auto_state3) == 0)
            stat4_avg += int(onehot2index(auto_state4) == 0)
            stat5_avg += int(onehot2index(auto_state5) == 0)
            stat6_avg += int(onehot2index(auto_state6) == 0)
            stat7_avg += int(onehot2index(auto_state7) == 0)
            stat8_avg += int(onehot2index(auto_state8) == 0)
            stat9_avg += int(onehot2index(auto_state9) == 0)
            stat10_avg += int(onehot2index(auto_state10) == 0)
            stat11_avg += int(onehot2index(auto_state11) == 0)
            stat12_avg += int(onehot2index(auto_state12) == 0)
            stat13_avg += int(onehot2index(auto_state13) == 0)
            stat14_avg += int(onehot2index(auto_state14) == 0)
            stat15_avg += int(onehot2index(auto_state15) == 0)
            stat_avg_total += int(onehot2index(auto_state0) == 0 and
                                 onehot2index(auto_state1) == 0 and
                                 onehot2index(auto_state2) == 0 and
                                 onehot2index(auto_state3) == 0 and
                                 onehot2index(auto_state4) == 0 and
                                 onehot2index(auto_state5) == 0 and
                                 onehot2index(auto_state6) == 0 and
                                 onehot2index(auto_state7) == 0 and
                                 onehot2index(auto_state8) == 0 and
                                 onehot2index(auto_state9) == 0 and
                                 onehot2index(auto_state10) == 0 and
                                 onehot2index(auto_state11) == 0 and
                                 onehot2index(auto_state12) == 0 and
                                 onehot2index(auto_state13) == 0 and
                                 onehot2index(auto_state14) == 0 and
                                 onehot2index(auto_state15) == 0)

            #print(stat_avg)
            #print(state_f)
            #plt.plot(range(0, nt), state_f)
            # plt.show()
        stat0_avg = stat0_avg / n
        stat1_avg = stat1_avg / n
        stat2_avg = stat2_avg / n
        stat3_avg = stat3_avg / n
        stat4_avg = stat4_avg / n
        stat5_avg = stat5_avg / n
        stat6_avg = stat6_avg / n
        stat7_avg = stat7_avg / n
        stat8_avg = stat8_avg / n
        stat9_avg = stat9_avg / n
        stat10_avg = stat10_avg / n
        stat11_avg = stat11_avg / n
        stat12_avg = stat12_avg / n
        stat13_avg = stat13_avg / n
        stat14_avg = stat14_avg / n
        stat15_avg = stat15_avg / n
        stat_avg_total = stat_avg_total / n

        #plt.xlabel('time step')
        #plt.ylabel('S')
        #plt.show()
        #plt.savefig(f1, format="png")
        print(stat_avg_total, stat0_avg, stat1_avg, stat2_avg, stat3_avg, stat4_avg, stat5_avg, stat6_avg, stat7_avg, stat8_avg,
              stat9_avg, stat10_avg, stat11_avg, stat12_avg, stat13_avg, stat14_avg, stat15_avg)
        SMC[i_1] = stat0_avg
        writer.add_scalar('SMC200', stat0_avg, i_1)
        i_1+=1
        #stat_avg = 0