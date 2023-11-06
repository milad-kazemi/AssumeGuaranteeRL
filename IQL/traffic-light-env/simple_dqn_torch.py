import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims1, input_dims2, input_dims3, fc11_dims, fc12_dims,
                 fc21_dims, fc22_dims, fc31_dims, fc32_dims,
                 n_actions1, n_actions2, n_actions3):
        super(DeepQNetwork, self).__init__()
        self.input_dims1 = input_dims1
        self.input_dims2 = input_dims2
        self.input_dims3 = input_dims3
        self.fc11_dims = fc11_dims
        self.fc12_dims = fc12_dims
        self.fc21_dims = fc21_dims
        self.fc22_dims = fc22_dims
        self.fc31_dims = fc31_dims
        self.fc32_dims = fc32_dims
        self.n_actions1 = n_actions1
        self.n_actions2 = n_actions2
        self.n_actions3 = n_actions3
        self.fc111 = nn.Linear(*self.input_dims1, self.fc11_dims)
        self.fc121 = nn.Linear(self.fc11_dims, self.fc12_dims)
        self.fc131 = nn.Linear(self.fc12_dims, self.n_actions1)
        self.fc112 = nn.Linear(*self.input_dims1, self.fc11_dims)
        self.fc122 = nn.Linear(self.fc11_dims, self.fc12_dims)
        self.fc132 = nn.Linear(self.fc12_dims, self.n_actions1)
        self.fc113 = nn.Linear(*self.input_dims1, self.fc11_dims)
        self.fc123 = nn.Linear(self.fc11_dims, self.fc12_dims)
        self.fc133 = nn.Linear(self.fc12_dims, self.n_actions1)
        self.fc114 = nn.Linear(*self.input_dims1, self.fc11_dims)
        self.fc124 = nn.Linear(self.fc11_dims, self.fc12_dims)
        self.fc134 = nn.Linear(self.fc12_dims, self.n_actions1)

        self.fc115 = nn.Linear(*self.input_dims1, self.fc11_dims)
        self.fc125 = nn.Linear(self.fc11_dims, self.fc12_dims)
        self.fc135 = nn.Linear(self.fc12_dims, self.n_actions1)

        self.fc116 = nn.Linear(*self.input_dims1, self.fc11_dims)
        self.fc126 = nn.Linear(self.fc11_dims, self.fc12_dims)
        self.fc136 = nn.Linear(self.fc12_dims, self.n_actions1)

        self.fc117 = nn.Linear(*self.input_dims1, self.fc11_dims)
        self.fc127 = nn.Linear(self.fc11_dims, self.fc12_dims)
        self.fc137 = nn.Linear(self.fc12_dims, self.n_actions1)

        self.fc118 = nn.Linear(*self.input_dims1, self.fc11_dims)
        self.fc128 = nn.Linear(self.fc11_dims, self.fc12_dims)
        self.fc138 = nn.Linear(self.fc12_dims, self.n_actions1)

        self.fc119 = nn.Linear(*self.input_dims1, self.fc11_dims)
        self.fc129 = nn.Linear(self.fc11_dims, self.fc12_dims)
        self.fc139 = nn.Linear(self.fc12_dims, self.n_actions1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward1(self, state):
        x = F.relu(self.fc111(state))
        x = F.relu(self.fc121(x))
        actions = self.fc131(x)

        return actions

    def forward2(self, state):
        x = F.relu(self.fc112(state))
        x = F.relu(self.fc122(x))
        actions = self.fc132(x)

        return actions

    def forward3(self, state):
        x = F.relu(self.fc113(state))
        x = F.relu(self.fc123(x))
        actions = self.fc133(x)

        return actions

    def forward4(self, state):
        x = F.relu(self.fc114(state))
        x = F.relu(self.fc124(x))
        actions = self.fc134(x)

        return actions

    def forward5(self, state):
        x = F.relu(self.fc115(state))
        x = F.relu(self.fc125(x))
        actions = self.fc135(x)

        return actions

    def forward6(self, state):
        x = F.relu(self.fc116(state))
        x = F.relu(self.fc126(x))
        actions = self.fc136(x)

        return actions

    def forward7(self, state):
        x = F.relu(self.fc117(state))
        x = F.relu(self.fc127(x))
        actions = self.fc137(x)

        return actions

    def forward8(self, state):
        x = F.relu(self.fc118(state))
        x = F.relu(self.fc128(x))
        actions = self.fc138(x)

        return actions

    def forward9(self, state):
        x = F.relu(self.fc119(state))
        x = F.relu(self.fc129(x))
        actions = self.fc139(x)

        return actions





class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims1, input_dims2, input_dims3, batch_size, n_actions1,
                 n_actions2, n_actions3,
                 max_mem_size=10000, eps_end=0.05, eps_dec=5e-4):#eps_dec=5e-4
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space1 = [i for i in range(n_actions1)]
        self.action_space2 = [i for i in range(n_actions2)]
        self.action_space3 = [i for i in range(n_actions3)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100
        self.input_dims1 = input_dims1
        self.input_dims2 = input_dims2
        self.input_dims3 = input_dims3
        self.n_actions1 = n_actions1
        self.n_actions2 = n_actions2
        self.n_actions3 = n_actions3

        self.Q_eval = DeepQNetwork(lr, n_actions1=n_actions1, n_actions2=n_actions2,
                                   n_actions3=n_actions3,
                                   input_dims1=input_dims1, input_dims2=input_dims2,
                                   input_dims3=input_dims3,
                                   fc11_dims=256, fc12_dims=256,
                                   fc21_dims=256, fc22_dims=256,
                                   fc31_dims=256, fc32_dims=256)
        self.Q_tgt = DeepQNetwork(lr, n_actions1=n_actions1, n_actions2=n_actions2,
                                   n_actions3=n_actions3,
                                   input_dims1=input_dims1, input_dims2=input_dims2,
                                   input_dims3=input_dims3,
                                   fc11_dims=256, fc12_dims=256,
                                   fc21_dims=256, fc22_dims=256,
                                   fc31_dims=256, fc32_dims=256)
        self.state1_memory = np.zeros((self.mem_size, *input_dims1),
                                     dtype=np.float32)
        self.new_state1_memory = np.zeros((self.mem_size, *input_dims1),
                                     dtype=np.float32)
        self.state2_memory = np.zeros((self.mem_size, *input_dims1),
                                      dtype=np.float32)
        self.new_state2_memory = np.zeros((self.mem_size, *input_dims1),
                                          dtype=np.float32)
        self.state3_memory = np.zeros((self.mem_size, *input_dims1),
                                      dtype=np.float32)
        self.new_state3_memory = np.zeros((self.mem_size, *input_dims1),
                                          dtype=np.float32)
        self.state4_memory = np.zeros((self.mem_size, *input_dims1),
                                      dtype=np.float32)
        self.new_state4_memory = np.zeros((self.mem_size, *input_dims1),
                                          dtype=np.float32)
        self.state5_memory = np.zeros((self.mem_size, *input_dims1),
                                      dtype=np.float32)
        self.new_state5_memory = np.zeros((self.mem_size, *input_dims1),
                                          dtype=np.float32)
        self.state6_memory = np.zeros((self.mem_size, *input_dims1),
                                      dtype=np.float32)
        self.new_state6_memory = np.zeros((self.mem_size, *input_dims1),
                                          dtype=np.float32)
        self.state7_memory = np.zeros((self.mem_size, *input_dims1),
                                      dtype=np.float32)
        self.new_state7_memory = np.zeros((self.mem_size, *input_dims1),
                                          dtype=np.float32)
        self.state8_memory = np.zeros((self.mem_size, *input_dims1),
                                      dtype=np.float32)
        self.new_state8_memory = np.zeros((self.mem_size, *input_dims1),
                                          dtype=np.float32)
        self.state9_memory = np.zeros((self.mem_size, *input_dims1),
                                      dtype=np.float32)
        self.new_state9_memory = np.zeros((self.mem_size, *input_dims1),
                                          dtype=np.float32)
        self.action1_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action2_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action3_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action4_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action5_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action6_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action7_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action8_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action9_memory = np.zeros(self.mem_size, dtype=np.int32)

        self.reward1_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward2_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward3_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward4_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward5_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward6_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward7_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward8_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward9_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.terminal1_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal2_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal3_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal4_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal5_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal6_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal7_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal8_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal9_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        #######1st player
        self.state1_memory[index] = state[0]
        self.new_state1_memory[index] = state_[0]
        self.reward1_memory[index] = reward[0]
        self.action1_memory[index] = action[0][0]  # need to be changed
        self.terminal1_memory[index] = terminal
        #######2nd player
        self.state2_memory[index] = state[1]
        self.new_state2_memory[index] = state_[1]
        self.reward2_memory[index] = reward[1]
        self.action2_memory[index] = action[0][1]  # need to be changed
        self.terminal2_memory[index] = terminal
        #######3rd player
        self.state3_memory[index] = state[2]
        self.new_state3_memory[index] = state_[2]
        self.reward3_memory[index] = reward[2]
        self.action3_memory[index] = action[0][2]  # need to be changed
        self.terminal3_memory[index] = terminal
        #######4th player
        self.state4_memory[index] = state[3]
        self.new_state4_memory[index] = state_[3]
        self.reward4_memory[index] = reward[3]
        self.action4_memory[index] = action[0][3]  # need to be changed
        self.terminal4_memory[index] = terminal
        #######5th player
        self.state5_memory[index] = state[4]
        self.new_state5_memory[index] = state_[4]
        self.reward5_memory[index] = reward[4]
        self.action5_memory[index] = action[0][4]  # need to be changed
        self.terminal5_memory[index] = terminal
        #######6th player
        self.state6_memory[index] = state[5]
        self.new_state6_memory[index] = state_[5]
        self.reward6_memory[index] = reward[5]
        self.action6_memory[index] = action[0][5]  # need to be changed
        self.terminal6_memory[index] = terminal
        #######7th player
        self.state7_memory[index] = state[6]
        self.new_state7_memory[index] = state_[6]
        self.reward7_memory[index] = reward[6]
        self.action7_memory[index] = action[0][6]  # need to be changed
        self.terminal7_memory[index] = terminal
        #######8th player
        self.state8_memory[index] = state[7]
        self.new_state8_memory[index] = state_[7]
        self.reward8_memory[index] = reward[7]
        self.action8_memory[index] = action[0][7]  # need to be changed
        self.terminal8_memory[index] = terminal
        #######9th player
        self.state9_memory[index] = state[8]
        self.new_state9_memory[index] = state_[8]
        self.reward9_memory[index] = reward[8]
        self.action9_memory[index] = action[0][8]  # need to be changed
        self.terminal9_memory[index] = terminal
        self.mem_cntr += 1

    def get_greedy(self, observation, action_set, player):
        ###############1st player
        temp1 = np.zeros((1, *self.input_dims1),
                        dtype=np.float32)
        temp1[0] = observation[0]
        state1 = T.tensor(temp1).to(self.Q_eval.device)
        actions1 = self.Q_eval.forward1(state1)
        action1 = T.argmax(actions1).item()
        ###############2nd player
        temp2 = np.zeros((1, *self.input_dims1),
                         dtype=np.float32)
        temp2[0] = observation[1]
        state2 = T.tensor(temp2).to(self.Q_eval.device)
        actions2 = self.Q_eval.forward2(state2)
        action2 = T.argmax(actions2).item()
        ###############3rd player
        temp3 = np.zeros((1, *self.input_dims1),
                         dtype=np.float32)
        temp3[0] = observation[2]
        state3 = T.tensor(temp3).to(self.Q_eval.device)
        actions3 = self.Q_eval.forward3(state3)
        action3 = T.argmax(actions3).item()
        ###############4th player
        temp4 = np.zeros((1, *self.input_dims1),
                         dtype=np.float32)
        temp4[0] = observation[3]
        state4 = T.tensor(temp4).to(self.Q_eval.device)
        actions4 = self.Q_eval.forward4(state4)
        action4 = T.argmax(actions4).item()
        ###############5th player
        temp5 = np.zeros((1, *self.input_dims1),
                         dtype=np.float32)
        temp5[0] = observation[4]
        state5 = T.tensor(temp5).to(self.Q_eval.device)
        actions5 = self.Q_eval.forward5(state5)
        action5 = T.argmax(actions5).item()
        ###############6nd player
        temp6 = np.zeros((1, *self.input_dims1),
                         dtype=np.float32)
        temp6[0] = observation[5]
        state6 = T.tensor(temp6).to(self.Q_eval.device)
        actions6 = self.Q_eval.forward6(state6)
        action6 = T.argmax(actions6).item()
        ###############7th player
        temp7 = np.zeros((1, *self.input_dims1),
                         dtype=np.float32)
        temp7[0] = observation[6]
        state7 = T.tensor(temp7).to(self.Q_eval.device)
        actions7 = self.Q_eval.forward7(state7)
        action7 = T.argmax(actions7).item()
        ###############8th player
        temp8 = np.zeros((1, *self.input_dims1),
                         dtype=np.float32)
        temp8[0] = observation[7]
        state8 = T.tensor(temp8).to(self.Q_eval.device)
        actions8 = self.Q_eval.forward8(state8)
        action8 = T.argmax(actions8).item()
        ###############9th player
        temp9 = np.zeros((1, *self.input_dims1),
                         dtype=np.float32)
        temp9[0] = observation[8]
        state9 = T.tensor(temp9).to(self.Q_eval.device)
        actions9 = self.Q_eval.forward9(state9)
        action9 = T.argmax(actions9).item()

        action = [np.array([action1, action2, action3, action4, action5, action6, action7, action8, action9], dtype=np.int64)]

        return action

    def choose_action(self, observation, action_set, player):
        if np.random.random() > self.epsilon:
            action = self.get_greedy(observation, action_set, player)
            #if action not in action_set:
            #    idx1 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            #    idx2 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            #    idx3 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            #    idx4 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            #    action =  [np.array([action_set[idx1], action_set[idx2],action_set[idx3],action_set[idx4]], dtype=np.int64)]
        else:
            idx1 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            idx2 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            idx3 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            idx4 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            idx5 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            idx6 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            idx7 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            idx8 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            idx9 = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))

            action = [np.array([action_set[idx1], action_set[idx2], action_set[idx3], action_set[idx4], action_set[idx5], action_set[idx6],action_set[idx7],action_set[idx8],action_set[idx9]], dtype=np.int64)]

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state1_batch = T.tensor(self.state1_memory[batch]).to(self.Q_eval.device)
        state2_batch = T.tensor(self.state2_memory[batch]).to(self.Q_eval.device)
        state3_batch = T.tensor(self.state3_memory[batch]).to(self.Q_eval.device)
        state4_batch = T.tensor(self.state4_memory[batch]).to(self.Q_eval.device)
        state5_batch = T.tensor(self.state5_memory[batch]).to(self.Q_eval.device)
        state6_batch = T.tensor(self.state6_memory[batch]).to(self.Q_eval.device)
        state7_batch = T.tensor(self.state7_memory[batch]).to(self.Q_eval.device)
        state8_batch = T.tensor(self.state8_memory[batch]).to(self.Q_eval.device)
        state9_batch = T.tensor(self.state9_memory[batch]).to(self.Q_eval.device)

        new_state1_batch = T.tensor(
            self.new_state1_memory[batch]).to(self.Q_eval.device)
        new_state2_batch = T.tensor(
            self.new_state2_memory[batch]).to(self.Q_eval.device)
        new_state3_batch = T.tensor(
            self.new_state3_memory[batch]).to(self.Q_eval.device)
        new_state4_batch = T.tensor(
            self.new_state4_memory[batch]).to(self.Q_eval.device)
        new_state5_batch = T.tensor(
            self.new_state5_memory[batch]).to(self.Q_eval.device)
        new_state6_batch = T.tensor(
            self.new_state6_memory[batch]).to(self.Q_eval.device)
        new_state7_batch = T.tensor(
            self.new_state7_memory[batch]).to(self.Q_eval.device)
        new_state8_batch = T.tensor(
            self.new_state8_memory[batch]).to(self.Q_eval.device)
        new_state9_batch = T.tensor(
            self.new_state9_memory[batch]).to(self.Q_eval.device)

        action1_batch = self.action1_memory[batch]
        action2_batch = self.action2_memory[batch]
        action3_batch = self.action3_memory[batch]
        action4_batch = self.action4_memory[batch]
        action5_batch = self.action5_memory[batch]
        action6_batch = self.action6_memory[batch]
        action7_batch = self.action7_memory[batch]
        action8_batch = self.action8_memory[batch]
        action9_batch = self.action9_memory[batch]

        reward1_batch = T.tensor(
            self.reward1_memory[batch]).to(self.Q_eval.device)
        reward2_batch = T.tensor(
            self.reward2_memory[batch]).to(self.Q_eval.device)
        reward3_batch = T.tensor(
            self.reward3_memory[batch]).to(self.Q_eval.device)
        reward4_batch = T.tensor(
            self.reward4_memory[batch]).to(self.Q_eval.device)
        reward5_batch = T.tensor(
            self.reward5_memory[batch]).to(self.Q_eval.device)
        reward6_batch = T.tensor(
            self.reward6_memory[batch]).to(self.Q_eval.device)
        reward7_batch = T.tensor(
            self.reward7_memory[batch]).to(self.Q_eval.device)
        reward8_batch = T.tensor(
            self.reward8_memory[batch]).to(self.Q_eval.device)
        reward9_batch = T.tensor(
            self.reward9_memory[batch]).to(self.Q_eval.device)


        terminal1_batch = T.tensor(
            self.terminal1_memory[batch]).to(self.Q_eval.device)
        terminal2_batch = T.tensor(
            self.terminal2_memory[batch]).to(self.Q_eval.device)
        terminal3_batch = T.tensor(
            self.terminal3_memory[batch]).to(self.Q_eval.device)
        terminal4_batch = T.tensor(
            self.terminal4_memory[batch]).to(self.Q_eval.device)
        terminal5_batch = T.tensor(
            self.terminal5_memory[batch]).to(self.Q_eval.device)
        terminal6_batch = T.tensor(
            self.terminal6_memory[batch]).to(self.Q_eval.device)
        terminal7_batch = T.tensor(
            self.terminal7_memory[batch]).to(self.Q_eval.device)
        terminal8_batch = T.tensor(
            self.terminal8_memory[batch]).to(self.Q_eval.device)
        terminal9_batch = T.tensor(
            self.terminal9_memory[batch]).to(self.Q_eval.device)

        q_eval1 = self.Q_eval.forward1(state1_batch)[batch_index, action1_batch]
        q_eval2 = self.Q_eval.forward2(state2_batch)[batch_index, action2_batch]
        q_eval3 = self.Q_eval.forward3(state3_batch)[batch_index, action3_batch]
        q_eval4 = self.Q_eval.forward4(state4_batch)[batch_index, action4_batch]
        q_eval5 = self.Q_eval.forward5(state5_batch)[batch_index, action5_batch]
        q_eval6 = self.Q_eval.forward6(state6_batch)[batch_index, action6_batch]
        q_eval7 = self.Q_eval.forward7(state7_batch)[batch_index, action7_batch]
        q_eval8 = self.Q_eval.forward8(state8_batch)[batch_index, action8_batch]
        q_eval9 = self.Q_eval.forward9(state9_batch)[batch_index, action9_batch]

        # Using the next Q function
        q_next1 = self.Q_tgt.forward1(new_state1_batch)  #self.Q_eval.forward2(new_state1_batch)
        q_next2 = self.Q_tgt.forward2(new_state2_batch)  # self.Q_eval.forward2(new_state1_batch)
        q_next3 = self.Q_tgt.forward3(new_state3_batch)  # self.Q_eval.forward2(new_state1_batch)
        q_next4 = self.Q_tgt.forward4(new_state4_batch)  # self.Q_eval.forward2(new_state1_batch)
        q_next5 = self.Q_tgt.forward5(new_state5_batch)  # self.Q_eval.forward2(new_state1_batch)
        q_next6 = self.Q_tgt.forward6(new_state6_batch)  # self.Q_eval.forward2(new_state1_batch)
        q_next7 = self.Q_tgt.forward7(new_state7_batch)  # self.Q_eval.forward2(new_state1_batch)
        q_next8 = self.Q_tgt.forward8(new_state8_batch)  # self.Q_eval.forward2(new_state1_batch)
        q_next9 = self.Q_tgt.forward9(new_state9_batch)  # self.Q_eval.forward2(new_state1_batch)

        q_next1[terminal1_batch] = 0.0
        q_next2[terminal2_batch] = 0.0
        q_next3[terminal3_batch] = 0.0
        q_next4[terminal4_batch] = 0.0
        q_next1[terminal5_batch] = 0.0
        q_next2[terminal6_batch] = 0.0
        q_next3[terminal7_batch] = 0.0
        q_next1[terminal8_batch] = 0.0
        q_next2[terminal9_batch] = 0.0

        # Need to be checked
        q_target1 = reward1_batch + self.gamma * T.max(q_next1, dim=1)[0]
        q_target2 = reward2_batch + self.gamma * T.max(q_next2, dim=1)[0]
        q_target3 = reward3_batch + self.gamma * T.max(q_next3, dim=1)[0]
        q_target4 = reward4_batch + self.gamma * T.max(q_next4, dim=1)[0]
        q_target5 = reward5_batch + self.gamma * T.max(q_next5, dim=1)[0]
        q_target6 = reward6_batch + self.gamma * T.max(q_next6, dim=1)[0]
        q_target7 = reward7_batch + self.gamma * T.max(q_next7, dim=1)[0]
        q_target8 = reward8_batch + self.gamma * T.max(q_next8, dim=1)[0]
        q_target9 = reward9_batch + self.gamma * T.max(q_next9, dim=1)[0]

        loss1 = self.Q_eval.loss(q_target1, q_eval1).to(self.Q_eval.device)
        loss2 = self.Q_eval.loss(q_target2, q_eval2).to(self.Q_eval.device)
        loss3 = self.Q_eval.loss(q_target3, q_eval3).to(self.Q_eval.device)
        loss4 = self.Q_eval.loss(q_target4, q_eval4).to(self.Q_eval.device)
        loss5 = self.Q_eval.loss(q_target5, q_eval5).to(self.Q_eval.device)
        loss6 = self.Q_eval.loss(q_target6, q_eval6).to(self.Q_eval.device)
        loss7 = self.Q_eval.loss(q_target7, q_eval7).to(self.Q_eval.device)
        loss8 = self.Q_eval.loss(q_target8, q_eval8).to(self.Q_eval.device)
        loss9 = self.Q_eval.loss(q_target9, q_eval9).to(self.Q_eval.device)


        loss1.backward()
        loss2.backward()
        loss3.backward()
        loss4.backward()
        loss5.backward()
        loss6.backward()
        loss7.backward()
        loss8.backward()
        loss9.backward()

        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
