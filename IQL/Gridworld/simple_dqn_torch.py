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
        self.action1_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action2_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward1_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward2_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal1_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal2_memory = np.zeros(self.mem_size, dtype=np.bool)

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
        action = [np.array([action1, action2], dtype=np.int64)]
        return action

    def choose_action(self, observation, action_set, player):
        if np.random.random() > self.epsilon:
            action = self.get_greedy(observation, action_set, player)
            if action[0][0] not in action_set[0][0]:
                idx1 = np.random.choice(T.tensor(list(range(len(action_set[0])))).to(self.Q_eval.device))
                action[0][0] = action_set[0][0][idx1]
            if action[0][1] not in action_set[1][0]:
                idx1 = np.random.choice(T.tensor(list(range(len(action_set[1][0])))).to(self.Q_eval.device))
                action[0][1] = action_set[1][0][idx1]
        else:
            idx1 = np.random.choice(T.tensor(list(range(len(action_set[0][0])))).to(self.Q_eval.device))
            idx2 = np.random.choice(T.tensor(list(range(len(action_set[1][0])))).to(self.Q_eval.device))
            action =  [np.array([action_set[0][0][idx1], action_set[1][0][idx2]], dtype=np.int64)]

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

        new_state1_batch = T.tensor(
            self.new_state1_memory[batch]).to(self.Q_eval.device)
        new_state2_batch = T.tensor(
            self.new_state2_memory[batch]).to(self.Q_eval.device)

        action1_batch = self.action1_memory[batch]
        action2_batch = self.action2_memory[batch]

        reward1_batch = T.tensor(
            self.reward1_memory[batch]).to(self.Q_eval.device)
        reward2_batch = T.tensor(
            self.reward2_memory[batch]).to(self.Q_eval.device)
        terminal1_batch = T.tensor(
            self.terminal1_memory[batch]).to(self.Q_eval.device)
        terminal2_batch = T.tensor(
            self.terminal2_memory[batch]).to(self.Q_eval.device)

        q_eval1 = self.Q_eval.forward1(state1_batch)[batch_index, action1_batch]
        q_eval2 = self.Q_eval.forward2(state2_batch)[batch_index, action2_batch]

        # Using the next Q function
        q_next1 = self.Q_tgt.forward1(new_state1_batch)  #self.Q_eval.forward2(new_state1_batch)
        q_next2 = self.Q_tgt.forward2(new_state2_batch)  # self.Q_eval.forward2(new_state1_batch)

        q_next1[terminal1_batch] = 0.0
        q_next2[terminal2_batch] = 0.0

        # Need to be checked
        q_target1 = reward1_batch + self.gamma * T.max(q_next1, dim=1)[0]
        q_target2 = reward2_batch + self.gamma * T.max(q_next2, dim=1)[0]

        loss1 = self.Q_eval.loss(q_target1, q_eval1).to(self.Q_eval.device)
        loss2 = self.Q_eval.loss(q_target2, q_eval2).to(self.Q_eval.device)

        loss1.backward()
        loss2.backward()

        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
