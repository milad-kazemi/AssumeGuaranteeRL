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
        self.fc11 = nn.Linear(*self.input_dims1, self.fc11_dims)
        self.fc12 = nn.Linear(self.fc11_dims, self.fc12_dims)
        self.fc13 = nn.Linear(self.fc12_dims, self.n_actions1)
        self.fc21 = nn.Linear(*self.input_dims2, self.fc21_dims)
        self.fc22 = nn.Linear(self.fc21_dims, self.fc22_dims)
        self.fc23 = nn.Linear(self.fc22_dims, self.n_actions2)
        self.fc31 = nn.Linear(*self.input_dims3, self.fc31_dims)
        self.fc32 = nn.Linear(self.fc31_dims, self.fc32_dims)
        self.fc33 = nn.Linear(self.fc32_dims, self.n_actions3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward1(self, state):
        x = F.relu(self.fc11(state))
        x = F.relu(self.fc12(x))
        actions = self.fc13(x)

        return actions

    def forward2(self, state):
        x = F.relu(self.fc21(state))
        x = F.relu(self.fc22(x))
        actions = self.fc23(x)

        return actions

    def forward3(self, state):
        x = F.relu(self.fc31(state))
        x = F.relu(self.fc32(x))
        actions = self.fc33(x)

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
        self.new_state1_memory = np.zeros((self.mem_size, *input_dims2),
                                     dtype=np.float32)
        self.state2_memory = np.zeros((self.mem_size, *input_dims2),
                                      dtype=np.float32)
        self.new_state2_memory = np.zeros((self.mem_size, *input_dims3),
                                         dtype=np.float32)
        self.state3_memory = np.zeros((self.mem_size, *input_dims3),
                                      dtype=np.float32)
        self.new_state3_memory = np.zeros((self.mem_size, *input_dims1),
                                         dtype=np.float32)
        self.action1_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action2_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action3_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward1_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward2_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward3_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal1_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal2_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal3_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        if len(state) == self.input_dims1[0]:
            self.state1_memory[index] = state
            self.new_state1_memory[index] = state_
            self.reward1_memory[index] = reward
            self.action1_memory[index] = action #need to be changed
            self.terminal1_memory[index] = terminal
        elif len(state) == self.input_dims2[0]:
            self.state2_memory[index] = state
            self.new_state2_memory[index] = state_
            self.reward2_memory[index] = reward
            self.action2_memory[index] = action
            self.terminal2_memory[index] = terminal
        elif len(state) == self.input_dims3[0]:
            self.state3_memory[index] = state
            self.new_state3_memory[index] = state_
            self.reward3_memory[index] = reward
            self.action3_memory[index] = action
            self.terminal3_memory[index] = terminal
            self.mem_cntr += 1
        else:
            print('error')

    def get_greedy(self, observation, action_set, player):
        if len(observation) == self.input_dims1[0]:
            temp = np.zeros((1, *self.input_dims1),
                                          dtype=np.float32)
            temp[0] = observation
            state = T.tensor(temp).to(self.Q_eval.device)
            actions = self.Q_eval.forward1(state)
            if player == 0:
                action = T.argmax(actions).item()
            else:
                action = T.argmin(actions).item()
            #action += 65
            #action = chr(action)
        elif len(observation) == self.input_dims2[0]:
            temp = np.zeros((1, *self.input_dims2),
                            dtype=np.float32)
            temp[0] = observation
            state = T.tensor(temp).to(self.Q_eval.device)
            actions = self.Q_eval.forward2(state)
            if player == 0:
                action = T.argmax(actions).item()
            else:
                action = T.argmin(actions).item()
            #if action == 0:
            #        action = 'N'
            #elif action == 1:
            #        action = 'S'
            #elif action == 2:
            #        action = 'E'
            #elif action == 3:
            #        action = 'W'

        elif len(observation) == self.input_dims3[0]:
            temp = np.zeros((1, *self.input_dims3),
                            dtype=np.float32)
            temp[0] = observation
            state = T.tensor(temp).to(self.Q_eval.device)
            actions = self.Q_eval.forward3(state)
            if player == 0:
                action = T.argmax(actions).item()
            else:
                action = T.argmin(actions).item()
            #action = (action, False)
        else:
            print('error')

        return action

    def choose_action(self, observation, action_set, player):
        if np.random.random() > self.epsilon:
            action = self.get_greedy(observation, action_set, player)
            if action not in action_set:
                idx = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
                action = action_set[idx]
        else:
            idx = np.random.choice(T.tensor(list(range(len(action_set)))).to(self.Q_eval.device))
            action = action_set[idx]
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

        new_state1_batch = T.tensor(
            self.new_state1_memory[batch]).to(self.Q_eval.device)
        new_state2_batch = T.tensor(
            self.new_state2_memory[batch]).to(self.Q_eval.device)
        new_state3_batch = T.tensor(
            self.new_state3_memory[batch]).to(self.Q_eval.device)
        action1_batch = self.action1_memory[batch]
        action2_batch = self.action2_memory[batch]
        action3_batch = self.action3_memory[batch]

        reward1_batch = T.tensor(
            self.reward1_memory[batch]).to(self.Q_eval.device)
        reward2_batch = T.tensor(
            self.reward2_memory[batch]).to(self.Q_eval.device)
        reward3_batch = T.tensor(
            self.reward3_memory[batch]).to(self.Q_eval.device)
        terminal1_batch = T.tensor(
            self.terminal1_memory[batch]).to(self.Q_eval.device)
        terminal2_batch = T.tensor(
            self.terminal2_memory[batch]).to(self.Q_eval.device)
        terminal3_batch = T.tensor(
            self.terminal3_memory[batch]).to(self.Q_eval.device)

        q_eval1 = self.Q_eval.forward1(state1_batch)[batch_index, action1_batch]
        q_eval2 = self.Q_eval.forward2(state2_batch)[batch_index, action2_batch]
        q_eval3 = self.Q_eval.forward3(state3_batch)[batch_index, action3_batch]

        # Using the next Q function
        q_next1 = self.Q_tgt.forward2(new_state1_batch)#self.Q_eval.forward2(new_state1_batch)
        q_next2 = self.Q_tgt.forward3(new_state2_batch)
        q_next3 = self.Q_tgt.forward1(new_state3_batch)

        q_next1[terminal1_batch] = 0.0
        q_next2[terminal2_batch] = 0.0
        q_next3[terminal3_batch] = 0.0

        # Need to be checked
        q_target1 = reward1_batch + self.gamma * T.max(q_next1, dim=1)[0]
        q_target2 = reward2_batch + self.gamma * T.min(q_next2, dim=1)[0]
        q_target3 = reward3_batch + self.gamma * T.min(q_next3, dim=1)[0]

        loss1 = self.Q_eval.loss(q_target1, q_eval1).to(self.Q_eval.device)
        loss2 = self.Q_eval.loss(q_target2, q_eval2).to(self.Q_eval.device)
        loss3 = self.Q_eval.loss(q_target3, q_eval3).to(self.Q_eval.device)

        loss1.backward()
        loss2.backward()
        loss3.backward()

        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
