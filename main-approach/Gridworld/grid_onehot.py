import numpy as np
# A simple environment:
# Two agents on a 4 x 3 grid world. Their goal is to swap places
# while staying on the left side of the arena from their perspective.
# Each column is labeled "A" "B" "C" from left to right.
# Movement is deterministic. States are numbered from left to right, top down.

# The grid:
# 0 | 1  | 2
# 3 | 4  | 5
# 6 | 7  | 8
# 9 | 10 | 11
#
# Agent 1 starts in 10
# Agent 2 starts in 1

# A = {0,3,6,9} # Avoid for agent 2
# B = {4,7}
# C = {2,5,8,11} # Avoid for agent 1
# D = {1} # Goal for agent 1; agent 2 avoids after first timestep; agent 1 avoids on first timestep
# E = {10} # Goal for agent 2; agent 1 avoids after first timestep; agent 2 avoids on first timestep
# F = agents collided # Avoid for agents 1 and 2

# Automaton for agent 1
class Automaton1():
    def __init__(self):
        self.n_s = 4

    def onehot2index(self, onehot):
        index = np.where(onehot == 1)[0][0]
        return index

    def index2onehot(self, index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    def reset(self):
        q_i = 0
        self.q = self.index2onehot(q_i, self.n_s)
        return self.q  # State, isAccepting

    def step(self, label):
        q_i = self.onehot2index(self.q)
        label = self.onehot2index(label)
        if q_i == 0:
            if label == 2 or label == 5 or label == 3:  # Go to trap
                q_i = 3
            else:
                q_i = 1
        elif q_i == 1:
            if label == 2 or label == 5 or label == 4:
                q_i = 3
            elif label == 3:  # Accept
                q_i = 2
        elif q_i == 2:
            if label == 2 or label == 5 or label == 4:
                q_i = 3
        elif q_i == 3:
            q_i = 3
        self.q = self.index2onehot(q_i, self.n_s)
        return self.q


# Automaton for agent 2
class Automaton2():
    def __init__(self):
        self.n_s = 4

    def onehot2index(self, onehot):
        index = np.where(onehot == 1)[0][0]
        return index

    def index2onehot(self, index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    def reset(self):
        q_i = 0
        self.q = self.index2onehot(q_i, self.n_s)
        return self.q  # State, isAccepting

    def step(self, label):
        q_i = self.onehot2index(self.q)
        label = self.onehot2index(label)
        if q_i == 0:
            if label == 0 or label == 5 or label == 4:  # Go to trap
                q_i = 3
            else:
                q_i = 1
        elif q_i == 1:
            if label == 0 or label == 5 or label == 3:
                q_i = 3
            elif label == 4:  # Accept
                q_i = 2
        elif q_i == 2:
            if label == 0 or label == 5 or label == 3:
                q_i = 3
        self.q = self.index2onehot(q_i, self.n_s)
        return self.q


# Subsystem for agent 1
# Its state is (grid position, has collided)
class Subsystem1():
    def __init__(self):
        self.n_s = 12

    def onehot2index(self, onehot):
        index = np.where(onehot == 1)[0][0]
        return index

    def index2onehot(self, index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    def getActionSet(self):
        action_set = []
        s_i = self.onehot2index(self.s)
        if s_i != 0 and s_i != 1 and s_i != 2:
            action_set.append(0)
        if s_i != 0 and s_i != 3 and s_i != 6 and s_i != 9:
            action_set.append(1)
        if s_i != 9 and s_i != 10 and s_i != 11:
            action_set.append(2)
        if s_i != 2 and s_i != 5 and s_i != 8 and s_i != 11:
            action_set.append(3)
        return action_set

    def reset(self):
        s_i = 10
        self.s = self.index2onehot(s_i, self.n_s)
        return self.s, self.getActionSet()

    def step(self, action, state_i):
        # I don't check if the action selected is valid in this implementation...
        # The directions are North East South West.
        s_i = self.onehot2index(state_i) #need to be checked

        #need to be checked
        #if self.s == state_i[0]:
        #    self.collided = True

        if action == 0:
            s_i = s_i - 3
        elif action == 3:
            s_i = s_i + 1
        elif action == 2:
            s_i = s_i + 3
        elif action == 1:
            s_i = s_i - 1

        self.s = self.index2onehot(s_i, self.n_s)

        return self.s, self.getActionSet()

    # Subsystem for agent 2


# Its state is (grid position, has collided)
class Subsystem2():
    def __init__(self):
        self.n_s = 12

    def onehot2index(self, onehot):
        index = np.where(onehot == 1)[0][0]
        return index

    def index2onehot(self, index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    def getActionSet(self):
        action_set = []
        s_i = self.onehot2index(self.s)
        if s_i != 0 and s_i != 1 and s_i != 2:
            action_set.append(0)
        if s_i != 0 and s_i != 3 and s_i != 6 and s_i != 9:
            action_set.append(1)
        if s_i != 9 and s_i != 10 and s_i != 11:
            action_set.append(2)
        if s_i != 2 and s_i != 5 and s_i != 8 and s_i != 11:
            action_set.append(3)
        return action_set

    def reset(self):
        s_i = 1
        self.s = self.index2onehot(s_i, self.n_s)
        return self.s, self.getActionSet()

    def step(self, action, state_i):
        # I don't check if the action selected is valid in this implementation...
        # The directions are North East South West.
        s_i = self.onehot2index(self.s)

        #need to be checked
        #if self.s == state_i[0]:
        #    self.collided = True

        if action == 0:
            s_i = s_i - 3
        elif action == 3:
            s_i = s_i + 1
        elif action == 2:
            s_i = s_i + 3
        elif action == 1:
            s_i = s_i - 1

        self.s = self.index2onehot(s_i, self.n_s)

        return self.s, self.getActionSet()

    # Subsystem for agent 2






class Env():
    def __init__(self):
        self.automaton1 = Automaton1()
        self.automaton2 = Automaton2()
        self.subsystem1 = Subsystem1()
        self.max_action_set = None
        self.T = 6  # Horizon
        self.n_l = 6
        self.n_s = 11
        self.n_a_1 = 6
        self.n_a_2 = 4
        self.n_a_3 = 12
    def onehot2index(self, onehot):
        index = np.where(onehot == 1)[0][0]
        return index

    def index2onehot(self, index, n):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    def allLabels(self):
        return [0, 1, 2, 3, 4]

    def getLabel(self, state):
        #if state[1]:
        #    return 5
        x = self.onehot2index(state)
        if x % 3 == 0:
            l=0
            return self.index2onehot(l, self.n_l)
        elif x % 3 == 2:
            l=2
            return self.index2onehot(l, self.n_l)
        elif x == 1:
            l=3
            return self.index2onehot(l, self.n_l)
        elif x == 10:
            l=4
            return self.index2onehot(l, self.n_l)
        else:
            l=1
            return self.index2onehot(l, self.n_l)

    def getStates(self, label):
        #label = self.onehot2index(self, label)
        if label == 5:
            return [x for x in range(12)]
        if label == 0:
            a = [0, 3, 6, 9]
            return [x for x in a]
        elif label == 1:
            a = [4, 7]
            return [x for x in a]
        elif label == 2:
            a = [2, 5, 8, 11]
            return [x for x in a]
        elif label == 3:
            a = [1]
            return [x for x in a]
        elif label == 4:
            a = [10]
            return [x for x in a]

    def reset(self):
        self.t = 0
        self.player = 1  # 0 for maximizing player, 1 for minimizing
        sub_state, max_action_set = self.subsystem1.reset()
        self.max_action_set = max_action_set
        auto_state1 = self.automaton1.reset()
        auto_state2 = self.automaton2.reset() #auto_state2, _
        self.state = np.concatenate((sub_state, auto_state1, auto_state2), axis=0)
        action_set = self.allLabels()  # Labels
        return self.state, action_set, self.player

    def step(self, action):
        action_set = None
        acc1 = None
        acc2 = None
        rej1 = None
        rej2 = None

        reward = 0
        if len(self.state) == 20:  # min label selected
            self.player = 0
            action = self.index2onehot(action, self.n_a_1)
            self.state = np.concatenate((self.state, action), axis=0)#self.state + (action,)  # Append label to state
            action_set = self.max_action_set
        elif len(self.state) == 26:  # max action selected
            self.player = 1
            action_set = self.getStates(self.onehot2index(self.state[20:]))  # Get states in label
            action = self.index2onehot(action, self.n_a_2)
            self.state = np.concatenate((self.state, action), axis=0)#self.state + (action,)  # Append action to state
        elif len(self.state) == 30:  # min state selected
            self.t = self.t + 1
            temp_1 = self.state[:12]
            #print(self.onehot2index(temp_1), action)
            if self.onehot2index(temp_1)==action:
                label_colliding = self.index2onehot(5, self.n_l)
            else:
                label_colliding = temp_1
            auto_state1 = self.automaton1.step(self.getLabel(label_colliding))
            ll1 = self.onehot2index(temp_1)
            ll2 = action
            #temp_2 = self.state[16:20]
            auto_state2 = self.automaton2.step(self.getLabel(self.index2onehot(action, 12)))
            #print(self.onehot2index(auto_state2))

            acc1 = self.onehot2index(auto_state1) == 2
            rej1 = self.onehot2index(auto_state1) == 3
            acc2 = self.onehot2index(auto_state2) == 2
            rej2 = self.onehot2index(auto_state2) == 3

            temp_3 = self.state[26:30]
            sub_state, max_action_set = \
                self.subsystem1.step(self.onehot2index(temp_3), temp_1)
            self.max_action_set = max_action_set
            self.state = np.concatenate((sub_state, auto_state1, auto_state2), axis=0)
            action_set = self.allLabels()
        else:
            raise RuntimeError("unreachable")

        #reward = 0
        done = False
        reward = 1 if not rej1 else -10

        #print(acc1, rej2)
        if self.t >= self.T-1:
            if acc1:
                reward = 1
            elif not acc1 and not acc2:
                reward = -100
            else:  # Punish on last step if reward is zero
                reward = -100
            done = True

        return self.state, reward, done, action_set, self.player