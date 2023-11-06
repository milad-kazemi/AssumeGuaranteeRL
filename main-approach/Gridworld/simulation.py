from simple_dqn_torch import Agent  # Imports the Agent class for the DQN agent
from grid_onehot import Env          # Imports the Environment class for the simulation
# Additional classes likely represent different components or entities within the simulation
from grid_onehot import Subsystem1   
from grid_onehot import Subsystem2   
from grid_onehot import Automaton1   
from grid_onehot import Automaton2   
import matplotlib as plt             # Matplotlib for plotting capabilities (plt is not used in this script)
import os                            # os module for operating system dependent functionality
import numpy as np                   # Numpy library for numerical operations
import torch as T                    # PyTorch library for tensor computations
from torch.utils.tensorboard import SummaryWriter  # For logging data for visualization in TensorBoard
writer = SummaryWriter("runs")  # Initialize a TensorBoard SummaryWriter to log data for visualization

# Utility functions for one-hot encoding transformations
def onehot2index(onehot):
    """Converts a one-hot encoded vector to its corresponding index."""
    index = np.where(onehot == 1)[0][0]
    return index

def index2onehot(index, n):
    """Converts an index to a one-hot encoded vector of length n."""
    onehot = np.zeros(n)
    onehot[index] = 1
    return onehot

def getLabel(state):
    """Generates a label for the given state."""
    n_l = 6
    x = onehot2index(state)
    if x % 3 == 0:
        l = 0
    elif x % 3 == 2:
        l = 2
    elif x == 1:
        l = 3
    elif x == 10:
        l = 4
    else:
        l = 1
    return index2onehot(l, n_l)

flip = {0: 2, 2: 0, 1: 3, 3: 1}  # Dictionary to flip actions, used to determine symmetrical actions

# Main execution block
if __name__ == '__main__':
    # Initialization of subsystems and automata
    subsystem1 = Subsystem1()
    subsystem2 = Subsystem2()
    automaton1 = Automaton1()
    automaton2 = Automaton2()
    n = 1  # Number of simulations
    nt = 6  # Number of steps in each simulation
    SMC = np.zeros(n)  # Array to hold the results of the simulations
    i_1 = 0  # Counter for simulations
    # Directory where saved models are located
    directory = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\Gridworld\\saved-models"
    
    # Loop through each saved model in the directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        print(f)  # Print the file path for debugging or logging
        agent = T.load(f)  # Load the saved agent model
        stat_avg = 0  # Initialize the statistic average
        
        # Perform simulations
        for i in range(0, n):
            # Reset states and labels for subsystems and automata
            state1, action_set1 = subsystem1.reset()
            state2, action_set2 = subsystem2.reset()
            auto_state1 = automaton1.reset()
            auto_state2 = automaton2.reset()
            label1 = getLabel(state1)
            label2 = getLabel(state2)

            # Simulation steps
            for t in range(0, nt):
                S = np.zeros((4, 3))  # Initialize state matrix

                # Update state matrix with positions of subsystems
                S[onehot2index(state1) // 3, onehot2index(state1) % 3] = 1
                S[onehot2index(state2) // 3, onehot2index(state2) % 3] = 2
                
                # Concatenate state information for the environment
                env_state = np.concatenate((state1, auto_state1, auto_state2, label2), axis=0)
                
                # Get the action from the agent
                act1 = agent.get_greedy(env_state, action_set1, 0)
                # Determine the symmetrical action for the second subsystem
                act2 = flip[act1]
                
                # Step the automata and subsystems forward
                auto_state1 = automaton1.step(getLabel(state1))
                auto_state2 = automaton2.step(getLabel(state2))
                next_state1, action_set1 = subsystem1.step(act1, state1)
                next_state2, action_set2 = subsystem2.step(act2, state2)
                
                # Update states and labels
                state1, state2 = next_state1, next_state2
                label1 = getLabel(state1)
                label2 = getLabel(state2)
            
            # Update statistic based on the final state of automaton1
            stat_avg += int(onehot2index(auto_state1) == 2)

        # Calculate the average of the statistic and print it
        stat_avg = stat_avg / n
        print(stat_avg)
        # Optionally: Log the statistic to TensorBoard
        #writer.add_scalar('SMC200', stat_avg, i_1)
        i_1 += 1  # Increment simulation counter

