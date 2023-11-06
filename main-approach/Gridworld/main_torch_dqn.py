# Importing necessary modules and classes
from simple_dqn_torch import Agent  # Import the Agent class from the DQN implementation
from grid_onehot import Env        # Import the Environment class
from grid_onehot import Subsystem1 # Subsystem1 not used directly in this script, but likely part of the environment setup
from grid_onehot import Subsystem2 # Subsystem2 not used directly either, same as above
from grid_onehot import Automaton1 # Automaton1 is probably a component of the environment
from grid_onehot import Automaton2 # Automaton2 as well
import matplotlib as plt           # Matplotlib for plotting (plt is not used in this script though)
import numpy as np                 # Numpy for numerical operations
import torch as T                  # PyTorch imported as T for operations on tensors
from torch.utils.tensorboard import SummaryWriter # For logging data for visualization in TensorBoard

writer = SummaryWriter("runs")  # Initialize a TensorBoard SummaryWriter to log data for visualization

SYNC_TARGET_FRAMES = 50  # Frequency of synchronization between the target and evaluation networks

# Function to run the training process
def running(itr):
    env = Env()  # Instantiate the environment
    # Instantiate the agent with specified parameters
    agent = Agent(gamma=1, epsilon=1, batch_size=64, n_actions1=6, n_actions2=4, n_actions3=12, eps_end=0.2,
                  input_dims1=[20], input_dims2=[26], input_dims3=[30], lr=0)
    scores, eps_history = [], []  # Lists to store scores and epsilon history for analysis
    n_games = 501  # Number of games to play
    
    for i in range(n_games):  # Main loop over the number of games
        score = 0  # Initialize the score for the game
        done = False  # Initialize 'done' flag for the game loop
        observation, action_set, player = env.reset()  # Reset the environment and get the initial state

        while not done:  # Loop until the game is finished
            action = agent.choose_action(observation, action_set, player)  # Agent selects an action
            # Environment steps forward and returns new observations and reward
            observation_, reward, done, action_set, player = env.step(action)
            score += reward  # Update the score with the reward
            # Store the transition in the agent's memory
            agent.store_transition(observation, action, reward, observation_, done)
            
            if i % SYNC_TARGET_FRAMES == 0:  # Synchronize the target and evaluation networks at specified intervals
                agent.Q_tgt.load_state_dict(agent.Q_eval.state_dict())

            agent.learn()  # Agent learns from the stored transition
            observation = observation_  # Update the observation

        scores.append(score)  # Append the score for the game to the scores list
        eps_history.append(agent.epsilon)  # Append the current epsilon to the history

        avg_score = np.mean(scores[-100:])  # Calculate the average score over the last 100 games

        if i%100==0:  # Every 100 episodes, save the model
            path_str = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\Gridworld\\saved-models\\itr{}_episode{}.".format(itr, i)
            T.save(agent, path_str)  # Save the agent's state

        # Print out the episode stats
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        # Log training loss and average reward in TensorBoard
        writer.add_scalar('training loss', score, i)
        writer.add_scalar('average reward', avg_score, i)

    x = [i+1 for i in range(n_games)]  # Generate a sequence of game numbers (not used in this script)

# Entry point of the script
if __name__ == '__main__':
    for itr in range(5):  # Run the running function 5 times (5 iterations)
        running(itr)  # Call the running function with the current iteration number

