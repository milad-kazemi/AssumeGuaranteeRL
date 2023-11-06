# This script defines and runs a simulation environment using reinforcement learning, specifically a DQN (Deep Q-Network) agent.

# Importing required classes and libraries.
from simple_dqn_torch import Agent  # Agent class implementing the DQN.
from room_temp_original import Env   # Environment class for the simulation.
from room_temp_original import Subsystem1  # Represents a subsystem within the environment.
from room_temp_original import Automaton1  # Represents an automaton within the environment.
import matplotlib as plt            # Matplotlib library for plotting (not utilized in this script).
import numpy as np                  # Numpy library for array and numerical operations.
import torch as T                   # PyTorch library for tensor computations and neural networks.
from torch.utils.tensorboard import SummaryWriter  # For logging and visualizing data in TensorBoard.
writer = SummaryWriter("runs")  # Initializes a TensorBoard SummaryWriter.

# Constants defining the frequency of synchronization between the target and evaluation networks.
SYNC_TARGET_FRAMES = 20

# Function that runs the simulation.
def running(itr):
    # Initializes the environment and the agent with specific parameters.
    env = Env()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions1=8, n_actions2=5, n_actions3=17,
                  eps_end=0.1, input_dims1=[7], input_dims2=[15], input_dims3=[20], lr=0.000001)
    
    # Lists to hold scores and epsilon history for analysis.
    scores, eps_history = [], []
    n_games = 501  # Number of games to be played in the simulation.
    
    # Main loop to run the games.
    for i in range(n_games):
        score = 0  # Initialize the score for the current game.
        done = False  # Flag to indicate if the game has ended.
        # Resets the environment and gets the initial observation, action set, and player.
        observation, action_set, player = env.reset()
        
        # Inner loop that runs until the game is finished.
        while not done:
            # Agent selects an action based on the current state.
            action = agent.choose_action(observation, action_set, player)
            # Environment takes a step based on the action and returns new state and reward.
            observation_, reward, done, action_set, player = env.step(action)
            # Update the score with the reward.
            score += reward
            # Store the transition in the agent's memory.
            agent.store_transition(observation, action, reward, observation_, done)
            
            # Synchronize the target and evaluation networks at defined intervals.
            if i % SYNC_TARGET_FRAMES == 0:
                agent.Q_tgt.load_state_dict(agent.Q_eval.state_dict())
            # Agent learns from the stored transitions.
            agent.learn()
            # Update the observation with the new state.
            observation = observation_
        
        # Append the score to the scores list and update epsilon history.
        scores.append(score)
        eps_history.append(agent.epsilon)

        # Calculate and print the average score.
        avg_score = np.mean(scores[-100:])
        if i % 100 == 0:
            # Every 100 episodes, save the model.
            path_str = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\room-temp\\saved-models\\iter{}-episode {}.".format(itr, i)
            T.save(agent, path_str)

        # Logging the score and average score.
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
        writer.add_scalar('training loss', score, i)
        writer.add_scalar('average reward', avg_score, i)
    # Close the TensorBoard writer.
    writer.close()

# Entry point of the script.
if __name__ == '__main__':
    # Loop through a range of iterations to run multiple simulations.
    for itr in range(600, 700):
        running(itr)  # Execute the running function for each iteration.

