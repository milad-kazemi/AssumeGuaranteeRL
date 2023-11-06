# Assume-Guarantee RL

This project utilizes a Deep Q-Network (DQN) to design policies for three environments: gridworld, temperature control, and traffic light.

## Main Scripts

- `main_torch_dqn.py`: The primary script for initiating the learning process.
- `simulation.py`: The script to run simulations of the environment.

## Getting Started

To run the scripts, you will need to set up your environment with the required dependencies and SUMO.

### Prerequisites

Ensure you have the following packages installed:

- numpy
- traci
- sumolib
- scipy
- pytorch
- pandas

### SUMO Environment

This project requires the SUMO simulation environment to model urban mobility. SUMO is an open-source, highly portable, microscopic and continuous road traffic simulation package designed to handle large road networks.

### TraCI

[TraCI](https://sumo.dlr.de/docs/TraCI.html) is the Traffic Control Interface for SUMO, which allows for interaction with the simulation. Through TraCI, it's possible to manipulate the simulation and retrieve data using Python.

## Running the Scripts

Once all dependencies are installed, and SUMO is set up:

1. Execute `main_torch_dqn.py` to start the learning process.
2. Run `simulation.py` to initiate the simulations.
