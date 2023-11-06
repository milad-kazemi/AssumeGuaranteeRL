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

SYNC_TARGET_FRAMES = 1
def running(itr):
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    device = torch.device('cpu')
    constants = load_constants('constants/constants.json')
    #id = 'eval_0'
    #env = IntersectionsEnv(constants, device, id, False, get_net_path(constants))
    env = Env()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions1=81, n_actions2=2, n_actions3=81, eps_end=0.1,
                  input_dims1=[9], input_dims2=[21], input_dims3=[23], lr=0.000001)
    scores, eps_history = [], []
    n_games = 600

    for i in range(n_games):
        score = 0
        done = False
        observation, action_set, player = env.reset()
        while not done:
            action = agent.choose_action(observation, action_set, player)
            observation_, reward, done, action_set, player = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                    observation_, done)
            if i % SYNC_TARGET_FRAMES == 0:
                agent.Q_tgt.load_state_dict(agent.Q_eval.state_dict())
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        if i%100==0:
            path_str = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\scale-up\\traffic-light-env-1x1---3x3simulation\\saved-models\\iter{}-episode {}.".format(itr, i)
            T.save(agent, path_str)


        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        writer.add_scalar('training loss', score, i)
        writer.add_scalar('average reward', avg_score, i)

    #x = [i+1 for i in range(n_games)]
    writer.close()

if __name__ == '__main__':
    itr = 100
    running(itr)


