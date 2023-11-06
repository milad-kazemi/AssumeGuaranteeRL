from simple_dqn_torch import Agent
from grid_onehot import Env
from grid_onehot import Subsystem1
from grid_onehot import Subsystem2
from grid_onehot import Automaton1
from grid_onehot import Automaton2
import matplotlib as plt
import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs")

SYNC_TARGET_FRAMES = 20
def running(itr):
    env = Env()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions1=6, n_actions2=4, n_actions3=12, eps_end=0.01,
                  input_dims1=[16], input_dims2=[26], input_dims3=[30], lr=0.000001)
    scores, eps_history = [], []
    n_games = 501
    
    for i in range(n_games):
        score1 = 0
        score2 = 0

        done = False
        observation, action_set, player = env.reset()
        while not done:
            action = agent.choose_action(observation, action_set, player)
            observation_, reward, done, action_set, player = env.step(action)
            score1 += reward[0]
            score2 += reward[1]

            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            if i % SYNC_TARGET_FRAMES == 0:
                agent.Q_tgt.load_state_dict(agent.Q_eval.state_dict())
            agent.learn()
            observation = observation_
        scores.append(score1)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        if (score1==1 and score2==1):
            path_str = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\IQL\\Gridworld\\saved-models\\iter{}-episode {}.".format(itr, i)
            T.save(agent, path_str)


        print('episode ', i, 'satisfaction %.2f' %score1==1 and score2==1, 'score1 %.2f' % score1,
              'score2 %.2f' % score2,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        writer.add_scalar('training loss1', score1, i)
        writer.add_scalar('training loss2', score1, i)
        writer.add_scalar('average reward', avg_score, i)
    x = [i+1 for i in range(n_games)]

if __name__ == '__main__':
    for itr in range(20):
        running(itr)
