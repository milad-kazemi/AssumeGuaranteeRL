from simple_dqn_torch import Agent
from room_temp import Env
from room_temp import Subsystem1
from room_temp import Automaton1
import matplotlib as plt
import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs")


SYNC_TARGET_FRAMES = 20
def running(itr):
    env = Env()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions1=8, n_actions2=5, n_actions3=17, eps_end=0.01,
                  input_dims1=[4], input_dims2=[15], input_dims3=[20], lr=0.000001)
    scores, eps_history = [], []
    n_games = 501
    
    for i in range(n_games):
        score1 = 0
        score2 = 0
        #score3 = 0
        #score4 = 0
        done = False
        observation, action_set, player = env.reset()
        while not done:
            action = agent.choose_action(observation, action_set, player)
            observation_, reward, done, action_set, player = env.step(action)
            score1 += reward[0]
            score2 += reward[1]
        #    score3 += reward[2]
        #    score4 += reward[3]

            agent.store_transition(observation, action, reward,
                                    observation_, done)
            if i % SYNC_TARGET_FRAMES == 0:
                agent.Q_tgt.load_state_dict(agent.Q_eval.state_dict())
            agent.learn()
            observation = observation_
        scores.append(score1)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        if i%100==0:
            path_str = "C:\\Users\\Admin\\PycharmProjects\\python-ppo\\IQL\\room-temp\\saved-models\\iter{}-episode {}.".format(itr, i)
            T.save(agent, path_str)


        print('episode ', i, 'score1 %.2f' % score1,
              'score2 %.2f' % score2,
              #'score3 %.2f' % score3,
              #'score4 %.2f' % score4,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        writer.add_scalar('training loss1', score1, i)
        writer.add_scalar('training loss2', score2, i)
        #writer.add_scalar('training loss3', score3, i)
        #writer.add_scalar('training loss4', score4, i)
        writer.add_scalar('average reward1', avg_score, i)
    x = [i+1 for i in range(n_games)]
    writer.close()


if __name__ == '__main__':
    for itr in range(5, 20):
        running(itr)