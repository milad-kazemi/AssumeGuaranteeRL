import time
from utils.utils import *
from utils.random_search import grid_choices_random
from utils.grid_search import grid_choices, get_num_grid_choices
import sys
import os
from utils.utils import *
from environments.intersections import IntersectionsEnv, PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE
from prod_env import *


if __name__ == '__main__':
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    device = torch.device('cpu')
    constants = load_constants('constants/constants.json')

    #id = 'eval_0'
    #env = IntersectionsEnv(constants, device, id, False, get_net_path(constants))
    #env = Env()
    #next_state = env.reset()
    #next_state, reward, done, action_set, player = env.step(2)
    #np.array([1, 0, 1, 0], dtype=np.int64)
    env = Env()
    state, actions, player = env.reset()
    #state1, state2, state3, state4, action_set, player = env.reset()
    #env.step(np.array([[1, 0, 1, 0]], dtype=np.int64))

    #next_state, reward, done, action_set, player = env.step(2)
    #next_state, reward, done, action_set, player = env.step(2)
    for i in range(10):
        state, reward, done, action_set, player\
            = env.step([np.array([1, 0, 1, 0], dtype=np.int64)])


