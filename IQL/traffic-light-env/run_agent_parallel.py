from models.ppo_model import NN_Model
from utils.utils import *
from environments.intersections import IntersectionsEnv, PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE
from copy import deepcopy
from workers.ppo_worker import PPOWorker
from workers.rule_worker import RuleBasedWorker
from math import pow
from utils.net_scrape import get_intersection_neighborhoods


# target for multiprocess
def train_worker(id, shared_NN, data_collector, optimizer, rollout_counter, constants, device, max_neighborhood_size):
    s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size, constants)
    env = IntersectionsEnv(constants, device, id, False, get_net_path(constants))
    # Assumes PPO worker
    local_NN = NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'], device).to(device)
    worker = PPOWorker(constants, device, env, None, shared_NN, local_NN, optimizer, id)
    train_step = 0
    while rollout_counter.get() < constants['episode']['num_train_rollouts'] + 1:
        worker.train_rollout(train_step)
        rollout_counter.increment()
    # Kill connection to sumo server
    worker.env.connection.close()
    # print('...Training worker {} done'.format(id))


# target for multiprocess
def eval_worker(id, shared_NN, data_collector, rollout_counter, constants, device, max_neighborhood_size):
    s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size, constants)
    env = IntersectionsEnv(constants, device, id, True, get_net_path(constants))#This is important
    # I think you say this is the preprint and in the arxiv page you say it is accepted as a full paper in this conference
    # Assumes PPO worker
    local_NN = NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'], device).to(device)
    worker = PPOWorker(constants, device, env, data_collector, shared_NN, local_NN, None, id)
    last_eval = 0
    while True:
        curr_r = rollout_counter.get()
        if curr_r % constants['episode']['eval_freq'] == 0 and last_eval != curr_r:
            last_eval = curr_r
            worker.eval_episodes(curr_r, model_state=worker.NN.state_dict())
        # End the eval agent
        if curr_r >= constants['episode']['num_train_rollouts'] + 1:
            break
    # Eval at end
    worker.eval_episodes(curr_r, model_state=worker.NN.state_dict())
    # Kill connection to sumo server
    worker.env.connection.close()
    # print('...Eval worker {} done'.format(id))


# target for multiprocess
def test_worker(id, ep_counter, constants, device, worker=None, data_collector=None, shared_NN=None, max_neighborhood_size=None):
    # assume PPO agent if agent=None
    if not worker:
        s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size, constants)
        env = IntersectionsEnv(constants, device, id, True, get_net_path(constants))
        local_NN = NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'], device).to(device)
        worker = PPOWorker(constants, device, env, data_collector,
                         shared_NN, local_NN, None, id)
    while ep_counter.get() < constants['episode']['test_num_eps']:
        worker.eval_episodes(None, ep_count=ep_counter.get())
        ep_counter.increment(constants['episode']['eval_num_eps'])
    # Kill connection to sumo server
    worker.env.connection.close()
    # print('...Testing agent {} done'.format(id))


# ======================================================================================================================


def train_PPO(constants, device, data_collector):
    _, max_neighborhood_size = get_intersection_neighborhoods(get_net_path(constants))
    s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size, constants)
    shared_NN = NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'], device).to(device)
    shared_NN.share_memory()
    optimizer = torch.optim.Adam(shared_NN.parameters(), constants['ppo']['learning_rate'])
    rollout_counter = Counter()  # To keep track of all the rollouts amongst agents
    processes = []
    # Run eval agent
    id = 'eval_0'
    p = mp.Process(target=eval_worker, args=(id, shared_NN, data_collector, rollout_counter, constants, device, max_neighborhood_size))
    p.start()
    processes.append(p)
    # Run training agents
    for i in range(constants['parallel']['num_workers']):
        id = 'train_'+str(i)
        p = mp.Process(target=train_worker, args=(id, shared_NN, data_collector, optimizer, rollout_counter, constants, device, max_neighborhood_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def test_PPO(constants, device, data_collector, loaded_model):
    _, max_neighborhood_size = get_intersection_neighborhoods(get_net_path(constants))
    s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size, constants)
    shared_NN = NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'], device).to(device)
    shared_NN.load_state_dict(loaded_model)
    shared_NN.share_memory()
    ep_counter = Counter()  # Eps across all agents
    processes = []
    for i in range(constants['parallel']['num_workers']):
        id = 'test_'+str(i)
        p = mp.Process(target=test_worker, args=(id, ep_counter, constants, device, None, data_collector, shared_NN, max_neighborhood_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


# verbose means test prints at end of each batch of eps
def test_rule_based(constants, device, data_collector):
    _, max_neighborhood_size = get_intersection_neighborhoods(get_net_path(constants))
    # Check rule set
    rule_set_class = get_rule_set_class(constants['rule']['rule_set'])
    ep_counter = Counter()  # Eps across all agents
    processes = []
    for i in range(constants['parallel']['num_workers']):
        id = 'test_'+str(i)
        env = IntersectionsEnv(constants, device, id, True, get_net_path(constants))
        rule_set_params = deepcopy(constants['rule']['rule_set_params'])
        rule_set_params['phases'] = env.phases
        worker = RuleBasedWorker(constants, device, env, rule_set_class(rule_set_params, get_net_path(constants), constants), data_collector, id)
        p = mp.Process(target=test_worker, args=(id, ep_counter, constants, device, worker, None, None, max_neighborhood_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
