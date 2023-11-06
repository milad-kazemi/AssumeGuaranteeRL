import time
import numpy as np
import torch
import torch.nn as nn
from utils.utils import Storage, tensor, random_sample, ensure_shared_grads
from workers.worker import Worker


# Code adapted from: Shangtong Zhang (https://github.com/ShangtongZhang)
class PPOWorker(Worker):
    def __init__(self, constants, device, env, data_collector, shared_NN, local_NN, optimizer, id, dont_reset=False):
        super(PPOWorker, self).__init__(constants, device, env, id, data_collector)
        self.NN = local_NN
        self.shared_NN = shared_NN
        if not dont_reset:  # for the vis agent script this messes things up
            self.state = self.env.reset()
        self.ep_step = 0
        self.opt = optimizer
        self.num_agents = len(env.intersections) if not self.constants['agent']['single_agent'] else 1
        self.NN.eval()

    def _reset(self):
        pass

    def _get_prediction(self, states, actions=None, ep_step=None):
        return self.NN(tensor(states, self.device), actions)

    def _get_action(self, prediction):
        return prediction['a'].cpu().numpy()

    def _copy_shared_model_to_local(self):
        self.NN.load_state_dict(self.shared_NN.state_dict())

    def _stack(self, val):
        assert not isinstance(val, list)
        return np.stack([val] * self.num_agents)

    def train_rollout(self, total_step):
        storage = Storage(self.constants['episode']['rollout_length'])
        state = np.copy(self.state)
        step_times = []
        # Sync.
        self._copy_shared_model_to_local()
        rollout_amt = 0  # keeps track of the amt in the current rollout storage
        while rollout_amt < self.constants['episode']['rollout_length']:
            start_step_time = time.time()
            prediction = self._get_prediction(state)
            action = self._get_action(prediction)
            next_state, reward, done = self.env.step(action, self.ep_step, get_global_reward=False)
            self.ep_step += 1
            if done:
                # Sync local model with shared model at start of each ep
                self._copy_shared_model_to_local()
                self.ep_step = 0

            # This is to stabilize learning, since the first some amt of states wont represent the env very well
            # since it will be more empty than normal
            if self.ep_step > self.constants['episode']['warmup_ep_steps']:
                storage.add(prediction)
                storage.add({'r': tensor(reward, self.device).unsqueeze(-1),
                             'm': tensor(self._stack(1 - done), self.device).unsqueeze(-1),
                             's': tensor(state, self.device)})
                rollout_amt += 1
            state = np.copy(next_state)

            total_step += 1

            end_step_time = time.time()
            step_times.append(end_step_time - start_step_time)

        self.state = np.copy(state)

        prediction = self._get_prediction(state)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((self.num_agents, 1)), self.device)
        returns = prediction['v'].detach()
        for i in reversed(range(self.constants['episode']['rollout_length'])):
            # Disc. Return
            returns = storage.r[i] + self.constants['ppo']['discount'] * storage.m[i] * returns
            # GAE
            td_error = storage.r[i] + self.constants['ppo']['discount'] * storage.m[i] * storage.v[i + 1] - storage.v[i]
            advantages = advantages * self.constants['ppo']['gae_tau'] * self.constants['ppo']['discount'] * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()


        states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        # Train
        self.NN.train()
        batch_times = []
        train_pred_times = []
        for _ in range(self.constants['ppo']['optimization_epochs']):
            # Sync. at start of each epoch
            self._copy_shared_model_to_local()
            sampler = random_sample(np.arange(states.size(0)), self.constants['ppo']['minibatch_size'])
            for batch_indices in sampler:
                start_batch_time = time.time()

                batch_indices = tensor(batch_indices, self.device).long()

                # Important Node: these are tensors but dont have a grad
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                start_pred_time = time.time()
                prediction = self._get_prediction(sampled_states, sampled_actions)
                end_pred_time = time.time()
                train_pred_times.append(end_pred_time - start_pred_time)

                # Calc. Loss
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()

                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.constants['ppo']['ppo_ratio_clip'],
                                          1.0 + self.constants['ppo']['ppo_ratio_clip']) * sampled_advantages

                # policy loss and value loss are scalars
                policy_loss = -torch.min(obj, obj_clipped).mean() - self.constants['ppo']['entropy_weight'] * prediction['ent'].mean()

                value_loss = self.constants['ppo']['value_loss_coef'] * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                if self.constants['ppo']['clip_grads']:
                    nn.utils.clip_grad_norm_(self.NN.parameters(), self.constants['ppo']['gradient_clip'])
                ensure_shared_grads(self.NN, self.shared_NN)
                self.opt.step()
                end_batch_time = time.time()
                batch_times.append(end_batch_time - start_batch_time)
        self.NN.eval()
        return total_step
