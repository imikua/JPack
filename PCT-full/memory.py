# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import jittor as jt


Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))


class Value():
    def __init__(self, value):
        self.value = value


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size, obs_len, args):
        self.distributed = False
        self.index = Value(0)
        self.full = Value(False)

        self.size = size
        self.max = 1

        self.sum_tree = jt.zeros((2 * size - 1,), dtype=jt.float32)
        self.timesteps = jt.zeros((size, 1), dtype=jt.int32)
        self.states = jt.zeros((size, obs_len), dtype=jt.float32)
        self.actions = jt.zeros((size, 1), dtype=jt.int32)
        self.rewards = jt.zeros((size, 1), dtype=jt.float32)
        self.nonterminals = jt.zeros((size, 1), dtype=jt.bool)

    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    def update(self, index, value):
        self.sum_tree[index] = value
        self._propagate(index, value)
        self.max = max(float(value), float(self.max))

    def append(self, data, value):
        index = self.index.value
        self.timesteps[index] = data[0]
        self.states[index] = data[1]
        self.actions[index] = data[2]
        self.rewards[index] = data[3]
        self.nonterminals[index] = data[4]
        self.update(index + self.size - 1, value)
        self.index.value = (index + 1) % self.size
        self.full.value = self.full.value or self.index.value == 0
        self.max = max(float(value), float(self.max))

    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= float(self.sum_tree[left].item()):
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - float(self.sum_tree[left].item()))

    def find(self, value):
        index = self._retrieve(0, value)
        data_index = index - self.size + 1
        return float(self.sum_tree[index].item()), data_index, index

    def getBatch(self, data_indexs):
        data_indexs = np.asarray(data_indexs) % self.size
        index_var = jt.array(data_indexs)
        return self.timesteps[index_var], self.states[index_var], self.actions[index_var], self.rewards[index_var], self.nonterminals[index_var]

    def total(self):
        return float(self.sum_tree[0].item())


class ReplayMemory():
    def __init__(self, args, capacity, obs_len):
        self.device = args.device
        self.capacity = capacity
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_weight = args.priority_weight
        self.priority_exponent = args.priority_exponent
        self.t = 0
        self.transitions = SegmentTree(capacity, obs_len, args)
        self.max_resample_attempts = 256
        self.blank_trans = [
            jt.array([0], dtype=jt.int32),
            jt.zeros((obs_len,), dtype=jt.float32),
            jt.array([0], dtype=jt.int32),
            jt.array([0.0], dtype=jt.float32),
            jt.array([False], dtype=jt.bool)
        ]
        self.n_step_scaling = jt.array([self.discount ** i for i in range(self.n)], dtype=jt.float32)

    def append(self, state, action, reward, terminal):
        if not isinstance(state, jt.Var):
            state = jt.array(state)
        if not isinstance(action, jt.Var):
            action = jt.array(action)
        if not isinstance(reward, jt.Var):
            reward = jt.array(reward)

        state = state.float32().reshape((-1,))
        action = action.int32().reshape((1,)) if len(action.shape) == 0 else action.int32().reshape((-1,))[:1]
        reward = reward.float32().reshape((1,)) if len(reward.shape) == 0 else reward.float32().reshape((-1,))[:1]
        nonterminal = jt.array([not terminal], dtype=jt.bool)
        timestep = jt.array([self.t], dtype=jt.int32)
        self.transitions.append(Transition(timestep, state, action, reward, nonterminal), self.transitions.max)
        self.t = 0 if terminal else self.t + 1

    def _get_transition_new(self, idx):
        timesteps, states, actions, rewards, nonterminals = [], [], [], [], []

        for t in range(0, 1 + self.n):
            if t == 0 or bool(nonterminals[-1].item()):
                timestep, state, action, reward, nonterminal = self.transitions.getBatch(idx + t)
            else:
                timestep, state, action, reward, nonterminal = self.blank_trans
            timesteps.append(timestep)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            nonterminals.append(nonterminal)

        return jt.concat(timesteps, dim=0), jt.stack(states, dim=0), jt.concat(actions, dim=0), jt.concat(rewards, dim=0), jt.concat(nonterminals, dim=0)

    def _get_transitions_batch(self, idxs):
        transition_idxs = np.arange(0, self.n + 1) + np.expand_dims(idxs, axis=1)
        index_shape = transition_idxs.shape
        timesteps, states, actions, rewards, nonterminals = self.transitions.getBatch(transition_idxs)
        transitions_firsts = (timesteps == 0).reshape(index_shape)
        blank_mask = np.zeros(index_shape, dtype=np.bool_)

        for t in range(1, 1 + self.n):
            blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t].numpy())
        blank_mask_flat = blank_mask.reshape(-1)

        timesteps_np = timesteps.numpy().reshape(-1)
        timesteps_np[blank_mask_flat] = 0
        timesteps = jt.array(timesteps_np.reshape(index_shape), dtype=jt.int32)

        states_np = states.numpy().reshape((-1, states.shape[-1]))
        states_np[blank_mask_flat] = 0
        states = jt.array(states_np.reshape((*index_shape, -1)), dtype=jt.float32)

        actions_np = actions.numpy().reshape(-1)
        actions_np[blank_mask_flat] = 0
        actions = jt.array(actions_np.reshape(index_shape), dtype=jt.int32)

        rewards_np = rewards.numpy().reshape(-1)
        rewards_np[blank_mask_flat] = 0
        rewards = jt.array(rewards_np.reshape(index_shape), dtype=jt.float32)

        nonterminals_np = nonterminals.numpy().reshape(-1)
        nonterminals_np[blank_mask_flat] = False
        nonterminals = jt.array(nonterminals_np.reshape(index_shape), dtype=jt.bool)

        return timesteps, states, actions, rewards, nonterminals

    def _is_valid_index(self, idx, prob):
        return ((self.transitions.index.value - idx) % self.capacity > self.n and
                (idx - self.transitions.index.value) % self.capacity >= 1 and
                prob != 0)

    def _fallback_valid_index(self, start_idx):
        capacity = self.capacity if self.transitions.full.value else self.transitions.index.value
        if capacity <= self.n + 1:
            raise RuntimeError('ReplayMemory does not have enough valid samples yet.')
        for offset in range(capacity):
            idx = (start_idx + offset) % capacity
            tree_idx = idx + self.transitions.size - 1
            prob = float(self.transitions.sum_tree[tree_idx].item())
            if self._is_valid_index(idx, prob):
                return prob, idx, tree_idx
        raise RuntimeError('ReplayMemory could not find a valid transition index.')

    def _get_sample_from_segment(self, segment, i):
        prob, idx, tree_idx = 0.0, 0, 0
        valid = False
        for _ in range(self.max_resample_attempts):
            sample = np.random.uniform(i * segment, (i + 1) * segment)
            prob, idx, tree_idx = self.transitions.find(sample)
            if self._is_valid_index(idx, prob):
                valid = True
                break

        if not valid:
            start_idx = int((i * segment) % max(1, self.capacity))
            prob, idx, tree_idx = self._fallback_valid_index(start_idx)

        Btimesteps, Bstates, Bactions, Brewards, Bnonterminals = self._get_transition_new(idx)

        state = Bstates[0].float32()
        next_state = Bstates[self.n].float32()
        if len(state.shape) > 1 and state.shape[0] == 1:
            state = state.squeeze(0)
        if len(next_state.shape) > 1 and next_state.shape[0] == 1:
            next_state = next_state.squeeze(0)
        action = Bactions[0].int32()
        reward = Brewards[0:-1].float32()
        R = jt.sum(reward * self.n_step_scaling)
        nonterminal = Bnonterminals[self.n].float32()

        return prob, idx, tree_idx, state, action, R, next_state, nonterminal

    def sample(self, batch_size):
        p_total = self.transitions.total()
        segment = p_total / batch_size
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)

        states, next_states = jt.stack(states), jt.stack(next_states)
        actions = jt.stack(actions)
        returns = jt.stack(returns).reshape((batch_size, 1))
        nonterminals = jt.stack(nonterminals).reshape((batch_size, 1))
        probs = np.array(probs, dtype=np.float32) / p_total
        capacity = self.capacity if self.transitions.full.value else self.transitions.index.value
        weights = (capacity * probs) ** -self.priority_weight
        weights = jt.array(weights / weights.max(), dtype=jt.float32)
        return list(tree_idxs), states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        if hasattr(priorities, 'numpy'):
            priorities = priorities.numpy()
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    def __iter__(self):
        self.current_idx = 0
        return self

