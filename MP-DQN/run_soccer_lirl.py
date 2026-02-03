#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LIRL Agent for Soccer Environment - 优化版本
=============================================
核心特性：
1. LIRL动作投影：匈牙利算法选择离散动作 + QP投影连续参数
2. Multipass Q网络：对每个动作单独计算Q值（来自PDQN的核心优势）
3. N-step returns + Inverting Gradients
"""

import os
import click
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy
import gym
import gym_soccer
from gym.wrappers import Monitor
from common import ClickPythonLiteralOption
from common.soccer_domain import SoccerScaledParameterisedActionWrapper

from agents.memory.memory import MemoryNStepReturns
from agents.utils import soft_update_target_network, hard_update_target_network
from agents.utils.noise import OrnsteinUhlenbeckActionNoise

# 尝试导入scipy用于LIRL动作投影
try:
    from scipy.optimize import minimize, linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using numpy fallback")

device = torch.device("cpu")


# =======================
# LIRL动作投影
# =======================
class LIRLActionProjection:
    """
    LIRL动作投影：将网络输出投影到有效的离散-连续混合动作空间
    
    核心方法：
    1. 离散动作选择：匈牙利算法（最小化代价矩阵）
    2. 连续参数投影：QP求解（投影到可行域）
    """
    
    def __init__(self, num_actions, action_parameter_sizes, param_min, param_max):
        self.num_actions = num_actions
        self.action_parameter_sizes = action_parameter_sizes
        self.offsets = np.cumsum([0] + list(action_parameter_sizes))
        self.param_min = param_min
        self.param_max = param_max
    
    def project(self, action_scores, action_parameters):
        """
        LIRL动作投影
        
        Args:
            action_scores: 离散动作的分数 [num_actions]
            action_parameters: 连续动作参数 [total_param_size]
        
        Returns:
            action: 选择的离散动作
            action_params: 投影后的连续参数
        """
        # 1. 离散动作选择 - 使用匈牙利算法
        action = self._hungarian_select(action_scores)
        
        # 2. 连续参数投影 - 使用QP
        start_idx = self.offsets[action]
        end_idx = self.offsets[action + 1]
        params = action_parameters[start_idx:end_idx]
        projected_params = self._qp_project(params, start_idx, end_idx)
        
        return action, projected_params
    
    def _hungarian_select(self, action_scores):
        """使用匈牙利算法选择离散动作"""
        # 构建代价矩阵：代价 = -分数（最大化分数 = 最小化负分数）
        cost_matrix = -action_scores.reshape(1, -1)
        
        if SCIPY_AVAILABLE:
            _, col_ind = linear_sum_assignment(cost_matrix)
            return col_ind[0]
        else:
            # Fallback: argmax
            return np.argmax(action_scores)
    
    def _qp_project(self, params, start_idx, end_idx):
        """使用QP将参数投影到可行域 [min, max]"""
        param_min = self.param_min[start_idx:end_idx]
        param_max = self.param_max[start_idx:end_idx]
        
        if SCIPY_AVAILABLE:
            # QP: min ||x - params||^2 s.t. min <= x <= max
            def objective(x):
                return 0.5 * np.sum((x - params) ** 2)
            
            def gradient(x):
                return x - params
            
            bounds = [(param_min[i], param_max[i]) for i in range(len(params))]
            x0 = np.clip(params, param_min, param_max)
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                                bounds=bounds, options={'maxiter': 50})
                if result.success:
                    return result.x
            except:
                pass
        
        # Fallback: clip
        return np.clip(params, param_min, param_max)


# =======================
# Multipass Q网络（来自PDQN的核心优势）
# =======================
class MultiPassQNetwork(nn.Module):
    """
    Multipass Q网络：对每个动作单独计算Q值
    
    关键：对于动作a，只使用动作a对应的参数，其他参数置零
    这大大减少了不相关参数的干扰
    """
    
    def __init__(self, state_size, action_size, action_parameter_size_list, hidden_layers=None,
                 init_type="kaiming", init_std=0.01, activation="leaky_relu"):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size_list = action_parameter_size_list
        self.action_parameter_size = sum(action_parameter_size_list)
        self.activation = activation
        
        # 构建网络
        self.layers = nn.ModuleList()
        input_size = state_size + self.action_parameter_size
        last_hidden_size = input_size
        
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(input_size, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            last_hidden_size = hidden_layers[nh - 1]
        
        self.layers.append(nn.Linear(last_hidden_size, action_size))
        
        # 初始化
        for i in range(len(self.layers) - 1):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity='leaky_relu')
            nn.init.zeros_(self.layers[i].bias)
        
        if init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=init_std)
        nn.init.zeros_(self.layers[-1].bias)
        
        # 计算偏移量
        self.offsets = np.cumsum([0] + list(action_parameter_size_list))
    
    def forward(self, state, action_parameters):
        """
        Multipass前向传播：对每个动作单独计算Q值
        """
        batch_size = state.shape[0]
        Q = []
        
        # 对每个动作，只填充对应的参数，其他置零
        x = torch.cat((state, torch.zeros_like(action_parameters)), dim=1)
        x = x.repeat(self.action_size, 1)
        
        for a in range(self.action_size):
            x[a * batch_size:(a + 1) * batch_size,
              self.state_size + self.offsets[a]:self.state_size + self.offsets[a + 1]] = \
                action_parameters[:, self.offsets[a]:self.offsets[a + 1]]
        
        # 前向传播
        negative_slope = 0.01
        for i in range(len(self.layers) - 1):
            if self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                x = F.relu(self.layers[i](x))
        
        Qall = self.layers[-1](x)
        
        # 提取每个动作的Q值
        for a in range(self.action_size):
            Qa = Qall[a * batch_size:(a + 1) * batch_size, a]
            if len(Qa.shape) == 1:
                Qa = Qa.unsqueeze(1)
            Q.append(Qa)
        
        return torch.cat(Q, dim=1)


class ParamActor(nn.Module):
    """参数Actor网络 - 输出动作参数"""
    
    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=None,
                 init_type="kaiming", init_std=0.01, activation="leaky_relu"):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation
        
        self.layers = nn.ModuleList()
        last_hidden_size = state_size
        
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(state_size, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            last_hidden_size = hidden_layers[nh - 1]
        
        self.layers.append(nn.Linear(last_hidden_size, action_parameter_size))
        
        # Passthrough layer（残差连接）
        self.passthrough_layer = nn.Linear(state_size, action_parameter_size)
        nn.init.zeros_(self.passthrough_layer.weight)
        nn.init.zeros_(self.passthrough_layer.bias)
        self.passthrough_layer.requires_grad = False
        self.passthrough_layer.weight.requires_grad = False
        self.passthrough_layer.bias.requires_grad = False
        
        # 初始化
        for i in range(len(self.layers) - 1):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity='leaky_relu')
            nn.init.zeros_(self.layers[i].bias)
        
        if init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=init_std)
        nn.init.zeros_(self.layers[-1].bias)
    
    def forward(self, state):
        x = state
        negative_slope = 0.01
        
        for i in range(len(self.layers) - 1):
            if self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                x = F.relu(self.layers[i](x))
        
        action_params = self.layers[-1](x)
        action_params += self.passthrough_layer(state)
        
        return action_params


# =======================
# LIRL Agent
# =======================
class LIRLAgent:
    """
    LIRL Agent - 优化版本
    
    核心特性：
    1. LIRL动作投影（匈牙利算法 + QP）
    2. Multipass Q网络（来自PDQN）
    3. N-step returns + Inverting Gradients
    """
    
    def __init__(self, observation_space, action_space,
                 hidden_layers=[1024, 512, 256, 128],
                 learning_rate_actor=0.001, learning_rate_actor_param=0.00001,
                 gamma=0.99, tau_actor=0.001, tau_actor_param=0.001,
                 replay_memory_size=500000, batch_size=32,
                 clip_grad=1.0, inverting_gradients=True,
                 epsilon_initial=1.0, epsilon_final=0.1, epsilon_steps=1000,
                 initial_memory_threshold=1000, beta=0.2,
                 n_step_returns=True, use_ornstein_noise=False,
                 indexed=False, zero_index_gradients=False,
                 seed=None):
        
        self.num_actions = action_space.spaces[0].n
        self.action_parameter_sizes = np.array([action_space.spaces[i].shape[0] for i in range(1, self.num_actions + 1)])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        
        # 动作范围
        self.action_parameter_max_numpy = np.concatenate([action_space.spaces[i].high for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([action_space.spaces[i].low for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
        
        self.gamma = gamma
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self.batch_size = batch_size
        self.clip_grad = clip_grad
        self.inverting_gradients = inverting_gradients
        self.initial_memory_threshold = initial_memory_threshold
        self.beta = beta
        self.n_step_returns = n_step_returns
        self.indexed = indexed
        self.zero_index_gradients = zero_index_gradients
        
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        
        self._step = 0
        self._episode = 0
        
        # 随机种子
        self.seed = seed
        self.np_random = np.random.RandomState(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # LIRL动作投影
        self.action_projector = LIRLActionProjection(
            self.num_actions, self.action_parameter_sizes,
            self.action_parameter_min_numpy, self.action_parameter_max_numpy
        )
        
        # 噪声
        self.use_ornstein_noise = use_ornstein_noise
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.np_random)
        
        # 经验回放
        self.replay_memory = MemoryNStepReturns(
            replay_memory_size, observation_space.shape,
            (1 + self.action_parameter_size,),  # action index + parameters
            next_actions=False, n_step_returns=n_step_returns
        )
        
        # 网络
        state_size = observation_space.shape[0]
        
        # Multipass Q网络（Actor in PDQN terminology）
        self.actor = MultiPassQNetwork(
            state_size, self.num_actions, self.action_parameter_sizes,
            hidden_layers=hidden_layers, init_type="kaiming", activation="leaky_relu"
        ).to(device)
        self.actor_target = MultiPassQNetwork(
            state_size, self.num_actions, self.action_parameter_sizes,
            hidden_layers=hidden_layers, init_type="kaiming", activation="leaky_relu"
        ).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()
        
        # 参数网络
        self.actor_param = ParamActor(
            state_size, self.num_actions, self.action_parameter_size,
            hidden_layers=hidden_layers, init_type="kaiming", activation="leaky_relu"
        ).to(device)
        self.actor_param_target = ParamActor(
            state_size, self.num_actions, self.action_parameter_size,
            hidden_layers=hidden_layers, init_type="kaiming", activation="leaky_relu"
        ).to(device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()
        
        # 优化器
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=learning_rate_actor_param)
        
        self.loss_func = F.mse_loss

    def _invert_gradients(self, grad, vals, inplace=True):
        """Inverting Gradients"""
        max_p = self.action_parameter_max.cpu()
        min_p = self.action_parameter_min.cpu()
        rnge = self.action_parameter_range.cpu()
        
        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            for n in range(grad.shape[0]):
                index = grad[n] > 0
                grad[n][index] *= (index.float() * (max_p - vals[n]) / rnge)[index]
                grad[n][~index] *= ((~index).float() * (vals[n] - min_p) / rnge)[~index]
        return grad

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        """Zero gradients for non-selected actions"""
        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            offsets = np.cumsum([0] + list(self.action_parameter_sizes))
            for n in range(grad.shape[0]):
                action = batch_action_indices[n].item()
                for a in range(self.num_actions):
                    if a != action:
                        grad[n][offsets[a]:offsets[a + 1]] = 0.
        return grad

    def act(self, state):
        """选择动作 - 使用LIRL动作投影"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).to(device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            all_action_parameters = self.actor_param.forward(state_tensor)
            all_action_parameters = all_action_parameters.detach().cpu().data.numpy().squeeze()
            
            # 获取Q值作为动作分数
            Q_values = self.actor.forward(state_tensor, torch.from_numpy(all_action_parameters).unsqueeze(0).to(device))
            action_scores = Q_values.detach().cpu().data.numpy().squeeze()
            
            # Epsilon-greedy探索
            if self.np_random.uniform() < self.epsilon:
                action_scores = self.np_random.uniform(size=action_scores.shape)
                offsets = np.cumsum([0] + list(self.action_parameter_sizes))
                if not self.use_ornstein_noise:
                    for i in range(self.num_actions):
                        all_action_parameters[offsets[i]:offsets[i + 1]] = self.np_random.uniform(
                            self.action_parameter_min_numpy[offsets[i]:offsets[i + 1]],
                            self.action_parameter_max_numpy[offsets[i]:offsets[i + 1]])
            
            # LIRL动作投影：匈牙利算法 + QP
            action, action_parameters = self.action_projector.project(action_scores, all_action_parameters)
            
            if self.use_ornstein_noise and self.noise is not None:
                offset = np.cumsum([0] + list(self.action_parameter_sizes))[action]
                action_parameters += self.noise.sample()[offset:offset + self.action_parameter_sizes[action]]
                action_parameters = np.clip(action_parameters,
                    self.action_parameter_min_numpy[offset:offset + self.action_parameter_sizes[action]],
                    self.action_parameter_max_numpy[offset:offset + self.action_parameter_sizes[action]])
        
        return action, action_parameters, all_action_parameters

    def start_episode(self):
        pass

    def end_episode(self):
        self._episode += 1
        if self._episode < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (self._episode / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def _optimize_td_loss(self):
        """优化TD损失"""
        if self.replay_memory.nb_entries < self.batch_size or \
                self.replay_memory.nb_entries < self.initial_memory_threshold:
            return
        
        # 采样
        if self.n_step_returns:
            states, actions, rewards, next_states, terminals, n_step_returns = \
                self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
        else:
            states, actions, rewards, next_states, terminals = \
                self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
            n_step_returns = None
        
        states = torch.from_numpy(states).to(device)
        actions_combined = torch.from_numpy(actions).to(device)
        action_indices = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(device).squeeze()
        next_states = torch.from_numpy(next_states).to(device)
        terminals = torch.from_numpy(terminals).to(device).squeeze()
        if self.n_step_returns:
            n_step_returns = torch.from_numpy(n_step_returns).to(device)
        
        # 优化Q网络
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()
            
            off_policy_target = rewards + (1 - terminals) * self.gamma * Qprime
            if self.n_step_returns:
                on_policy_target = n_step_returns.squeeze()
                target = self.beta * on_policy_target + (1. - self.beta) * off_policy_target
            else:
                target = off_policy_target
        
        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, action_indices.view(-1, 1)).squeeze()
        loss_Q = self.loss_func(y_predicted, target)
        
        self.actor_optimiser.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()
        
        # 优化参数网络
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        Q = self.actor(states, action_params)
        
        if self.indexed:
            Q_indexed = Q.gather(1, action_indices.unsqueeze(1))
            Q_loss = torch.mean(Q_indexed)
        else:
            Q_loss = torch.mean(torch.sum(Q, 1))
        
        self.actor.zero_grad()
        Q_loss.backward()
        delta_a = deepcopy(action_params.grad.data)
        
        action_params = self.actor_param(Variable(states))
        
        if self.inverting_gradients:
            delta_a[:] = self._invert_gradients(delta_a, action_params, inplace=True)
        
        if self.zero_index_gradients:
            delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=action_indices, inplace=True)
        
        out = -torch.mul(delta_a, action_params)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(device))
        
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)
        self.actor_param_optimiser.step()
        
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)
        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)

    def __str__(self):
        return (f"LIRL Agent (Optimized with Multipass Q + LIRL Projection)\n"
                f"Num actions: {self.num_actions}\n"
                f"Action param size: {self.action_parameter_size}\n"
                f"Gamma: {self.gamma}, Beta: {self.beta}\n"
                f"Inverting gradients: {self.inverting_gradients}\n"
                f"N-step returns: {self.n_step_returns}\n"
                f"Indexed: {self.indexed}\n"
                f"Epsilon: {self.epsilon:.4f}\n"
                f"Memory size: {self.replay_memory.nb_entries}")


def pad_action(act, act_param):
    action = np.zeros((7,))
    action[0] = act
    if act == 0:
        action[[1, 2]] = act_param
    elif act == 1:
        action[3] = act_param
    elif act == 2:
        action[[4, 5]] = act_param
    return action


def evaluate(env, agent, episodes=1000):
    returns = []
    timesteps = []
    goals = []
    for _ in range(episodes):
        state = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        info = {'status': "NOT_SET"}
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, _ = agent.act(state)
            action = pad_action(act, act_param)
            state, reward, terminal, info = env.step(action)
            total_reward += reward
        goal = info['status'] == 'GOAL'
        timesteps.append(t)
        returns.append(total_reward)
        goals.append(goal)
    return np.column_stack((returns, timesteps, goals))


def compute_n_step_returns(episode_transitions, gamma):
    n = len(episode_transitions)
    n_step_returns = np.zeros((n,))
    n_step_returns[n - 1] = episode_transitions[n - 1][2]
    for i in range(n - 2, -1, -1):
        reward = episode_transitions[i][2]
        target = n_step_returns[i + 1]
        n_step_returns[i] = reward + gamma * target
    return n_step_returns


@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=20000, help='Number of episodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=32, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--beta', default=0.2, help='Averaging factor for on-policy and off-policy targets.', type=float)
@click.option('--update-ratio', default=0.1, help='Ratio of updates to samples.', type=float)
@click.option('--initial-memory-threshold', default=1000, help='Number of transitions required to start learning.', type=int)
@click.option('--replay-memory-size', default=500000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.1, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.001, help='Soft target network update factor for Q network.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update factor for param network.', type=float)
@click.option('--learning-rate-actor', default=0.001, help="Q network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.00001, help="Param network learning rate.", type=float)
@click.option('--clip-grad', default=1.0, help="Gradient clipping.", type=float)
@click.option('--inverting-gradients', default=True, help="Use inverting gradients.", type=bool)
@click.option('--n-step-returns', default=True, help="Use n-step returns.", type=bool)
@click.option('--indexed', default=False, help="Use indexed loss.", type=bool)
@click.option('--zero-index-gradients', default=False, help="Zero gradients for non-selected actions.", type=bool)
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--use-ornstein-noise', default=False, help="Use Ornstein-Uhlenbeck noise.", type=bool)
@click.option('--save-dir', default="results/soccer", help='Model save directory.', type=str)
@click.option('--layers', default="[256,128,64]", help='Hidden layer sizes.', cls=ClickPythonLiteralOption)
@click.option('--title', default="LIRL", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, beta, update_ratio,
        initial_memory_threshold, replay_memory_size, epsilon_steps, epsilon_final,
        tau_actor, tau_actor_param, learning_rate_actor, learning_rate_actor_param,
        clip_grad, inverting_gradients, n_step_returns, indexed, zero_index_gradients,
        scale_actions, use_ornstein_noise, layers, save_dir, title):
    
    env = gym.make('SoccerScoreGoal-v0')
    if scale_actions:
        env = SoccerScaledParameterisedActionWrapper(env)
    
    dir = os.path.join(save_dir, title)
    env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    
    print(env.action_space)
    print(env.observation_space)
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    agent = LIRLAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_layers=layers,
        learning_rate_actor=learning_rate_actor,
        learning_rate_actor_param=learning_rate_actor_param,
        gamma=gamma,
        tau_actor=tau_actor,
        tau_actor_param=tau_actor_param,
        replay_memory_size=replay_memory_size,
        batch_size=batch_size,
        clip_grad=clip_grad,
        inverting_gradients=inverting_gradients,
        epsilon_initial=1.0,
        epsilon_final=epsilon_final,
        epsilon_steps=epsilon_steps,
        initial_memory_threshold=initial_memory_threshold,
        beta=beta,
        n_step_returns=n_step_returns,
        indexed=indexed,
        zero_index_gradients=zero_index_gradients,
        use_ornstein_noise=use_ornstein_noise,
        seed=seed
    )
    
    print(agent)
    
    max_steps = 15000
    total_reward = 0.
    returns = []
    timesteps = []
    goals = []
    start_time = time.time()
    
    for i in range(episodes):
        info = {'status': "NOT_SET"}
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        
        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)
        
        episode_reward = 0.
        agent.start_episode()
        transitions = []
        
        for j in range(max_steps):
            next_state, reward, terminal, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            
            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            
            if n_step_returns:
                transitions.append([
                    state,
                    np.concatenate(([act], all_action_parameters)).ravel(),
                    reward,
                    next_state,
                    np.concatenate(([next_act], next_all_action_parameters)).ravel(),
                    terminal
                ])
            else:
                agent.replay_memory.append(
                    state=state,
                    action=np.concatenate(([act], all_action_parameters)).ravel(),
                    reward=reward,
                    next_state=next_state,
                    terminal=terminal
                )
            
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state
            episode_reward += reward
            
            if terminal:
                break
        
        agent.end_episode()
        
        if n_step_returns:
            nsreturns = compute_n_step_returns(transitions, gamma)
            for t, nsr in zip(transitions, nsreturns):
                agent.replay_memory.append(
                    state=t[0], action=t[1], reward=t[2],
                    next_state=t[3], next_action=t[4],
                    terminal=t[5], time_steps=None, n_step_return=nsr
                )
        
        n_updates = int(update_ratio * j)
        for _ in range(n_updates):
            agent._optimize_td_loss()
        
        returns.append(episode_reward)
        timesteps.append(j)
        goals.append(info['status'] == 'GOAL')
        total_reward += episode_reward
        
        if (i + 1) % 100 == 0:
            avg_reward = total_reward / (i + 1)
            goal_rate = sum(goals) / len(goals)
            recent_goals = sum(goals[-100:]) / min(100, len(goals))
            print('{0:5s} R:{1:.4f} r100:{2:.4f} G:{3:.2%} g100:{4:.2%} Eps:{5:.4f}'.format(
                str(i + 1), avg_reward, np.mean(returns[-100:]), goal_rate, recent_goals, agent.epsilon))
    
    end_time = time.time()
    print("Training took %.2f seconds" % (end_time - start_time))
    
    returns = env.get_episode_rewards()
    np.save(os.path.join(dir, title + "{}".format(str(seed))), np.column_stack((returns, timesteps, goals)))
    
    torch.save(agent.actor.state_dict(), os.path.join(dir, 'actor_{}.pth'.format(seed)))
    torch.save(agent.actor_param.state_dict(), os.path.join(dir, 'actor_param_{}.pth'.format(seed)))
    
    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon = 0.
        agent.noise = None
        agent.actor.eval()
        agent.actor_param.eval()
        start_time_eval = time.time()
        evaluation_results = evaluate(env, agent, evaluation_episodes)
        end_time_eval = time.time()
        print("Ave. evaluation return =", sum(evaluation_results[:, 0]) / evaluation_results.shape[0])
        print("Ave. timesteps =", sum(evaluation_results[:, 1]) / evaluation_results.shape[0])
        goal_timesteps = evaluation_results[:, 1][evaluation_results[:, 2] == 1]
        if len(goal_timesteps) > 0:
            print("Ave. timesteps per goal =", sum(goal_timesteps) / len(goal_timesteps))
        print("Ave. goal prob. =", sum(evaluation_results[:, 2]) / evaluation_results.shape[0])
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_results)
        print("Evaluation time: %.2f seconds" % (end_time_eval - start_time_eval))
    
    print("Total training time: %.2f seconds" % (end_time - start_time))
    print(agent)
    env.close()


if __name__ == '__main__':
    run()
