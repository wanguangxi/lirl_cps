"""
PDQN (Parameterized Deep Q-Network) for CityFlow Multi-Intersection Traffic Signal Control

Designed for a hybrid action space:
- Discrete action: phase selection (per intersection: 0 ~ num_phases-1)
- Continuous parameter: green duration per phase (min_duration ~ max_duration)

PDQN key idea:
1. Q-network: takes state + continuous parameters, outputs Q-values for each discrete action
2. Parameter network: takes state, outputs continuous parameters for each discrete action
3. Action selection: pick the discrete action with highest Q, use its corresponding continuous parameter

Reference:
- Xiong et al. "Parameterized Deep Q-Networks Learning: Reinforcement Learning with 
  Discrete-Continuous Hybrid Action Space" (2018)
"""

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add environment path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env"))
from cityflow_multi_env import CityFlowMultiIntersectionEnv, get_default_config

# =======================
# Device detection
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[DEVICE] CUDA version: {torch.version.cuda}", flush=True)

# =======================
# HYPERPARAMETERS CONFIG
# =======================
CONFIG = {
    # Learning parameters
    'lr_q': 0.001,           # Q-network learning rate
    'lr_param': 0.0003,      # parameter-network learning rate
    'gamma': 0.99,           # discount factor
    'batch_size': 256 if torch.cuda.is_available() else 64,  # larger batch on GPU
    'buffer_limit': 100000,
    'tau': 0.005,            # target network soft update coefficient
    
    # Environment parameters
    'episode_length': 3600,
    'ctrl_interval': 10,
    'min_green': 10,
    'min_duration': 10,
    'max_duration': 60,
    'num_of_episodes': 200,
    
    # Exploration parameters
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.995,
    'param_noise_sigma': 0.2,  # exploration noise for continuous parameters
    
    # Network architecture
    'hidden_dim1': 256,
    'hidden_dim2': 128,
    
    # Training parameters
    'memory_threshold': 500,
    'training_iterations': 10,
    'clip_grad_norm': 1.0,
    
    # Output parameters
    'print_interval': 10,
    'save_models': True,
    'output_dir': './outputs/pdqn_cityflow',
}


class ReplayBuffer:
    """Replay buffer for PDQN hybrid actions (GPU-friendly tensors)."""
    def __init__(self, buffer_limit=None, device=None):
        limit = buffer_limit or CONFIG['buffer_limit']
        self.buffer = collections.deque(maxlen=limit)
        self.device = device or DEVICE

    def put(self, transition):
        """
        Store transition: (state, discrete_action, continuous_params, reward, next_state, done)
        """
        self.buffer.append(transition)
        
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, disc_a_lst, cont_params_lst, r_lst, s_prime_lst, done_mask_lst = \
            [], [], [], [], [], []

        for transition in mini_batch:
            s, disc_a, cont_params, r, s_prime, done = transition
            s_lst.append(s)
            disc_a_lst.append(disc_a)
            cont_params_lst.append(cont_params)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        # Create tensors directly on the target device
        return (
            torch.FloatTensor(np.array(s_lst)).to(self.device),
            torch.LongTensor(np.array(disc_a_lst)).to(self.device),
            torch.FloatTensor(np.array(cont_params_lst)).to(self.device),
            torch.FloatTensor(np.array(r_lst)).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(s_prime_lst)).to(self.device),
            torch.FloatTensor(np.array(done_mask_lst)).to(self.device)
        )
    
    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q 网络：输入状态和所有连续参数，输出每个离散动作的 Q 值
    
    对于多路口场景：
    - 输入: [state, all_continuous_params]
    - 输出: Q 值向量，维度 = num_intersections * num_phases
    
    这里采用分解式 Q 值（每个路口独立），然后聚合
    """
    def __init__(self, state_size: int, num_intersections: int, num_phases: int,
                 hidden_dim1: int = 256, hidden_dim2: int = 128):
        super(QNetwork, self).__init__()
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        
        # 连续参数维度 = num_intersections * num_phases (每个动作一个参数)
        self.param_dim = num_intersections * num_phases
        
        # 共享特征提取层
        self.fc_shared = nn.Sequential(
            nn.Linear(state_size + self.param_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        
        # 每个路口独立的 Q 值头
        self.q_heads = nn.ModuleList([
            nn.Linear(hidden_dim2, num_phases) for _ in range(num_intersections)
        ])
        
    def forward(self, state: torch.Tensor, continuous_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
            continuous_params: (batch, num_intersections * num_phases)
        
        Returns:
            q_values: (batch, num_intersections * num_phases) - 所有路口所有相位的 Q 值
        """
        # 拼接状态和连续参数
        x = torch.cat([state, continuous_params], dim=1)
        
        # 共享特征
        shared_features = self.fc_shared(x)
        
        # 每个路口的 Q 值
        q_list = []
        for i, q_head in enumerate(self.q_heads):
            q_i = q_head(shared_features)  # (batch, num_phases)
            q_list.append(q_i)
        
        # 拼接所有 Q 值
        q_values = torch.cat(q_list, dim=1)  # (batch, num_intersections * num_phases)
        
        return q_values
    
    def get_q_per_intersection(self, state: torch.Tensor, continuous_params: torch.Tensor) -> List[torch.Tensor]:
        """返回每个路口的 Q 值（用于动作选择）"""
        x = torch.cat([state, continuous_params], dim=1)
        shared_features = self.fc_shared(x)
        
        q_list = []
        for q_head in self.q_heads:
            q_i = q_head(shared_features)
            q_list.append(q_i)
        
        return q_list


class ParameterNetwork(nn.Module):
    """
    参数网络：输入状态，输出所有离散动作对应的连续参数
    
    对于多路口场景：
    - 输入: state
    - 输出: 每个路口每个相位的绿灯时长参数 (归一化到 [0, 1])
    """
    def __init__(self, state_size: int, num_intersections: int, num_phases: int,
                 hidden_dim1: int = 256, hidden_dim2: int = 128):
        super(ParameterNetwork, self).__init__()
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.param_dim = num_intersections * num_phases
        
        # 共享特征提取
        self.fc_shared = nn.Sequential(
            nn.Linear(state_size, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        
        # 参数输出层
        self.fc_params = nn.Linear(hidden_dim2, self.param_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
        
        Returns:
            params: (batch, num_intersections * num_phases) - 每个动作的连续参数 [0, 1]
        """
        features = self.fc_shared(state)
        params = torch.sigmoid(self.fc_params(features))
        return params


class PDQNAgent:
    """
    PDQN 智能体 - 用于 CityFlow 多路口交通信号控制 (GPU 加速版)
    
    混合动作空间：
    - 离散动作：每个路口选择一个相位
    - 连续参数：选定相位的绿灯时长
    """
    def __init__(self, env: CityFlowMultiIntersectionEnv, config: Dict = None, device=None):
        self.config = config or CONFIG.copy()
        self.env = env
        self.device = device or DEVICE
        
        # 环境参数
        self.state_size = env.observation_space.shape[0]
        self.num_intersections = env.num_intersections
        self.num_phases = env.num_phases
        self.min_duration = env.min_duration
        self.max_duration = env.max_duration
        self.min_green = env.min_green
        
        # 动作空间维度
        self.discrete_action_size = self.num_intersections  # 每个路口一个离散选择
        self.continuous_param_size = self.num_intersections * self.num_phases  # 每个动作一个参数
        
        print(f"[PDQN Agent] 初始化:")
        print(f"  设备: {self.device}")
        print(f"  状态维度: {self.state_size}")
        print(f"  路口数量: {self.num_intersections}")
        print(f"  每个路口相位数: {self.num_phases}")
        print(f"  连续参数维度: {self.continuous_param_size}")
        print(f"  绿灯时长范围: [{self.min_duration}, {self.max_duration}]秒")
        
        # 创建网络
        hidden1 = self.config['hidden_dim1']
        hidden2 = self.config['hidden_dim2']
        
        # Q 网络 (移动到 GPU)
        self.q_network = QNetwork(
            self.state_size, self.num_intersections, self.num_phases, hidden1, hidden2
        ).to(self.device)
        self.q_target = QNetwork(
            self.state_size, self.num_intersections, self.num_phases, hidden1, hidden2
        ).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        # 参数网络 (移动到 GPU)
        self.param_network = ParameterNetwork(
            self.state_size, self.num_intersections, self.num_phases, hidden1, hidden2
        ).to(self.device)
        self.param_target = ParameterNetwork(
            self.state_size, self.num_intersections, self.num_phases, hidden1, hidden2
        ).to(self.device)
        self.param_target.load_state_dict(self.param_network.state_dict())
        
        # 优化器
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['lr_q'])
        self.param_optimizer = optim.Adam(self.param_network.parameters(), lr=self.config['lr_param'])
        
        # 经验回放 (支持 GPU)
        self.memory = ReplayBuffer(self.config['buffer_limit'], device=self.device)
        
        # 探索参数
        self.epsilon = self.config['epsilon_start']
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        选择动作
        
        Args:
            state: 当前状态
            explore: 是否进行探索
        
        Returns:
            discrete_actions: 每个路口的相位选择 (num_intersections,)
            continuous_params: 选定动作的绿灯时长 (num_intersections,)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 获取所有连续参数
            all_params = self.param_network(state_tensor)  # (1, num_intersections * num_phases)
            
            # 添加参数噪声（探索）
            if explore:
                noise = torch.randn_like(all_params) * self.config['param_noise_sigma']
                all_params = torch.clamp(all_params + noise, 0, 1)
            
            # 获取每个路口的 Q 值
            q_per_intersection = self.q_network.get_q_per_intersection(state_tensor, all_params)
            
            discrete_actions = []
            continuous_params = []
            
            for i in range(self.num_intersections):
                q_i = q_per_intersection[i].squeeze(0)  # (num_phases,)
                
                # epsilon-greedy 选择
                if explore and random.random() < self.epsilon:
                    action_i = random.randint(0, self.num_phases - 1)
                else:
                    action_i = q_i.argmax().item()
                
                discrete_actions.append(action_i)
                
                # 获取选定动作的连续参数
                param_idx = i * self.num_phases + action_i
                param_i = all_params[0, param_idx].item()
                
                # 映射到实际时长
                duration_i = self.min_duration + param_i * (self.max_duration - self.min_duration)
                continuous_params.append(duration_i)
        
        return np.array(discrete_actions), np.array(continuous_params), all_params.squeeze(0).cpu().numpy()
    
    def convert_to_env_action(self, discrete_actions: np.ndarray, continuous_params: np.ndarray) -> np.ndarray:
        """
        转换为环境动作格式
        
        环境动作格式: [phase_0, duration_idx_0, phase_1, duration_idx_1, ...]
        """
        env_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        
        for i in range(self.num_intersections):
            env_action[i * 2] = discrete_actions[i]
            
            # 将连续时长转换为时长索引
            duration_idx = int(round(continuous_params[i] - self.min_duration))
            duration_idx = np.clip(duration_idx, 0, self.max_duration - self.min_duration)
            env_action[i * 2 + 1] = duration_idx
        
        return env_action
    
    def store_transition(self, state, discrete_actions, all_params, reward, next_state, done):
        """存储经验"""
        self.memory.put((state, discrete_actions, all_params, reward, next_state, done))
    
    def update(self) -> Tuple[float, float]:
        """
        更新网络
        
        Returns:
            q_loss, param_loss
        """
        if self.memory.size() < self.config['batch_size']:
            return 0.0, 0.0
        
        # 采样批次
        states, disc_actions, cont_params, rewards, next_states, done_masks = \
            self.memory.sample(self.config['batch_size'])
        
        # ========== 更新 Q 网络 ==========
        # 计算目标 Q 值
        with torch.no_grad():
            next_params = self.param_target(next_states)
            next_q_all = self.q_target(next_states, next_params)
            
            # 每个路口选择最大 Q 值
            next_q_max = torch.zeros(self.config['batch_size'], 1, device=self.device)
            for i in range(self.num_intersections):
                q_i = next_q_all[:, i * self.num_phases:(i + 1) * self.num_phases]
                next_q_max += q_i.max(dim=1, keepdim=True)[0]
            
            target_q = rewards + self.config['gamma'] * next_q_max * done_masks
        
        # 当前 Q 值
        current_q_all = self.q_network(states, cont_params)
        
        # 提取选定动作的 Q 值
        current_q = torch.zeros(self.config['batch_size'], 1, device=self.device)
        for i in range(self.num_intersections):
            q_i = current_q_all[:, i * self.num_phases:(i + 1) * self.num_phases]
            action_i = disc_actions[:, i].unsqueeze(1)  # (batch, 1)
            current_q += q_i.gather(1, action_i)
        
        # Q 网络损失
        q_loss = F.mse_loss(current_q, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config['clip_grad_norm'])
        self.q_optimizer.step()
        
        # ========== 更新参数网络 ==========
        # 参数网络的目标：最大化 Q 值
        params = self.param_network(states)
        q_all = self.q_network(states, params)
        
        # 计算总 Q 值（策略梯度）
        param_loss = -q_all.mean()
        
        self.param_optimizer.zero_grad()
        param_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.param_network.parameters(), self.config['clip_grad_norm'])
        self.param_optimizer.step()
        
        return q_loss.item(), param_loss.item()
    
    def soft_update_targets(self):
        """软更新目标网络"""
        tau = self.config['tau']
        
        for target_param, param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
        
        for target_param, param in zip(self.param_target.parameters(), self.param_network.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(
            self.config['epsilon_end'],
            self.epsilon * self.config['epsilon_decay']
        )
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'param_network': self.param_network.state_dict(),
            'param_target': self.param_target.state_dict(),
            'config': self.config,
        }, path)
        print(f"模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.param_network.load_state_dict(checkpoint['param_network'])
        self.param_target.load_state_dict(checkpoint['param_target'])
        print(f"模型已加载: {path} (设备: {self.device})")


class ActionConverter:
    """
    动作转换器 - 将 PDQN 输出转换为环境动作格式
    
    不做约束检查，约束违反由环境检测和记录
    """
    def __init__(self, num_intersections: int, num_phases: int, 
                 min_duration: int, max_duration: int):
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def convert(self, discrete_actions: np.ndarray, continuous_params: np.ndarray) -> np.ndarray:
        """
        将 PDQN 输出转换为环境动作格式
        
        不做约束检查，直接转换动作格式
        约束违反由环境的 _apply_action 检测和记录
        
        Args:
            discrete_actions: 每个路口的相位选择 (num_intersections,)
            continuous_params: 每个路口的绿灯时长 (num_intersections,)
        
        Returns:
            env_action: [phase_0, duration_idx_0, phase_1, duration_idx_1, ...]
        """
        env_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        
        for i in range(self.num_intersections):
            # 相位（仅做基本范围裁剪，不检查约束）
            phase = int(discrete_actions[i])
            phase = np.clip(phase, 0, self.num_phases - 1)
            env_action[i * 2] = phase
            
            # 时长索引
            duration_idx = int(round(continuous_params[i] - self.min_duration))
            duration_idx = np.clip(duration_idx, 0, self.max_duration - self.min_duration)
            env_action[i * 2 + 1] = duration_idx
        
        return env_action


def train(config: Dict = None, cityflow_config_path: str = None):
    """训练 PDQN 智能体"""
    config = config or CONFIG.copy()
    
    # 设置 CityFlow 配置路径
    if cityflow_config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cityflow_config_path = os.path.join(script_dir, "../examples/City_3_5/config.json")
    
    # 创建环境配置
    env_config = get_default_config(cityflow_config_path)
    env_config["episode_length"] = config['episode_length']
    env_config["ctrl_interval"] = config['ctrl_interval']
    env_config["min_green"] = config['min_green']
    env_config["min_duration"] = config['min_duration']
    env_config["max_duration"] = config['max_duration']
    env_config["verbose_violations"] = False
    env_config["log_violations"] = True
    
    # 创建环境
    env = CityFlowMultiIntersectionEnv(env_config)
    
    print(f"\n{'='*60}")
    print("PDQN for CityFlow Traffic Signal Control")
    print(f"{'='*60}")
    
    # 创建智能体
    agent = PDQNAgent(env, config)
    
    # 创建动作转换器（不做约束检查，约束由环境检测）
    action_converter = ActionConverter(
        env.num_intersections, env.num_phases,
        env.min_duration, env.max_duration
    )
    
    # 创建输出目录
    output_dir = config.get('output_dir', './outputs/pdqn_cityflow')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 训练记录
    episode_rewards = []
    episode_travel_times = []
    episode_throughputs = []
    episode_violations = []          # 总违反次数
    episode_violation_details = []   # 违反详情（分类）
    episode_q_losses = []
    episode_param_losses = []
    
    print(f"\n训练开始，输出目录: {run_dir}", flush=True)
    print(f"总 Episodes: {config['num_of_episodes']}", flush=True)
    print(f"每 Episode 步数: {config['episode_length'] // config['ctrl_interval']}", flush=True)
    print(f"打印间隔: 每 {config['print_interval']} episodes\n", flush=True)
    
    import time
    train_start_time = time.time()
    
    for n_epi in range(config['num_of_episodes']):
        episode_start_time = time.time()
        state, info = env.reset()
        done = False
        episode_reward = 0
        step = 0
        q_losses, param_losses = [], []
        
        total_steps = config['episode_length'] // config['ctrl_interval']
        
        while not done:
            # 选择动作
            discrete_actions, continuous_params, all_params = agent.select_action(state, explore=True)
            
            # 转换为环境动作格式（不做约束检查）
            env_action = action_converter.convert(discrete_actions, continuous_params)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            # 存储经验
            agent.store_transition(state, discrete_actions, all_params, reward, next_state, done)
            
            # 更新网络
            if agent.memory.size() >= config['memory_threshold']:
                for _ in range(config['training_iterations']):
                    q_loss, param_loss = agent.update()
                    q_losses.append(q_loss)
                    param_losses.append(param_loss)
                
                # 软更新目标网络
                agent.soft_update_targets()
            
            episode_reward += reward
            state = next_state
            step += 1
            
            # 每100步打印进度（使用进度条形式）
            if step % 100 == 0 or done:
                progress = step / total_steps
                bar_len = 20
                filled = int(bar_len * progress)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f"\r  [{bar}] {progress*100:5.1f}% | Step {step}/{total_steps} | "
                      f"R={episode_reward:.0f} | Mem={agent.memory.size()}", end="", flush=True)
        
        # 记录统计
        episode_rewards.append(episode_reward)
        avg_travel_time = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_travel_time)
        
        flow_stats = info.get('intersection_flow', {})
        total_throughput = sum(s.get('throughput', 0) for s in flow_stats.values())
        episode_throughputs.append(total_throughput)
        
        # 获取环境统计的约束违反（环境检测并阻止）
        env_violations = info.get('total_violations', {})
        total_viol = sum(env_violations.values()) if env_violations else 0
        episode_violations.append(total_viol)
        episode_violation_details.append(env_violations.copy() if env_violations else {})
        
        avg_q_loss = np.mean(q_losses) if q_losses else 0
        avg_param_loss = np.mean(param_losses) if param_losses else 0
        episode_q_losses.append(avg_q_loss)
        episode_param_losses.append(avg_param_loss)
        
        episode_time = time.time() - episode_start_time
        
        # 每个 Episode 结束打印简要信息（包含投影器阻止的违反次数）
        print(f"\n✓ Episode {n_epi+1}/{config['num_of_episodes']} 完成 | "
              f"Reward={episode_reward:.1f} | AvgTT={avg_travel_time:.0f}s | "
              f"Violations={total_viol} | Time={episode_time:.1f}s | ε={agent.epsilon:.3f}", flush=True)
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 打印详细进度（每 print_interval 个 episode）
        if (n_epi + 1) % config['print_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-config['print_interval']:])
            avg_tt = np.mean(episode_travel_times[-config['print_interval']:])
            avg_tp = np.mean(episode_throughputs[-config['print_interval']:])
            avg_viol = np.mean(episode_violations[-config['print_interval']:])
            elapsed = time.time() - train_start_time
            
            print(f"\n{'─'*60}")
            print(f"📊 Episode {n_epi+1}/{config['num_of_episodes']} 统计 (耗时: {elapsed/60:.1f}分钟)")
            print(f"   平均奖励: {avg_reward:.1f}")
            print(f"   平均行程时间: {avg_tt:.1f}s")
            print(f"   总吞吐量: {avg_tp:.0f}")
            print(f"   约束违反: {avg_viol:.1f}")
            print(f"   探索率 ε: {agent.epsilon:.3f}")
            print(f"   Q-Loss: {avg_q_loss:.4f}, Param-Loss: {avg_param_loss:.4f}")
            print(f"{'─'*60}\n", flush=True)
    
    # 保存模型
    if config.get('save_models', True):
        model_path = os.path.join(run_dir, "pdqn_cityflow_final.pt")
        agent.save(model_path)
    
    # 保存训练曲线
    import json
    stats_path = os.path.join(run_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            'episode_rewards': episode_rewards,
            'episode_travel_times': episode_travel_times,
            'episode_throughputs': episode_throughputs,
            'episode_violations': episode_violations,
            'episode_violation_details': episode_violation_details,
            'episode_q_losses': episode_q_losses,
            'episode_param_losses': episode_param_losses,
        }, f, indent=2)
    print(f"训练统计已保存: {stats_path}")
    
    # 打印最终违反统计（环境检测）
    if episode_violations:
        total_min_green = sum(d.get('min_green', 0) for d in episode_violation_details)
        total_target_dur = sum(d.get('target_duration', 0) for d in episode_violation_details)
        total_invalid = sum(d.get('invalid_phase', 0) for d in episode_violation_details)
        print(f"\n约束违反总计（环境检测）:")
        print(f"  最小绿灯时间违反: {total_min_green}")
        print(f"  目标时长违反: {total_target_dur}")
        print(f"  无效相位: {total_invalid}")
        print(f"  总计: {sum(episode_violations)}")
    
    env.close()
    
    return {
        'agent': agent,
        'episode_rewards': episode_rewards,
        'episode_travel_times': episode_travel_times,
        'run_dir': run_dir,
    }


def evaluate(model_path: str, cityflow_config_path: str = None, n_episodes: int = 5, render: bool = True):
    """评估训练好的模型"""
    # 加载模型
    checkpoint = torch.load(model_path)
    config = checkpoint.get('config', CONFIG)
    
    # 设置 CityFlow 配置路径
    if cityflow_config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cityflow_config_path = os.path.join(script_dir, "../examples/City_3_5/config.json")
    
    # 创建环境
    env_config = get_default_config(cityflow_config_path)
    env_config["episode_length"] = config.get('episode_length', 3600)
    env_config["ctrl_interval"] = config.get('ctrl_interval', 10)
    env_config["min_green"] = config.get('min_green', 10)
    env_config["min_duration"] = config.get('min_duration', 10)
    env_config["max_duration"] = config.get('max_duration', 60)
    env_config["verbose_violations"] = render
    
    env = CityFlowMultiIntersectionEnv(env_config, render_mode="human" if render else None)
    
    # 创建智能体并加载权重
    agent = PDQNAgent(env, config)
    agent.load(model_path)
    
    # 创建动作转换器（不做约束检查，约束由环境检测）
    action_converter = ActionConverter(
        env.num_intersections, env.num_phases,
        env.min_duration, env.max_duration
    )
    
    print(f"\n{'='*60}")
    print("PDQN 模型评估")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")
    print(f"评估 Episodes: {n_episodes}\n")
    
    episode_rewards = []
    episode_travel_times = []
    episode_violations = []
    
    for ep in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 选择动作（不探索）
            discrete_actions, continuous_params, _ = agent.select_action(state, explore=False)
            
            # 转换为环境动作格式（不做约束检查）
            env_action = action_converter.convert(discrete_actions, continuous_params)
            
            # 执行动作
            state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            episode_reward += reward
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        avg_travel_time = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_travel_time)
        
        # 获取环境违反统计
        env_violations = info.get('total_violations', {})
        total_viol = sum(env_violations.values()) if env_violations else 0
        episode_violations.append(total_viol)
        
        print(f"Episode {ep+1}/{n_episodes}: "
              f"Reward={episode_reward:.1f}, "
              f"AvgTravelTime={avg_travel_time:.1f}s, "
              f"Violations={total_viol}")
        
        if render:
            env.print_intersection_flow_summary()
            env.print_violation_summary()
    
    env.close()
    
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均行程时间: {np.mean(episode_travel_times):.2f}s")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_travel_times': episode_travel_times,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDQN for CityFlow Traffic Signal Control")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"],
                       help="运行模式")
    parser.add_argument("--config", type=str, default="../examples/City_3_5/config.json",
                       help="CityFlow 配置文件路径")
    parser.add_argument("--model", type=str, default=None,
                       help="模型文件路径 (evaluate 模式需要)")
    parser.add_argument("--episodes", type=int, default=200,
                       help="训练 episodes 数")
    parser.add_argument("--episode-length", type=int, default=3600,
                       help="每个 episode 的仿真时长（秒）")
    parser.add_argument("--min-duration", type=int, default=10,
                       help="最小绿灯时长（秒）")
    parser.add_argument("--max-duration", type=int, default=60,
                       help="最大绿灯时长（秒）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理配置文件路径
    if not os.path.isabs(args.config):
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config
    
    if args.mode == "train":
        # 更新配置
        config = CONFIG.copy()
        config['num_of_episodes'] = args.episodes
        config['episode_length'] = args.episode_length
        config['min_duration'] = args.min_duration
        config['max_duration'] = args.max_duration
        
        results = train(config=config, cityflow_config_path=config_path)
        
        print("\n训练完成!")
        print(f"最终平均奖励: {np.mean(results['episode_rewards'][-10:]):.2f}")
        print(f"最终平均行程时间: {np.mean(results['episode_travel_times'][-10:]):.2f}s")
        
    elif args.mode == "evaluate":
        if args.model is None:
            print("错误: evaluate 模式需要指定 --model 参数")
            sys.exit(1)
        
        if not os.path.isabs(args.model):
            model_path = os.path.join(script_dir, args.model)
        else:
            model_path = args.model
        
        results = evaluate(model_path, cityflow_config_path=config_path, n_episodes=5)
    
    print("\n完成!")

