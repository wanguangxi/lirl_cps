"""
多规模交通仿真环境训练与决策时间测试

本程序针对不同规模的交通网络（3x5, 5x10, 10x10）：
1. 使用 LIRL (DDPG-based) 算法训练策略
2. 测量各环节的决策时间：
   - 神经网络推理时间
   - 离散动作映射时间（相位选择）
   - 连续动作映射时间（绿灯时长参数）
   - 总决策时间
3. 保存训练好的模型
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "env"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "algs"))

from cityflow_multi_env import CityFlowMultiIntersectionEnv, get_default_config

# 从 lirl_cityflow.py 导入必要组件
from lirl_cityflow import (
    ActorNetwork, CriticNetwork, ReplayBuffer, 
    OrnsteinUhlenbeckNoise, CityFlowActionProjector,
    train_step, soft_update, CONFIG
)


# =======================
# 规模配置
# =======================
SCALE_CONFIGS = {
    "City_3_5": {
        "name": "3x5 (15 intersections)",
        "config_path": os.path.join(PROJECT_ROOT, "examples/City_3_5/config.json"),
        "expected_intersections": 15,
    },
    "City_5_10": {
        "name": "5x10 (50 intersections)",
        "config_path": os.path.join(PROJECT_ROOT, "examples/City_5_10/config.json"),
        "expected_intersections": 50,
    },
    "City_10_10": {
        "name": "10x10 (100 intersections)",
        "config_path": os.path.join(PROJECT_ROOT, "examples/City_10_10/config.json"),
        "expected_intersections": 100,
    },
}


class TimingProfiler:
    """决策时间分析器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有计时记录"""
        self.nn_inference_times: List[float] = []
        self.discrete_mapping_times: List[float] = []
        self.continuous_mapping_times: List[float] = []
        self.total_decision_times: List[float] = []
        self.projection_times: List[float] = []  # 完整投影时间
    
    def record_nn_inference(self, elapsed: float):
        """记录神经网络推理时间"""
        self.nn_inference_times.append(elapsed)
    
    def record_discrete_mapping(self, elapsed: float):
        """记录离散动作映射时间"""
        self.discrete_mapping_times.append(elapsed)
    
    def record_continuous_mapping(self, elapsed: float):
        """记录连续动作映射时间"""
        self.continuous_mapping_times.append(elapsed)
    
    def record_projection(self, elapsed: float):
        """记录完整投影时间"""
        self.projection_times.append(elapsed)
    
    def record_total_decision(self, elapsed: float):
        """记录总决策时间"""
        self.total_decision_times.append(elapsed)
    
    def get_statistics(self) -> Dict:
        """获取统计结果"""
        def calc_stats(times):
            if not times:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
            return {
                "mean": np.mean(times) * 1000,  # 转换为毫秒
                "std": np.std(times) * 1000,
                "min": np.min(times) * 1000,
                "max": np.max(times) * 1000,
                "count": len(times),
            }
        
        return {
            "nn_inference": calc_stats(self.nn_inference_times),
            "discrete_mapping": calc_stats(self.discrete_mapping_times),
            "continuous_mapping": calc_stats(self.continuous_mapping_times),
            "projection": calc_stats(self.projection_times),
            "total_decision": calc_stats(self.total_decision_times),
        }


class TimedActionProjector(CityFlowActionProjector):
    """
    带时间测量的动作投影器
    
    继承自 CityFlowActionProjector，在投影过程中测量各环节耗时
    """
    
    def __init__(self, num_intersections, num_phases, num_duration_options,
                 min_green=10, duration_options=None, profiler=None):
        super().__init__(num_intersections, num_phases, num_duration_options,
                        min_green, duration_options)
        self.profiler = profiler
    
    def project_with_timing(self, continuous_action, env=None) -> Tuple[np.ndarray, Dict]:
        """
        带时间测量的投影
        
        Returns:
            discrete_action: 投影后的离散动作
            timing_info: 各环节耗时信息
        """
        timing_info = {}
        
        total_start = time.perf_counter()
        
        if isinstance(continuous_action, torch.Tensor):
            continuous_action = continuous_action.detach().cpu().numpy()
        
        # 确保是一维数组且在 [0, 1] 范围内
        a_ = np.clip(continuous_action.flatten(), 0, 1)
        
        expected_dim = self.num_intersections * 2
        if len(a_) < expected_dim:
            padded = np.ones(expected_dim) * 0.5
            padded[:len(a_)] = a_
            a_ = padded
        elif len(a_) > expected_dim:
            a_ = a_[:expected_dim]
        
        discrete_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        
        # 如果没有环境对象，使用简单投影
        if env is None:
            discrete_action = self._simple_project(a_)
            total_elapsed = time.perf_counter() - total_start
            timing_info["total"] = total_elapsed
            return discrete_action, timing_info
        
        # ========== 约束感知投影（带时间测量）==========
        
        # 1. 获取环境状态快照（离散映射准备）
        discrete_mapping_start = time.perf_counter()
        try:
            current_phases = env.current_phases.copy()
            phase_elapsed = env.phase_elapsed.copy()
            target_durations = env.target_durations.copy()
            valid_phases = env.valid_phases.copy()
            intersection_ids = env.intersection_ids.copy()
            lane_waiting = env.eng.get_lane_waiting_vehicle_count()
        except Exception as e:
            discrete_action = self._simple_project(a_)
            total_elapsed = time.perf_counter() - total_start
            timing_info["total"] = total_elapsed
            return discrete_action, timing_info
        
        discrete_mapping_elapsed = 0.0
        continuous_mapping_elapsed = 0.0
        
        # 2. 对每个路口进行投影
        for i, inter_id in enumerate(intersection_ids):
            phase_prob = a_[i * 2]
            duration_prob = a_[i * 2 + 1]
            
            # ====== 离散映射（相位选择）======
            discrete_start = time.perf_counter()
            
            cur_phase = current_phases.get(inter_id, 0)
            elapsed = phase_elapsed.get(inter_id, 0.0)
            target_duration = target_durations.get(inter_id, self.duration_options[0])
            inter_valid_phases = valid_phases.get(inter_id, [True] * self.num_phases)
            
            # 获取等待队列
            queue_by_dir = {"N": 0, "E": 0, "S": 0, "W": 0}
            if hasattr(env, 'in_lanes') and inter_id in env.in_lanes:
                for direction in ["N", "E", "S", "W"]:
                    lanes = env.in_lanes[inter_id].get(direction, [])
                    queue_by_dir[direction] = sum(lane_waiting.get(lane, 0) for lane in lanes)
            
            # 确定可行相位集合
            feasible_phases = []
            for p in range(self.num_phases):
                if inter_valid_phases[p]:
                    if p == cur_phase:
                        feasible_phases.append(p)
                    elif elapsed >= self.min_green and elapsed >= target_duration:
                        feasible_phases.append(p)
            
            if not feasible_phases:
                feasible_phases = [cur_phase]
            
            # 计算每个可行相位的成本
            phase_costs = []
            for p in feasible_phases:
                desired_phase = int(phase_prob * (self.num_phases - 1) + 0.5)
                preference_cost = abs(p - desired_phase) / max(self.num_phases - 1, 1)
                switch_cost = 0.1 if p != cur_phase else 0.0
                total_cost = preference_cost + switch_cost
                phase_costs.append((p, total_cost))
            
            # 选择成本最小的相位
            phase_costs.sort(key=lambda x: x[1])
            selected_phase = phase_costs[0][0]
            
            discrete_mapping_elapsed += time.perf_counter() - discrete_start
            
            # ====== 连续映射（绿灯时长）======
            continuous_start = time.perf_counter()
            
            desired_duration_idx = int(duration_prob * (self.num_duration_options - 1) + 0.5)
            desired_duration_idx = np.clip(desired_duration_idx, 0, self.num_duration_options - 1)
            
            min_duration_idx = 0
            for idx, dur in enumerate(self.duration_options):
                if dur >= self.min_green:
                    min_duration_idx = idx
                    break
            
            selected_duration_idx = max(desired_duration_idx, min_duration_idx)
            
            continuous_mapping_elapsed += time.perf_counter() - continuous_start
            
            # 存储结果
            discrete_action[i * 2] = selected_phase
            discrete_action[i * 2 + 1] = selected_duration_idx
        
        total_elapsed = time.perf_counter() - total_start
        
        # 记录时间
        timing_info = {
            "discrete_mapping": discrete_mapping_elapsed,
            "continuous_mapping": continuous_mapping_elapsed,
            "total": total_elapsed,
        }
        
        if self.profiler:
            self.profiler.record_discrete_mapping(discrete_mapping_elapsed)
            self.profiler.record_continuous_mapping(continuous_mapping_elapsed)
            self.profiler.record_projection(total_elapsed)
        
        return discrete_action, timing_info


def create_environment(scale_name: str, config_overrides: Dict = None) -> CityFlowMultiIntersectionEnv:
    """创建指定规模的环境"""
    scale_config = SCALE_CONFIGS[scale_name]
    config_path = scale_config["config_path"]
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    env_config = get_default_config(config_path)
    
    # 应用默认配置
    env_config["episode_length"] = 3600
    env_config["ctrl_interval"] = 10
    env_config["min_green"] = 10
    env_config["min_duration"] = 10
    env_config["max_duration"] = 60
    env_config["verbose_violations"] = False
    env_config["log_violations"] = True
    
    # 应用覆盖配置
    if config_overrides:
        env_config.update(config_overrides)
    
    env = CityFlowMultiIntersectionEnv(env_config)
    
    print(f"  创建环境: {scale_config['name']}")
    print(f"    路口数量: {env.num_intersections}")
    print(f"    状态维度: {env.observation_space.shape[0]}")
    print(f"    动作维度: {env.action_space.shape}")
    
    return env


def train_for_scale(scale_name: str, num_episodes: int, output_dir: str,
                    device: str = "cpu") -> Dict:
    """
    针对指定规模训练策略
    
    Args:
        scale_name: 规模名称 (City_3_5, City_5_10, City_10_10)
        num_episodes: 训练轮数
        output_dir: 输出目录
        device: 计算设备 ("cpu" 或 "cuda")
    
    Returns:
        训练结果字典
    """
    scale_config = SCALE_CONFIGS[scale_name]
    print(f"\n{'='*70}")
    print(f"训练规模: {scale_config['name']}")
    print(f"{'='*70}")
    
    # 创建环境
    env = create_environment(scale_name)
    
    # 获取空间维度
    state_size = env.observation_space.shape[0]
    continuous_action_size = env.num_intersections * 2
    
    # 创建网络
    actor = ActorNetwork(state_size, continuous_action_size).to(device)
    actor_target = ActorNetwork(state_size, continuous_action_size).to(device)
    actor_target.load_state_dict(actor.state_dict())
    
    critic = CriticNetwork(state_size, continuous_action_size).to(device)
    critic_target = CriticNetwork(state_size, continuous_action_size).to(device)
    critic_target.load_state_dict(critic.state_dict())
    
    # 优化器
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=CONFIG['lr_mu'])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CONFIG['lr_q'])
    
    # 噪声和经验回放
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(continuous_action_size))
    memory = ReplayBuffer()
    
    # 创建动作投影器
    duration_options = list(range(env.min_duration, env.max_duration + 1))
    action_projector = CityFlowActionProjector(
        num_intersections=env.num_intersections,
        num_phases=env.num_phases,
        num_duration_options=env.num_duration_options,
        min_green=env.min_green,
        duration_options=duration_options
    )
    
    # 训练记录
    episode_rewards = []
    episode_travel_times = []
    training_start = time.time()
    
    print(f"\n开始训练 ({num_episodes} episodes)...")
    
    for n_epi in range(num_episodes):
        s, info = env.reset()
        done = False
        episode_reward = 0
        ou_noise.reset()
        
        while not done:
            # 获取连续动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
                continuous_action = actor(state_tensor).squeeze(0).cpu().numpy()
            
            # 添加探索噪声
            noise_scale = max(0.1, 1.0 - n_epi / (num_episodes * 0.8))
            noise = ou_noise() * noise_scale
            continuous_action = np.clip(continuous_action + noise, 0, 1)
            
            # 投影到离散动作
            discrete_action = action_projector.project_with_safety_check(continuous_action, env)
            
            # 执行动作
            s_prime, reward, terminated, truncated, info = env.step(discrete_action)
            done = terminated or truncated
            
            # 存储经验
            memory.put((s, continuous_action, reward, s_prime, done))
            
            episode_reward += reward
            s = s_prime
        
        # 训练网络
        if memory.size() > CONFIG['memory_threshold']:
            for _ in range(CONFIG['training_iterations']):
                # 需要将数据移到正确设备
                s_batch, a_batch, r_batch, s_prime_batch, done_mask_batch = memory.sample(CONFIG['batch_size'])
                s_batch = s_batch.to(device)
                a_batch = a_batch.to(device)
                r_batch = r_batch.to(device)
                s_prime_batch = s_prime_batch.to(device)
                done_mask_batch = done_mask_batch.to(device)
                
                # Critic 更新
                with torch.no_grad():
                    target_actions = actor_target(s_prime_batch)
                    target_q = critic_target(s_prime_batch, target_actions)
                    target = r_batch + CONFIG['gamma'] * target_q * done_mask_batch
                
                current_q = critic(s_batch, a_batch)
                critic_loss = torch.nn.functional.mse_loss(current_q, target)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optimizer.step()
                
                # Actor 更新
                actor_loss = -critic(s_batch, actor(s_batch)).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()
            
            # 软更新目标网络
            soft_update(actor, actor_target)
            soft_update(critic, critic_target)
        
        # 记录统计
        episode_rewards.append(episode_reward)
        avg_travel_time = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_travel_time)
        
        # 打印进度
        if (n_epi + 1) % max(1, num_episodes // 10) == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_tt = np.mean(episode_travel_times[-10:]) if len(episode_travel_times) >= 10 else np.mean(episode_travel_times)
            print(f"  Episode {n_epi+1}/{num_episodes}: Reward={avg_reward:.1f}, AvgTravelTime={avg_tt:.1f}s")
    
    training_time = time.time() - training_start
    
    # 保存模型
    scale_output_dir = os.path.join(output_dir, scale_name)
    os.makedirs(scale_output_dir, exist_ok=True)
    
    model_path = os.path.join(scale_output_dir, "model.pt")
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_target_state_dict': actor_target.state_dict(),
        'critic_target_state_dict': critic_target.state_dict(),
        'config': CONFIG,
        'scale_name': scale_name,
        'num_intersections': env.num_intersections,
        'state_size': state_size,
        'continuous_action_size': continuous_action_size,
    }, model_path)
    print(f"  模型已保存: {model_path}")
    
    env.close()
    
    return {
        'scale_name': scale_name,
        'num_intersections': env.num_intersections,
        'state_size': state_size,
        'action_size': continuous_action_size,
        'training_time': training_time,
        'final_avg_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards),
        'final_avg_travel_time': np.mean(episode_travel_times[-10:]) if len(episode_travel_times) >= 10 else np.mean(episode_travel_times),
        'model_path': model_path,
        'actor': actor,
        'critic': critic,
    }


def benchmark_decision_time(scale_name: str, model_path: str, num_steps: int = 1000,
                           device: str = "cpu") -> Dict:
    """
    测试指定规模的决策时间
    
    Args:
        scale_name: 规模名称
        model_path: 模型路径
        num_steps: 测试步数
        device: 计算设备
    
    Returns:
        时间统计结果
    """
    scale_config = SCALE_CONFIGS[scale_name]
    print(f"\n{'='*70}")
    print(f"决策时间测试: {scale_config['name']}")
    print(f"{'='*70}")
    
    # 创建环境
    env = create_environment(scale_name)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    state_size = checkpoint['state_size']
    continuous_action_size = checkpoint['continuous_action_size']
    
    actor = ActorNetwork(state_size, continuous_action_size).to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    # 创建带时间测量的投影器
    profiler = TimingProfiler()
    duration_options = list(range(env.min_duration, env.max_duration + 1))
    timed_projector = TimedActionProjector(
        num_intersections=env.num_intersections,
        num_phases=env.num_phases,
        num_duration_options=env.num_duration_options,
        min_green=env.min_green,
        duration_options=duration_options,
        profiler=profiler
    )
    
    print(f"  测试步数: {num_steps}")
    print(f"  设备: {device}")
    print(f"  路口数量: {env.num_intersections}")
    
    # 预热
    s, _ = env.reset()
    for _ in range(10):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
            _ = actor(state_tensor)
    
    # 正式测试
    s, _ = env.reset()
    step_count = 0
    
    print(f"\n  开始测试...")
    
    while step_count < num_steps:
        # ====== 1. 神经网络推理 ======
        nn_start = time.perf_counter()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
            continuous_action = actor(state_tensor).squeeze(0).cpu().numpy()
        nn_elapsed = time.perf_counter() - nn_start
        profiler.record_nn_inference(nn_elapsed)
        
        # ====== 2. 动作投影（包含离散和连续映射）======
        total_decision_start = time.perf_counter()
        discrete_action, timing_info = timed_projector.project_with_timing(continuous_action, env)
        total_decision_elapsed = time.perf_counter() - total_decision_start + nn_elapsed
        profiler.record_total_decision(total_decision_elapsed)
        
        # 执行动作
        s_prime, reward, terminated, truncated, info = env.step(discrete_action)
        done = terminated or truncated
        
        s = s_prime
        step_count += 1
        
        if done:
            s, _ = env.reset()
        
        if step_count % (num_steps // 10) == 0:
            print(f"    进度: {step_count}/{num_steps}")
    
    env.close()
    
    # 获取统计结果
    stats = profiler.get_statistics()
    
    print(f"\n  Decision Time Statistics (ms):")
    print(f"  {'='*50}")
    print(f"  {'Component':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"  {'-'*50}")
    
    for name, values in [
        ("NN Inference", stats["nn_inference"]),
        ("Discrete Mapping", stats["discrete_mapping"]),
        ("Continuous Mapping", stats["continuous_mapping"]),
        ("Projection Total", stats["projection"]),
        ("Total Decision", stats["total_decision"]),
    ]:
        print(f"  {name:<20} {values['mean']:<10.4f} {values['std']:<10.4f} "
              f"{values['min']:<10.4f} {values['max']:<10.4f}")
    
    return {
        'scale_name': scale_name,
        'num_intersections': env.num_intersections,
        'num_steps': num_steps,
        'device': device,
        'timing_stats': stats,
    }


def run_full_benchmark(scales: List[str], num_train_episodes: int, num_test_steps: int,
                      output_dir: str, device: str = "cpu") -> Dict:
    """
    运行完整的多规模训练和时间测试
    
    Args:
        scales: 要测试的规模列表
        num_train_episodes: 每个规模的训练轮数
        num_test_steps: 时间测试步数
        output_dir: 输出目录
        device: 计算设备
    
    Returns:
        完整结果字典
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"scale_benchmark_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"# 多规模交通仿真训练与决策时间测试")
    print(f"# 输出目录: {run_dir}")
    print(f"# 设备: {device}")
    print(f"# 规模: {', '.join(scales)}")
    print(f"# 训练轮数: {num_train_episodes}")
    print(f"# 测试步数: {num_test_steps}")
    print(f"{'#'*70}")
    
    results = {
        'timestamp': timestamp,
        'device': device,
        'num_train_episodes': num_train_episodes,
        'num_test_steps': num_test_steps,
        'scales': {},
    }
    
    # 阶段1: 训练各规模策略
    print(f"\n{'='*70}")
    print(f"阶段 1: 训练各规模策略")
    print(f"{'='*70}")
    
    for scale_name in scales:
        if scale_name not in SCALE_CONFIGS:
            print(f"警告: 未知规模 {scale_name}, 跳过")
            continue
        
        try:
            train_result = train_for_scale(
                scale_name=scale_name,
                num_episodes=num_train_episodes,
                output_dir=run_dir,
                device=device
            )
            results['scales'][scale_name] = {
                'training': train_result,
            }
        except Exception as e:
            print(f"训练失败 ({scale_name}): {e}")
            import traceback
            traceback.print_exc()
    
    # 阶段2: 测试决策时间
    print(f"\n{'='*70}")
    print(f"阶段 2: 测试决策时间")
    print(f"{'='*70}")
    
    for scale_name in scales:
        if scale_name not in results['scales']:
            continue
        
        model_path = results['scales'][scale_name]['training']['model_path']
        
        try:
            timing_result = benchmark_decision_time(
                scale_name=scale_name,
                model_path=model_path,
                num_steps=num_test_steps,
                device=device
            )
            results['scales'][scale_name]['timing'] = timing_result
        except Exception as e:
            print(f"时间测试失败 ({scale_name}): {e}")
            import traceback
            traceback.print_exc()
    
    # 生成汇总报告
    print(f"\n{'='*70}")
    print(f"汇总报告")
    print(f"{'='*70}")
    
    summary_data = []
    for scale_name in scales:
        if scale_name not in results['scales']:
            continue
        
        scale_data = results['scales'][scale_name]
        train_data = scale_data.get('training', {})
        timing_data = scale_data.get('timing', {})
        timing_stats = timing_data.get('timing_stats', {})
        
        row = {
            'Scale': SCALE_CONFIGS[scale_name]['name'],
            'Intersections': train_data.get('num_intersections', 'N/A'),
            'State Dim': train_data.get('state_size', 'N/A'),
            'Action Dim': train_data.get('action_size', 'N/A'),
            'Train Time(s)': f"{train_data.get('training_time', 0):.1f}",
            'Final Reward': f"{train_data.get('final_avg_reward', 0):.1f}",
            'NN Infer(ms)': f"{timing_stats.get('nn_inference', {}).get('mean', 0):.4f}",
            'Disc Map(ms)': f"{timing_stats.get('discrete_mapping', {}).get('mean', 0):.4f}",
            'Cont Map(ms)': f"{timing_stats.get('continuous_mapping', {}).get('mean', 0):.4f}",
            'Total Dec(ms)': f"{timing_stats.get('total_decision', {}).get('mean', 0):.4f}",
        }
        summary_data.append(row)
    
    # 打印汇总表格
    if summary_data:
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # 保存汇总表格
        csv_path = os.path.join(run_dir, "summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n汇总表格已保存: {csv_path}")
    
    # 保存完整结果
    # 移除不可序列化的对象
    results_to_save = {
        'timestamp': results['timestamp'],
        'device': results['device'],
        'num_train_episodes': results['num_train_episodes'],
        'num_test_steps': results['num_test_steps'],
        'scales': {},
    }
    
    for scale_name, scale_data in results['scales'].items():
        results_to_save['scales'][scale_name] = {
            'training': {
                'scale_name': scale_data['training'].get('scale_name'),
                'num_intersections': scale_data['training'].get('num_intersections'),
                'state_size': scale_data['training'].get('state_size'),
                'action_size': scale_data['training'].get('action_size'),
                'training_time': scale_data['training'].get('training_time'),
                'final_avg_reward': scale_data['training'].get('final_avg_reward'),
                'final_avg_travel_time': scale_data['training'].get('final_avg_travel_time'),
                'model_path': scale_data['training'].get('model_path'),
            },
            'timing': scale_data.get('timing', {}),
        }
    
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    print(f"完整结果已保存: {json_path}")
    
    print(f"\n{'='*70}")
    print(f"所有测试完成!")
    print(f"输出目录: {run_dir}")
    print(f"{'='*70}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="多规模交通仿真训练与决策时间测试")
    
    parser.add_argument("--scales", type=str, nargs="+",
                       default=["City_3_5", "City_5_10", "City_10_10"],
                       help="要测试的规模列表")
    parser.add_argument("--train-episodes", type=int, default=50,
                       help="每个规模的训练轮数")
    parser.add_argument("--test-steps", type=int, default=1000,
                       help="时间测试步数")
    parser.add_argument("--output-dir", type=str,
                       default=os.path.join(PROJECT_ROOT, "outputs/scale_benchmark"),
                       help="输出目录")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["cpu", "cuda", "auto"],
                       help="计算设备")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    # 仅训练或仅测试模式
    parser.add_argument("--train-only", action="store_true",
                       help="仅训练，不测试时间")
    parser.add_argument("--benchmark-only", action="store_true",
                       help="仅测试时间（需要已有模型）")
    parser.add_argument("--model-dir", type=str,
                       help="模型目录（benchmark-only 模式需要）")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 确定设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if args.benchmark_only:
        # 仅测试时间模式
        if not args.model_dir:
            print("错误: benchmark-only 模式需要指定 --model-dir")
            sys.exit(1)
        
        for scale_name in args.scales:
            model_path = os.path.join(args.model_dir, scale_name, "model.pt")
            if os.path.exists(model_path):
                benchmark_decision_time(
                    scale_name=scale_name,
                    model_path=model_path,
                    num_steps=args.test_steps,
                    device=device
                )
            else:
                print(f"模型不存在: {model_path}")
    
    elif args.train_only:
        # 仅训练模式
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, f"train_only_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        for scale_name in args.scales:
            if scale_name in SCALE_CONFIGS:
                train_for_scale(
                    scale_name=scale_name,
                    num_episodes=args.train_episodes,
                    output_dir=run_dir,
                    device=device
                )
    
    else:
        # 完整模式：训练 + 时间测试
        run_full_benchmark(
            scales=args.scales,
            num_train_episodes=args.train_episodes,
            num_test_steps=args.test_steps,
            output_dir=args.output_dir,
            device=device
        )


if __name__ == "__main__":
    main()

