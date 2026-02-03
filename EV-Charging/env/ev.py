import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import random
import math

class EVChargingEnv(gym.Env):
    """电动汽车充电站调度环境"""
    
    def __init__(self, 
                 n_stations: int = 10,
                 p_max: float = 150.0,
                 max_steps: int = 288,
                 arrival_rate: float = 0.8):
        super().__init__()
        
        # 环境参数
        self.n_stations = n_stations  # 充电桩数量
        self.p_max = p_max  # 单个充电桩最大功率 (kW)
        self.max_steps = max_steps  # 一天的时间步数 (288 = 24h * 12)
        self.arrival_rate = arrival_rate  # 车辆到达率 (泊松分布参数)
        self.max_vehicles = math.ceil(n_stations * 1.5)  # 最大容纳车辆数，向上取整
        self.max_wait_time = 3  # 最大等待时间 (15min = 3步)
        self.max_linger_time = 1  # 充满后最大滞留时间 (5min = 1步)
        
        # 电价参数 (分时电价)
        self.peak_hours = [(8, 12), (18, 22)]  # 峰时时段
        self.peak_price = 1.5  # 峰时电价倍数
        self.valley_price = 0.8  # 谷时电价倍数
        self.base_price = 1.0  # 基础电价
        
        # 定义动作空间：离散部分选择充电桩和车辆，连续部分调节功率
        self.action_space = spaces.Dict({
            'station_id': spaces.Discrete(n_stations),
            'vehicle_id': spaces.Discrete(self.max_vehicles),
            'power': spaces.Box(low=50.0, high=150.0, shape=(1,), dtype=np.float32)
        })
        
        # 定义观察空间
        obs_dim = n_stations + self.max_vehicles * 4 + 2  # 充电桩状态 + 车辆状态 + 时间信息
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 添加充电记录
        self.charging_records = []
        
        # 初始化环境状态
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        
        # 充电桩状态 (1-可用, 0-占用)
        self.station_status = np.ones(self.n_stations, dtype=np.int32)
        self.station_vehicle = np.full(self.n_stations, -1, dtype=np.int32)  # 充电桩对应的车辆ID
        self.station_power = np.zeros(self.n_stations, dtype=np.float32)  # 充电桩当前功率
        self.station_lifetime = np.ones(self.n_stations, dtype=np.float32)  # 充电桩寿命
        
        # 车辆列表
        self.vehicles: List[Optional[Dict]] = [None] * self.max_vehicles
        self.num_vehicles = 0
        
        # 初始化时创建1辆汽车
        initial_vehicle = {
            'energy_required': np.random.uniform(20, 90),  # 需要充电量 (kWh)
            'energy_charged': 0.0,
            'wait_time': 0,
            'charging': False,
            'station_id': -1,
            'fully_charged': False,
            'linger_time': 0,
            'will_linger': random.random() < 0.6,  # 60%概率选择滞留（与_handle_arrivals保持一致）
            'arrival_step': 0,
            'initial_wait_time': 0
        }
        
        # 将初始车辆放在第一个位置
        self.vehicles[0] = initial_vehicle
        self.num_vehicles = 1
        
        # 统计信息
        self.total_energy_delivered = 0.0
        self.total_cost = 0.0
        self.total_lifetime_damage = 0.0
        
        # 重置奖励计算的历史值
        self._prev_energy = 0.0
        self._prev_cost = 0.0
        self._prev_damage = 0.0
        
        # 重置充电记录
        self.charging_records = []
        
        # 重置episode统计信息
        self.episode_arrivals = 1  # 包含初始车辆
        self.episode_charged_count = 0  # 本episode充满的车辆数
        
        return self._get_obs()
    
    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步环境交互"""
        # 处理新到达的车辆
        self._handle_arrivals()
        
        # 执行动作
        reward = self._execute_action(action)
        
        # 更新充电状态
        self._update_charging()
        
        # 处理离开的车辆
        self._handle_departures()
        
        # 更新时间
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # 自动跳过无法执行充电动作的时间步
        auto_advance_steps = 0
        auto_advance_reward = 0.0  # 累积自动跳步期间的奖励
        while not done and self._should_auto_advance():
            # 处理新到达的车辆
            self._handle_arrivals()
            
            # 更新充电状态
            self._update_charging()
            
            # 累积自动跳步期间的奖励（修复：不丢失充电期间的奖励）
            auto_advance_reward += self._calculate_step_reward()
            
            # 处理离开的车辆
            self._handle_departures()
            
            # 更新时间
            self.current_step += 1
            done = self.current_step >= self.max_steps
            auto_advance_steps += 1
            
            # 避免无限循环，最多连续跳过50步
            if auto_advance_steps >= 50:
                # print(f"Warning: Auto-advanced {auto_advance_steps} steps, stopping to prevent infinite loop")
                break
        
        # 将自动跳步期间的奖励加到总奖励中
        reward += auto_advance_reward
        
        # 记录自动跳步信息 - 屏蔽输出
        # if auto_advance_steps > 0:
        #     available_stations = self._count_available_stations()
        #     valid_vehicles = self._count_valid_vehicles()
        #     print(f"Auto-advanced {auto_advance_steps} steps to step {self.current_step}. "
        #           f"Now: {available_stations} stations, {valid_vehicles} vehicles")
        
        # 获取观察
        obs = self._get_obs()
        
        # 信息
        info = {
            'total_energy': self.total_energy_delivered,
            'total_cost': self.total_cost,
            'total_lifetime_damage': self.total_lifetime_damage,
            'num_vehicles': self.num_vehicles,
            'available_stations': self._count_available_stations(),
            'valid_vehicles': self._count_valid_vehicles(),
            # 添加约束违反信息
            'constraint_violation': getattr(self, '_last_constraint_violation', None),
            # 添加episode统计信息
            'episode_arrivals': self.episode_arrivals,  # 本episode总到达车辆数
            'episode_charged_count': self.episode_charged_count  # 本episode充满车辆数
        }
        
        return obs, reward, done, info
    
    def _handle_arrivals(self):
        """处理新到达的车辆"""
        # 使用泊松分布生成到达车辆数
        num_arrivals = np.random.poisson(self.arrival_rate)  # arrival_rate = 0.5
        
        for _ in range(num_arrivals):
            if self.num_vehicles >= self.max_vehicles:
                break
                
            # 生成新车辆
            energy_required = np.random.uniform(20, 90)  # 需要充电量 (kWh)
            vehicle = {
                'energy_required': energy_required,
                'energy_charged': 0.0,
                'wait_time': 0,
                'charging': False,
                'station_id': -1,
                'fully_charged': False,
                'linger_time': 0,
                'will_linger': random.random() < 0.6  # 60%概率选择滞留
            }
            
            # 找到空位放置车辆
            for i in range(self.max_vehicles):
                if self.vehicles[i] is None:
                    self.vehicles[i] = vehicle
                    self.vehicles[i]['arrival_step'] = self.current_step
                    self.vehicles[i]['initial_wait_time'] = 0
                    self.num_vehicles += 1
                    self.episode_arrivals += 1  # 统计本episode的到达车辆数
                    break
    
    def _execute_action(self, action: Dict) -> float:
        """执行动作并返回即时奖励"""
        station_id = action['station_id']
        vehicle_id = action['vehicle_id']
        power = action['power'][0]
        
        # 初始化约束违反信息
        constraint_violation_info = {
            'has_violation': False,
            'violation_type': None,
            'violation_details': None,
            'attempted_action': {
                'station_id': station_id,
                'vehicle_id': vehicle_id,
                'power': power
            }
        }
        
        # 在执行动作前进行车辆存在性的最后检查和修正
        if vehicle_id >= self.max_vehicles or self.vehicles[vehicle_id] is None:
            # 寻找替代车辆
            found_alternative = False
            original_vehicle_id = vehicle_id
            
            for i in range(self.max_vehicles):
                if (self.vehicles[i] is not None and 
                    not self.vehicles[i]['charging'] and 
                    not self.vehicles[i]['fully_charged']):
                    vehicle_id = i
                    found_alternative = True
                    break
            
            # 如果没有找到替代车辆，尝试找任何存在的车辆（即使状态不理想）
            if not found_alternative:
                for i in range(self.max_vehicles):
                    if self.vehicles[i] is not None:
                        vehicle_id = i
                        found_alternative = True
                        break
            
            # 记录约束违反信息
            if not found_alternative:
                constraint_violation_info.update({
                    'has_violation': True,
                    'violation_type': 'no_vehicles_available',
                    'violation_details': f'No vehicles available. Attempted vehicle_id: {original_vehicle_id}'
                })
                # 将约束违反信息存储到环境中，供info返回
                self._last_constraint_violation = constraint_violation_info
                return -10.0
            else:
                # 记录车辆替换信息
                constraint_violation_info.update({
                    'has_violation': True,
                    'violation_type': 'vehicle_not_available',
                    'violation_details': f'Vehicle {original_vehicle_id} not available, using alternative vehicle {vehicle_id}'
                })
        
        # 检查约束条件并记录详细违反信息
        violation_result = self._check_action_constraints(station_id, vehicle_id)
        if not violation_result['is_valid']:
            constraint_violation_info.update({
                'has_violation': True,
                'violation_type': violation_result['violation_type'],
                'violation_details': violation_result['violation_details']
            })
            # 将约束违反信息存储到环境中，供info返回
            self._last_constraint_violation = constraint_violation_info
            return -10.0
        
        # 分配充电桩
        vehicle = self.vehicles[vehicle_id]
        vehicle['charging'] = True
        vehicle['station_id'] = station_id
        
        self.station_status[station_id] = 0
        self.station_vehicle[station_id] = vehicle_id
        self.station_power[station_id] = np.clip(power, 50.0, self.p_max)  # 确保功率在有效范围内
        
        # 记录充电开始
        vehicle['charge_start_step'] = self.current_step
        # 初始化累计成本与寿命损伤（若重复分配同一车辆，重置本次会话）
        vehicle['accumulated_cost'] = 0.0
        vehicle['accumulated_damage'] = 0.0
        
        # 如果有轻微违反（如车辆替换），仍然记录但不惩罚
        if constraint_violation_info['has_violation']:
            self._last_constraint_violation = constraint_violation_info
        else:
            self._last_constraint_violation = None
        
        # 计算并返回奖励
        return self.calculate_reward()
    
    def _check_action_constraints(self, station_id: int, vehicle_id: int) -> Dict:
        """检查动作约束并返回详细信息"""
        result = {
            'is_valid': True,
            'violation_type': None,
            'violation_details': None
        }
        
        # 检查充电桩ID范围
        if station_id < 0 or station_id >= self.n_stations:
            result.update({
                'is_valid': False,
                'violation_type': 'station_id_out_of_range',
                'violation_details': f'Station ID {station_id} out of range [0, {self.n_stations-1}]'
            })
            # print(f"充电桩ID {station_id} 超出范围 [0, {self.n_stations-1}]")
            return result
            
        # 检查充电桩是否可用
        if self.station_status[station_id] == 0:
            result.update({
                'is_valid': False,
                'violation_type': 'station_occupied',
                'violation_details': f'Station {station_id} is already occupied'
            })
            # print(f"充电桩 {station_id} 不可用")
            return result
        
        # 检查车辆ID范围
        if vehicle_id < 0 or vehicle_id >= self.max_vehicles:
            result.update({
                'is_valid': False,
                'violation_type': 'vehicle_id_out_of_range',
                'violation_details': f'Vehicle ID {vehicle_id} out of range [0, {self.max_vehicles-1}]'
            })
            # print(f"车辆ID {vehicle_id} 超出范围 [0, {self.max_vehicles-1}]")
            return result
            
        # 检查车辆是否存在
        if self.vehicles[vehicle_id] is None:
            result.update({
                'is_valid': False,
                'violation_type': 'vehicle_not_exist',
                'violation_details': f'Vehicle {vehicle_id} does not exist'
            })
            # print(f"车辆 {vehicle_id} 不存在")
            return result
        
        # 检查车辆状态
        vehicle = self.vehicles[vehicle_id]
        if vehicle['charging']:
            result.update({
                'is_valid': False,
                'violation_type': 'vehicle_already_charging',
                'violation_details': f'Vehicle {vehicle_id} is already charging'
            })
            # print(f"车辆 {vehicle_id} 已在充电")
            return result
            
        if vehicle['fully_charged']:
            result.update({
                'is_valid': False,
                'violation_type': 'vehicle_fully_charged',
                'violation_details': f'Vehicle {vehicle_id} is already fully charged'
            })
            # print(f"车辆 {vehicle_id} 已充满")
            return result
        
        return result
    
    def _is_valid_action(self, station_id: int, vehicle_id: int) -> bool:
        """检查动作是否有效"""
        # 检查充电桩ID范围
        if station_id < 0 or station_id >= self.n_stations:
            # print(f"充电桩ID {station_id} 超出范围 [0, {self.n_stations-1}]")
            return False
            
        # 检查充电桩是否可用
        if self.station_status[station_id] == 0:
            # print(f"充电桩 {station_id} 不可用")
            return False
        
        # 检查车辆ID范围
        if vehicle_id < 0 or vehicle_id >= self.max_vehicles:
            # print(f"车辆ID {vehicle_id} 超出范围 [0, {self.max_vehicles-1}]")
            return False
            
        # 检查车辆是否存在
        if self.vehicles[vehicle_id] is None:
            # print(f"车辆 {vehicle_id} 不存在")
            return False
        
        # 检查车辆状态
        vehicle = self.vehicles[vehicle_id]
        if vehicle['charging']:
            # print(f"车辆 {vehicle_id} 已在充电")
            return False
            
        if vehicle['fully_charged']:
            # print(f"车辆 {vehicle_id} 已充满")
            return False
        
        return True
    
    def _update_charging(self):
        """更新充电状态"""
        # 🔍 充电时间计算关键部分
        time_step_hours = 5 / 60  # 5分钟转换为小时 (每个时间步 = 5分钟)
        current_hour = (self.current_step * 5 // 60) % 24  # 当前小时数
        
        # 计算当前电价
        price_multiplier = self._get_price_multiplier(current_hour)
        
        for i in range(self.n_stations):
            if self.station_status[i] == 0:  # 充电桩被占用
                vehicle_id = self.station_vehicle[i]
                
                # 安全检查：确保 vehicle_id 有效且车辆存在
                if vehicle_id < 0 or vehicle_id >= self.max_vehicles or self.vehicles[vehicle_id] is None:
                    # 清理无效的充电桩状态
                    self.station_status[i] = 1
                    self.station_vehicle[i] = -1
                    self.station_power[i] = 0.0
                    continue
                
                vehicle = self.vehicles[vehicle_id]
                power = self.station_power[i]  # 当前充电功率 (kW)
                
                # 🔍 每个时间步的充电量计算
                energy_charged = power * time_step_hours  # 能量 = 功率 × 时间 (kWh)
                energy_before = vehicle['energy_charged']
                vehicle['energy_charged'] += energy_charged
                self.total_energy_delivered += energy_charged
                
                # 计算成本
                cost = energy_charged * self.base_price * price_multiplier
                self.total_cost += cost
                
                # 计算寿命损伤 (功率越大，损伤越大)
                lifetime_damage = (power / self.p_max) ** 2 * 0.01
                self.station_lifetime[i] -= lifetime_damage
                self.total_lifetime_damage += lifetime_damage

                # 累计到车辆本次充电会话
                vehicle.setdefault('accumulated_cost', 0.0)
                vehicle.setdefault('accumulated_damage', 0.0)
                vehicle['accumulated_cost'] += cost
                vehicle['accumulated_damage'] += lifetime_damage*10
                
                # 🔍 检查是否充满电
                if vehicle['energy_charged'] >= vehicle['energy_required']:
                    vehicle['fully_charged'] = True
                    vehicle['charging'] = False  # 修复：充满后标记为不再充电
                    self.episode_charged_count += 1  # 统计本episode充满的车辆数
                    
                    # 🔍 计算充电总时间 (steps)
                    # 修复：确保 charge_start_step 存在，否则至少为 1 步
                    start_step = vehicle.get('charge_start_step', max(0, self.current_step - 1))
                    charging_duration = max(1, self.current_step - start_step)  # 至少 1 步
                    charging_time_minutes = charging_duration * 5  # 转换为分钟
                    
                    # 记录充电完成
                    self.charging_records.append({
                        'vehicle_id': vehicle_id,
                        'station_id': i,
                        'start_step': start_step,
                        'end_step': self.current_step,
                        'charging_duration_steps': charging_duration,
                        'charging_time_minutes': charging_time_minutes,
                        'power': power,
                        'energy': vehicle['energy_charged'],
                        'cost': vehicle.get('accumulated_cost', cost),
                        'damage_delta': vehicle.get('accumulated_damage', 0.0),
                        'wait_time': vehicle.get('initial_wait_time', 0)
                    })
    
    def _handle_departures(self):
        """处理离开的车辆"""
        for i in range(self.max_vehicles):
            if self.vehicles[i] is None:
                continue
            
            vehicle = self.vehicles[i]
            
            # 更新等待时间
            if not vehicle['charging'] and not vehicle['fully_charged']:
                # 先记录初始等待时间（修复：在累加前记录）
                if 'initial_wait_time' not in vehicle:
                    vehicle['initial_wait_time'] = vehicle['wait_time']
                vehicle['wait_time'] += 1
                # 超过最大等待时间离开
                if vehicle['wait_time'] > self.max_wait_time:
                    self.vehicles[i] = None
                    self.num_vehicles -= 1
            
            # 处理充满电的车辆
            elif vehicle['fully_charged']:
                station_id = vehicle['station_id']
                
                # 释放充电桩
                if station_id >= 0:
                    self.station_status[station_id] = 1
                    self.station_vehicle[station_id] = -1
                    self.station_power[station_id] = 0.0
                    vehicle['station_id'] = -1
                
                # 处理滞留
                if vehicle['will_linger'] and vehicle['linger_time'] < self.max_linger_time:
                    vehicle['linger_time'] += 1
                else:
                    # 车辆离开
                    self.vehicles[i] = None
                    self.num_vehicles -= 1
    
    def _get_price_multiplier(self, hour: int) -> float:
        """获取电价倍数"""
        for start, end in self.peak_hours:
            if start <= hour < end:
                return self.peak_price
        
        # 谷时 (23:00 - 7:00)
        if hour >= 23 or hour < 7:
            return self.valley_price
        
        return self.base_price
    
    def _get_obs(self) -> np.ndarray:
        """获取观察状态"""
        obs = []
        
        # 充电桩状态
        obs.extend(self.station_status.tolist())
        
        # 车辆状态
        for i in range(self.max_vehicles):
            if self.vehicles[i] is None:
                obs.extend([0, 0, 0, 0])
            else:
                vehicle = self.vehicles[i]
                obs.extend([
                    vehicle['energy_required'] - vehicle['energy_charged'],
                    vehicle['wait_time'],
                    1 if vehicle['charging'] else 0,
                    1 if vehicle['fully_charged'] else 0
                ])
        
        # 时间信息
        obs.append(self.current_step / self.max_steps)
        obs.append((self.current_step * 5 // 60) % 24 / 24)  # 当前小时
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_step_reward(self) -> float:
        """计算单步奖励（内部使用，用于自动跳步期间的奖励累积）"""
        # 获取当前步的即时奖励
        prev_energy = getattr(self, '_prev_energy', 0.0)
        prev_cost = getattr(self, '_prev_cost', 0.0)
        prev_damage = getattr(self, '_prev_damage', 0.0)
        
        # 计算增量
        energy_delta = self.total_energy_delivered - prev_energy
        cost_delta = self.total_cost - prev_cost
        damage_delta = self.total_lifetime_damage - prev_damage
        
        # 更新之前的值
        self._prev_energy = self.total_energy_delivered
        self._prev_cost = self.total_cost
        self._prev_damage = self.total_lifetime_damage
        
        # 计算奖励
        energy_reward = energy_delta * 1.5  # 收益
        cost_penalty = -cost_delta  # 成本
        lifetime_penalty = -damage_delta * 100  # 寿命损伤
        
        return energy_reward + cost_penalty + lifetime_penalty
    
    def calculate_reward(self) -> float:
        """计算奖励函数（对外接口）"""
        return self._calculate_step_reward()
    
    def _should_auto_advance(self) -> bool:
        """判断是否应该自动推进时间步"""
        # 统计可用充电桩数量
        available_stations = self._count_available_stations()
        
        # 统计有效车辆数量（需要充电且未充电的车辆）
        valid_vehicles = self._count_valid_vehicles()
        
        # 可调度的场景：有空闲充电桩 AND 有未充电汽车
        # 如果这两个条件不同时满足，则需要自动推进时间步
        schedulable = available_stations > 0 and valid_vehicles > 0
        
        # 返回True表示需要自动推进（非可调度场景）
        return not schedulable
    
    def _count_available_stations(self) -> int:
        """统计可用充电桩数量"""
        return sum(1 for i in range(self.n_stations) if self.station_status[i] == 1)
    
    def _count_valid_vehicles(self) -> int:
        """统计有效车辆数量（需要充电且未充电的车辆）"""
        count = 0
        for i in range(self.max_vehicles):
            if self.vehicles[i] is not None:
                vehicle = self.vehicles[i]
                if not vehicle['charging'] and not vehicle['fully_charged']:
                    count += 1
        return count
