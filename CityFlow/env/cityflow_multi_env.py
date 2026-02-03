"""
CityFlow Multi-Intersection Traffic Signal Control Environment

This is a multi-intersection traffic signal control RL environment built on CityFlow.
It implements the OpenAI Gymnasium interface and supports a single agent controlling
signals for multiple intersections.

Reference: https://cityflow.readthedocs.io/en/latest/start.html
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import cityflow
except ImportError:
    raise ImportError(
        "CityFlow is not installed. Please install it following: "
        "https://cityflow.readthedocs.io/en/latest/install.html"
    )


class CityFlowMultiIntersectionEnv(gym.Env):
    """
    Multi-intersection traffic signal control RL environment based on CityFlow.
    
    Features:
    1. A single agent controls multiple intersections.
    2. Supports action constraints and signal-safety constraints.
    3. Uses the Gymnasium API (reset returns (obs, info); step returns (obs, reward, done, truncated, info)).
    
    Attributes:
        num_intersections: number of intersections
        num_phases: number of phases per intersection
        ctrl_interval: control interval (seconds)
        min_green: minimum green duration (seconds)
        episode_length: simulation length per episode (seconds)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: Dict[str, Any],
        render_mode: Optional[str] = None
    ):
        """
        Initialize the environment.
        
        Args:
            config: Config dict with keys:
                - cityflow_config_file: path to the CityFlow config file
                - thread_num: number of threads (default: 1)
                - ctrl_interval: control interval (default: 10 seconds)
                - episode_length: episode simulation length (default: 3600 seconds)
                - min_green: minimum green duration (default: 10 seconds)
                - max_phase_time: maximum phase duration (default: 60 seconds; used for normalization)
                - max_queue_per_dir: max queue length per direction (default: 50; used for normalization)
                - alpha: waiting penalty coefficient (default: 1.0)
                - beta: throughput reward coefficient (default: 5.0)
                - gamma: phase switching penalty coefficient (default: 0.1)
                - delta: constraint violation penalty coefficient (default: 0.5)
            render_mode: render mode
        """
        super().__init__()
        
        # ================== Basic config ==================
        self.config = config
        self.cityflow_config_file = config.get("cityflow_config_file", "./config.json")
        self.thread_num = config.get("thread_num", 1)
        self.ctrl_interval = config.get("ctrl_interval", 10)
        self.episode_length = config.get("episode_length", 3600)
        self.min_green = config.get("min_green", 10)
        self.max_phase_time = config.get("max_phase_time", 60)
        self.max_queue_per_dir = config.get("max_queue_per_dir", 50)
        
        # ================== Green duration options (1-second granularity) ==================
        # Range: [min_duration, max_duration], step = 1 second
        self.min_duration = config.get("min_duration", 10)  # min green duration (s)
        self.max_duration = config.get("max_duration", 60)  # max green duration (s)
        # Generate continuous duration options: 10, 11, 12, ..., 60
        self.duration_options = list(range(self.min_duration, self.max_duration + 1))
        self.num_duration_options = len(self.duration_options)  # = max_duration - min_duration + 1
        
        # ================== Reward hyperparameters ==================
        self.alpha = config.get("alpha", 1.0)      # waiting penalty coefficient
        self.beta = config.get("beta", 5.0)        # throughput reward coefficient
        self.gamma = config.get("gamma", 0.1)      # switching penalty coefficient
        self.delta = config.get("delta", 0.5)      # constraint-violation penalty coefficient
        
        # ================== Constraint-violation logging ==================
        self.verbose_violations = config.get("verbose_violations", False)  # print violation details
        self.log_violations = config.get("log_violations", True)           # record violations
        self.violation_log: List[Dict] = []  # violation log list
        
        self.render_mode = render_mode
        
        # ================== Parse road network ==================
        self._parse_roadnet()
        
        # ================== Define action/observation spaces ==================
        # Action space: MultiDiscrete. For each intersection choose [phase, green_duration]
        # Format: [phase_0, duration_0, phase_1, duration_1, ..., phase_n, duration_n]
        action_dims = []
        for _ in range(self.num_intersections):
            action_dims.append(self.num_phases)            # phase choice
            action_dims.append(self.num_duration_options)  # green duration choice
        self.action_space = spaces.MultiDiscrete(action_dims)
        
        # Observation space: Box, 7 features per intersection
        # [queue_N, queue_E, queue_S, queue_W, phase_norm, elapsed_norm, target_duration_norm]
        obs_dim = self.num_intersections * 7
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # ================== Initialize state variables ==================
        self.eng: Optional[cityflow.Engine] = None
        self.current_time = 0.0
        self.phase_elapsed: Dict[str, float] = {}  # elapsed time for current phase (per intersection)
        self.current_phases: Dict[str, int] = {}   # current phase (per intersection)
        self.prev_vehicle_ids: set = set()         # vehicle IDs from previous step
        self.last_num_switches = 0                  # number of switches in previous step
        self.constraint_violations: Dict[str, int] = {}  # constraint violation counters
        
        # Valid phase cache (reserved for future extension)
        # valid_phases[intersection_id][phase_index] = True/False
        self.valid_phases: Dict[str, List[bool]] = {}
        for inter_id in self.intersection_ids:
            self.valid_phases[inter_id] = [True] * self.num_phases
        
    def _parse_roadnet(self):
        """
        Parse the road network file and extract:
        1. Intersection IDs controlled by signals
        2. Number of phases per intersection
        3. Incoming lanes per direction for each intersection
        """
        # Read the CityFlow config file to locate the roadnet file
        with open(self.cityflow_config_file, 'r') as f:
            cityflow_config = json.load(f)
        
        roadnet_dir = cityflow_config.get("dir", "")
        roadnet_file = cityflow_config.get("roadnetFile", "roadnet.json")
        
        # Build candidate paths for the roadnet file
        config_dir = os.path.dirname(os.path.abspath(self.cityflow_config_file))
        
        # Try multiple path resolutions
        candidate_paths = []
        
        if roadnet_dir:
            if os.path.isabs(roadnet_dir):
                # Absolute path
                candidate_paths.append(os.path.join(roadnet_dir, roadnet_file))
            else:
                # Relative: to config file directory
                candidate_paths.append(os.path.join(config_dir, roadnet_file))
                # Relative: to parent of config dir (CityFlow working dir)
                parent_dir = os.path.dirname(config_dir)
                candidate_paths.append(os.path.join(parent_dir, roadnet_dir, roadnet_file))
                # Relative: to current working directory
                candidate_paths.append(os.path.join(roadnet_dir, roadnet_file))
        else:
            candidate_paths.append(os.path.join(config_dir, roadnet_file))
        
        # Pick the first existing path
        roadnet_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                roadnet_path = path
                break
        
        if roadnet_path is None:
            raise FileNotFoundError(
                f"Cannot find roadnet file. Tried paths: {candidate_paths}"
            )
        
        with open(roadnet_path, 'r') as f:
            roadnet_data = json.load(f)
        
        # ================== Extract non-virtual intersections ==================
        self.intersection_ids: List[str] = []
        self.intersection_info: Dict[str, Dict] = {}
        
        for intersection in roadnet_data.get("intersections", []):
            # Only handle non-virtual intersections (signal-controlled)
            if not intersection.get("virtual", True):
                inter_id = intersection["id"]
                self.intersection_ids.append(inter_id)
                
                # Get phase info for this intersection
                traffic_light = intersection.get("trafficLight", {})
                lightphases = traffic_light.get("lightphases", [])
                
                self.intersection_info[inter_id] = {
                    "num_phases": len(lightphases),
                    "roads": intersection.get("roads", []),
                    "lightphases": lightphases
                }
        
        # Sort intersection IDs for determinism
        self.intersection_ids.sort()
        self.num_intersections = len(self.intersection_ids)
        
        # Assume all intersections have the same number of phases (read from the first one)
        if self.num_intersections > 0:
            self.num_phases = self.intersection_info[self.intersection_ids[0]]["num_phases"]
        else:
            self.num_phases = 8  # default
        
        # ================== Parse incoming lanes per intersection and direction ==================
        self.in_lanes: Dict[str, Dict[str, List[str]]] = {}
        
        # Parse roads
        roads_data = roadnet_data.get("roads", [])
        self.road_info: Dict[str, Dict] = {}
        for road in roads_data:
            road_id = road["id"]
            # Get lane info
            lanes = road.get("lanes", [])
            lane_ids = [f"{road_id}_{i}" for i in range(len(lanes))]
            
            # Get road endpoints
            start_inter = road.get("startIntersection", "")
            end_inter = road.get("endIntersection", "")
            
            self.road_info[road_id] = {
                "lanes": lane_ids,
                "start": start_inter,
                "end": end_inter
            }
        
        # Infer direction from road naming conventions
        # Typically for road_X_Y_D: D=0(E), D=1(N), D=2(W), D=3(S)
        # Incoming direction is opposite to outgoing direction
        for inter_id in self.intersection_ids:
            self.in_lanes[inter_id] = {"N": [], "E": [], "S": [], "W": []}
            roads = self.intersection_info[inter_id]["roads"]
            
            for road_id in roads:
                if road_id not in self.road_info:
                    continue
                
                road = self.road_info[road_id]
                # Incoming lane: road whose end node is the current intersection
                if road["end"] == inter_id:
                    lanes = road["lanes"]
                    
                    # Infer direction from road name: road_X_Y_D
                    parts = road_id.split("_")
                    if len(parts) >= 4:
                        direction_code = int(parts[-1])
                        # Incoming direction (where vehicles come from)
                        # D=0 (eastbound): comes from west -> W
                        # D=1 (northbound): comes from south -> S
                        # D=2 (westbound): comes from east -> E
                        # D=3 (southbound): comes from north -> N
                        direction_map = {0: "W", 1: "S", 2: "E", 3: "N"}
                        direction = direction_map.get(direction_code, "N")
                        self.in_lanes[inter_id][direction].extend(lanes)
        
        print(f"[INFO] Parsed: {self.num_intersections} intersections, "
              f"{self.num_phases} phases per intersection")
    
    def _create_engine(self) -> cityflow.Engine:
        """Create a CityFlow simulation engine."""
        # Save current working directory
        original_cwd = os.getcwd()
        # Switch to the config directory (CityFlow's `dir` is relative to CWD)
        config_dir = os.path.dirname(os.path.abspath(self.cityflow_config_file))
        os.chdir(config_dir)
        try:
            # Create engine using the config filename
            config_filename = os.path.basename(self.cityflow_config_file)
            engine = cityflow.Engine(config_filename, thread_num=self.thread_num)
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
        return engine
    
    def _sanitize_action(self, action) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and sanitize the action.
        
        Args:
            action: Input action in the format [phase_0, duration_0, phase_1, duration_1, ...]
            
        Returns:
            (phases, durations): 
                - phases: phase selection per intersection (numpy array)
                - durations: green-duration index per intersection (numpy array)
        """
        sanitized_info = {"clipped_phase": 0, "clipped_duration": 0, "type_error": False}
        expected_len = self.num_intersections * 2  # 2 values per intersection
        
        # Type check
        try:
            action = np.array(action, dtype=np.int64).flatten()
        except (ValueError, TypeError):
            # Cannot convert; return default action
            sanitized_info["type_error"] = True
            phases = np.zeros(self.num_intersections, dtype=np.int64)
            durations = np.zeros(self.num_intersections, dtype=np.int64)
            for i, inter_id in enumerate(self.intersection_ids):
                phases[i] = self.current_phases.get(inter_id, 0)
                durations[i] = 0  # default: shortest duration
            self._last_sanitize_info = sanitized_info
            return phases, durations
        
        # Length check
        if len(action) < expected_len:
            # Pad with zeros
            padded = np.zeros(expected_len, dtype=np.int64)
            padded[:len(action)] = action
            action = padded
        elif len(action) > expected_len:
            action = action[:expected_len]
        
        # Parse action: alternating phase and duration
        phases = np.zeros(self.num_intersections, dtype=np.int64)
        durations = np.zeros(self.num_intersections, dtype=np.int64)
        
        for i in range(self.num_intersections):
            # Phase
            phase_idx = i * 2
            original_phase = action[phase_idx]
            phases[i] = np.clip(action[phase_idx], 0, self.num_phases - 1)
            if phases[i] != original_phase:
                sanitized_info["clipped_phase"] += 1
            
            # Green duration
            duration_idx = i * 2 + 1
            original_duration = action[duration_idx]
            durations[i] = np.clip(action[duration_idx], 0, self.num_duration_options - 1)
            if durations[i] != original_duration:
                sanitized_info["clipped_duration"] += 1
        
        self._last_sanitize_info = sanitized_info
        return phases, durations
    
    def _get_duration_seconds(self, duration_index: int) -> int:
        """Convert a green-duration index to seconds."""
        return self.duration_options[duration_index]
    
    def _apply_action(self, phases: np.ndarray, durations: np.ndarray):
        """
        Apply actions to the CityFlow simulation with constraint checks.
        
        Args:
            phases: target phases per intersection (numpy array)
            durations: green-duration indices per intersection (numpy array)
        """
        self.last_num_switches = 0
        sanitize_info = getattr(self, '_last_sanitize_info', {})
        self.constraint_violations = {
            "min_green": 0,
            "target_duration": 0,
            "invalid_phase": 0,
            "action_clipped": sanitize_info.get('clipped_phase', 0) + sanitize_info.get('clipped_duration', 0)
        }
        
        step_violations = []  # violations in this step
        
        for i, inter_id in enumerate(self.intersection_ids):
            desired_phase = int(phases[i])
            desired_duration_idx = int(durations[i])
            desired_duration = self._get_duration_seconds(desired_duration_idx)
            
            # Current phase and elapsed time
            cur_phase = self.current_phases.get(inter_id, 0)
            elapsed = self.phase_elapsed.get(inter_id, 0.0)
            target_duration = self.target_durations.get(inter_id, self.duration_options[0])
            
            # ================== Phase validity check ==================
            if not self.valid_phases[inter_id][desired_phase]:
                # Invalid phase -> keep current phase
                self.constraint_violations["invalid_phase"] += 1
                violation_detail = {
                    "type": "invalid_phase",
                    "intersection": inter_id,
                    "desired_phase": desired_phase,
                    "current_phase": cur_phase,
                    "time": self.current_time
                }
                step_violations.append(violation_detail)
                if self.verbose_violations:
                    print(f"[VIOLATION] Invalid phase: intersection {inter_id}, "
                          f"desired {desired_phase} is invalid; keep {cur_phase}")
                desired_phase = cur_phase
            
            # ================== Phase switching logic ==================
            if desired_phase != cur_phase:
                # Agent requests a phase switch
                
                # Check minimum green constraint
                if elapsed < self.min_green:
                    self.constraint_violations["min_green"] += 1
                    violation_detail = {
                        "type": "min_green",
                        "intersection": inter_id,
                        "desired_phase": desired_phase,
                        "current_phase": cur_phase,
                        "elapsed": elapsed,
                        "min_green": self.min_green,
                        "time": self.current_time
                    }
                    step_violations.append(violation_detail)
                    if self.verbose_violations:
                        print(f"[VIOLATION] Min green: intersection {inter_id}, "
                              f"phase {cur_phase} elapsed {elapsed:.1f}s < {self.min_green}s; "
                              f"reject switch to {desired_phase}")
                    # Keep current phase
                    self.phase_elapsed[inter_id] = elapsed + self.ctrl_interval
                    
                # Check target-duration constraint
                elif elapsed < target_duration:
                    self.constraint_violations["target_duration"] += 1
                    violation_detail = {
                        "type": "target_duration",
                        "intersection": inter_id,
                        "desired_phase": desired_phase,
                        "current_phase": cur_phase,
                        "elapsed": elapsed,
                        "target_duration": target_duration,
                        "time": self.current_time
                    }
                    step_violations.append(violation_detail)
                    if self.verbose_violations:
                        print(f"[VIOLATION] Target duration: intersection {inter_id}, "
                              f"phase {cur_phase} elapsed {elapsed:.1f}s < target {target_duration}s; "
                              f"reject switch to {desired_phase}")
                    # Keep current phase
                    self.phase_elapsed[inter_id] = elapsed + self.ctrl_interval
                    
                else:
                    # Allow switching
                    self.eng.set_tl_phase(inter_id, desired_phase)
                    self.current_phases[inter_id] = desired_phase
                    self.phase_elapsed[inter_id] = self.ctrl_interval
                    self.target_durations[inter_id] = desired_duration  # set target duration for new phase
                    self.last_num_switches += 1
                    
                    if self.verbose_violations:
                        print(f"[INFO] Intersection {inter_id}: switched to phase {desired_phase}, "
                              f"target green duration {desired_duration}s")
            else:
                # Keep current phase; may update target duration
                self.phase_elapsed[inter_id] = elapsed + self.ctrl_interval
                # If agent specifies a new duration, update target duration (does not bypass min-green)
                if desired_duration != target_duration:
                    self.target_durations[inter_id] = max(desired_duration, elapsed)
        
        # Record action-clipping violations
        if self.constraint_violations["action_clipped"] > 0:
            violation_detail = {
                "type": "action_clipped",
                "count": self.constraint_violations["action_clipped"],
                "time": self.current_time
            }
            step_violations.append(violation_detail)
            if self.verbose_violations:
                print(f"[VIOLATION] Action clipped: {self.constraint_violations['action_clipped']} actions were clipped")
        
        # Accumulate totals
        self.total_violations["min_green"] += self.constraint_violations["min_green"]
        self.total_violations["target_duration"] += self.constraint_violations["target_duration"]
        self.total_violations["invalid_phase"] += self.constraint_violations["invalid_phase"]
        self.total_violations["action_clipped"] += self.constraint_violations["action_clipped"]
        
        # Save to log
        if self.log_violations and step_violations:
            self.violation_log.extend(step_violations)
    
    def _get_obs(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Normalized observation vector with dim = num_intersections * 7.
            Per-intersection features: [queue_N, queue_E, queue_S, queue_W, phase_norm, elapsed_norm, target_duration_norm]
        """
        # Get waiting vehicle counts on all lanes
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        
        obs_list = []
        max_duration = max(self.duration_options)  # normalize target duration
        
        for inter_id in self.intersection_ids:
            # ================== Queues by direction ==================
            queue_N = sum(lane_waiting.get(lane, 0) 
                        for lane in self.in_lanes[inter_id]["N"])
            queue_E = sum(lane_waiting.get(lane, 0) 
                        for lane in self.in_lanes[inter_id]["E"])
            queue_S = sum(lane_waiting.get(lane, 0) 
                        for lane in self.in_lanes[inter_id]["S"])
            queue_W = sum(lane_waiting.get(lane, 0) 
                        for lane in self.in_lanes[inter_id]["W"])
            
            # Normalize queue lengths
            queue_N_norm = min(queue_N / self.max_queue_per_dir, 1.0)
            queue_E_norm = min(queue_E / self.max_queue_per_dir, 1.0)
            queue_S_norm = min(queue_S / self.max_queue_per_dir, 1.0)
            queue_W_norm = min(queue_W / self.max_queue_per_dir, 1.0)
            
            # ================== Current phase (normalized) ==================
            cur_phase = self.current_phases.get(inter_id, 0)
            phase_norm = cur_phase / max(self.num_phases - 1, 1)
            
            # ================== Phase elapsed time (normalized) ==================
            elapsed = self.phase_elapsed.get(inter_id, 0.0)
            elapsed_norm = min(elapsed / self.max_phase_time, 1.0)
            
            # ================== Target green duration (normalized) ==================
            target_duration = self.target_durations.get(inter_id, self.duration_options[0])
            target_duration_norm = target_duration / max_duration
            
            # Compose per-intersection feature vector (7 dims)
            inter_obs = [
                queue_N_norm, queue_E_norm, queue_S_norm, queue_W_norm,
                phase_norm, elapsed_norm, target_duration_norm
            ]
            obs_list.extend(inter_obs)
        
        return np.array(obs_list, dtype=np.float32)
    
    def _get_reward(self) -> float:
        """
        Compute reward.
        
        reward = -α * total_waiting + β * new_departures - γ * num_switches - δ * constraint_violations
        
        Returns:
            Reward value for the current step.
        """
        # ================== Waiting penalty ==================
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        total_waiting = sum(lane_waiting.values())
        
        # ================== Departure reward ==================
        current_vehicle_ids = set(self.eng.get_vehicles(include_waiting=True))
        # Departed vehicles = present in previous step but not in current step
        departed_vehicles = self.prev_vehicle_ids - current_vehicle_ids
        new_departures = len(departed_vehicles)
        
        # ================== Switching penalty ==================
        num_switches = self.last_num_switches
        
        # ================== Constraint-violation penalty ==================
        total_violations = sum(self.constraint_violations.values())
        
        # Total reward
        reward = (
            - self.alpha * (total_waiting / max(self.num_intersections, 1))  # normalize by intersections
            + self.beta * new_departures
            - self.gamma * num_switches
            - self.delta * total_violations
        )
        
        return reward
    
    def _check_done(self) -> Tuple[bool, bool]:
        """
        Check whether the episode is done.
        
        Returns:
            (terminated, truncated):
                - terminated: ended due to a terminal condition
                - truncated: ended due to time limit
        """
        terminated = False
        truncated = False
        
        # Time limit
        if self.current_time >= self.episode_length:
            truncated = True
        
        # Optional: other termination conditions (e.g., no vehicles and no new arrivals)
        # vehicle_count = self.eng.get_vehicle_count()
        # if vehicle_count == 0 and self.current_time > some_threshold:
        #     terminated = True
        
        return terminated, truncated
    
    def _update_intersection_flow(self):
        """
        Update per-intersection flow statistics.
        
        Flow calculation:
        - Compare vehicles on incoming lanes between current and previous step to estimate passed vehicles
        - Also record waiting vehicles and total vehicles per intersection
        """
        # Vehicles on each lane
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        
        for inter_id in self.intersection_ids:
            # All incoming lanes for this intersection
            all_in_lanes = []
            for direction in ["N", "E", "S", "W"]:
                all_in_lanes.extend(self.in_lanes[inter_id].get(direction, []))
            
            # Vehicles currently on incoming lanes
            current_vehicles = set()
            for lane in all_in_lanes:
                current_vehicles.update(lane_vehicles.get(lane, []))
            
            # Vehicles on incoming lanes in previous step
            prev_vehicles = self.prev_lane_vehicles.get(inter_id, set())
            
            # Passed vehicles = present previously but not now (passed or left)
            vehicles_passed = prev_vehicles - current_vehicles
            
            # Update stats
            self.intersection_flow[inter_id]["total_vehicles_passed"] += len(vehicles_passed)
            
            # Current waiting vehicles
            waiting_count = sum(lane_waiting.get(lane, 0) for lane in all_in_lanes)
            self.intersection_flow[inter_id]["total_waiting_vehicles"] += waiting_count
            
            # Current total vehicles
            vehicle_count = sum(lane_vehicle_count.get(lane, 0) for lane in all_in_lanes)
            self.intersection_flow[inter_id]["total_vehicle_count"] += vehicle_count
            
            # Step count
            self.intersection_flow[inter_id]["step_count"] += 1
            
            # Update previous vehicle record
            self.prev_lane_vehicles[inter_id] = current_vehicles
    
    def get_intersection_flow_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get per-intersection flow statistics.
        
        Returns:
            Dict with per-intersection statistics:
            - throughput: total vehicles passed
            - avg_throughput_per_step: average vehicles passed per step
            - avg_waiting: average waiting vehicles
            - avg_vehicle_count: average vehicle count
        """
        stats = {}
        for inter_id in self.intersection_ids:
            flow = self.intersection_flow[inter_id]
            step_count = max(flow["step_count"], 1)
            
            stats[inter_id] = {
                "throughput": flow["total_vehicles_passed"],
                "avg_throughput_per_step": flow["total_vehicles_passed"] / step_count,
                "avg_waiting": flow["total_waiting_vehicles"] / step_count,
                "avg_vehicle_count": flow["total_vehicle_count"] / step_count,
            }
        return stats
    
    def print_intersection_flow_summary(self):
        """Print per-intersection flow statistics summary."""
        stats = self.get_intersection_flow_stats()
        
        print(f"\n{'='*70}")
        print("Per-intersection flow statistics")
        print(f"{'='*70}")
        print(f"{'Intersection':<20} {'TotalPassed':<12} {'AvgPassed/Step':<14} {'AvgWaiting':<12} {'AvgVehicles':<12}")
        print(f"{'-'*70}")
        
        total_throughput = 0
        total_avg_waiting = 0
        total_avg_vehicle_count = 0
        
        for inter_id in self.intersection_ids:
            s = stats[inter_id]
            total_throughput += s["throughput"]
            total_avg_waiting += s["avg_waiting"]
            total_avg_vehicle_count += s["avg_vehicle_count"]
            print(f"{inter_id:<20} {s['throughput']:<12.0f} {s['avg_throughput_per_step']:<12.2f} "
                  f"{s['avg_waiting']:<12.2f} {s['avg_vehicle_count']:<12.2f}")
        
        n_intersections = max(len(self.intersection_ids), 1)
        print(f"{'-'*70}")
        print(f"{'Total/Avg':<20} {total_throughput:<12.0f} "
              f"{total_throughput/n_intersections:<12.2f} "
              f"{total_avg_waiting/n_intersections:<12.2f} "
              f"{total_avg_vehicle_count/n_intersections:<12.2f}")
        print(f"{'='*70}\n")
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get extra info.
        
        Returns:
            Dict containing statistics.
        """
        # Flow stats
        flow_stats = self.get_intersection_flow_stats() if hasattr(self, 'intersection_flow') else {}
        
        return {
            "current_time": self.current_time,
            "vehicle_count": self.eng.get_vehicle_count(),
            "average_travel_time": self.eng.get_average_travel_time(),
            "num_switches": self.last_num_switches,
            "constraint_violations": self.constraint_violations.copy(),
            "total_violations": getattr(self, "total_violations", {}).copy(),
            "violation_count": len(getattr(self, "violation_log", [])),
            "sanitize_info": getattr(self, "_last_sanitize_info", {}),
            "intersection_flow": flow_stats,
        }
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """
        Get constraint violation summary.
        
        Returns:
            Dict with constraint violation stats.
        """
        return {
            "total_violations": self.total_violations.copy(),
            "violation_count": len(self.violation_log),
            "violations_by_type": {
                "min_green": self.total_violations.get("min_green", 0),
                "target_duration": self.total_violations.get("target_duration", 0),
                "invalid_phase": self.total_violations.get("invalid_phase", 0),
                "action_clipped": self.total_violations.get("action_clipped", 0),
            }
        }
    
    def save_violation_log(self, filepath: str):
        """
        Save the constraint violation log to a file.
        
        Args:
            filepath: output file path
        """
        log_data = {
            "summary": self.get_violation_summary(),
            "violations": self.violation_log
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Constraint violation log saved to: {filepath}")
    
    def print_violation_summary(self):
        """Print constraint violation summary."""
        summary = self.get_violation_summary()
        total = sum(summary["violations_by_type"].values())
        print(f"\n{'='*50}")
        print("Constraint violation summary")
        print(f"{'='*50}")
        print(f"  Total violations: {total}")
        print(f"  - Min green: {summary['violations_by_type']['min_green']}")
        print(f"  - Target duration: {summary['violations_by_type']['target_duration']}")
        print(f"  - Invalid phase: {summary['violations_by_type']['invalid_phase']}")
        print(f"  - Action clipped: {summary['violations_by_type']['action_clipped']}")
        print(f"{'='*50}\n")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: random seed
            options: extra options
            
        Returns:
            (observation, info): initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Re-create CityFlow engine
        if self.eng is not None:
            del self.eng
        self.eng = self._create_engine()
        
        # Reset state variables
        self.current_time = 0.0
        self.phase_elapsed = {inter_id: 0.0 for inter_id in self.intersection_ids}
        self.current_phases = {inter_id: 0 for inter_id in self.intersection_ids}
        self.target_durations = {inter_id: self.duration_options[0] for inter_id in self.intersection_ids}  # target green duration
        self.prev_vehicle_ids = set()
        self.last_num_switches = 0
        self.constraint_violations = {"min_green": 0, "target_duration": 0, "invalid_phase": 0, "action_clipped": 0}
        
        # Reset accumulated violations and log
        self.total_violations = {"min_green": 0, "target_duration": 0, "invalid_phase": 0, "action_clipped": 0}
        self.violation_log = []
        
        # Reset flow stats
        # Per-intersection stats: total passed vehicles, total waiting vehicles
        self.intersection_flow = {inter_id: {
            "total_vehicles_passed": 0,      # total passed vehicles
            "total_waiting_vehicles": 0,     # total waiting vehicles (for averages)
            "total_vehicle_count": 0,        # total vehicle count (for averages)
            "step_count": 0,                 # step counter
        } for inter_id in self.intersection_ids}
        
        # Vehicles on incoming lanes in the previous step (for flow estimation)
        self.prev_lane_vehicles = {}
        
        # Set random seed
        if seed is not None:
            self.eng.set_random_seed(seed)
        
        # Initial phases
        for inter_id in self.intersection_ids:
            # CityFlow starts from phase 0 by default
            self.current_phases[inter_id] = 0
        
        # Initial observation
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Run one environment step.
        
        Args:
            action: action vector (phase/duration per intersection)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        try:
            # ================== Action preprocessing ==================
            # _sanitize_action returns (phases, durations)
            phases, durations = self._sanitize_action(action)
            
            # ================== Apply action (with constraint checks) ==================
            self._apply_action(phases, durations)
            
            # ================== Simulation advance ==================
            # Run multiple micro-steps within the control interval
            for _ in range(int(self.ctrl_interval)):
                self.eng.next_step()
                self.current_time += 1.0
            
            # ================== Update vehicle ID set ==================
            # Get current vehicles for next-step departure computation
            current_vehicle_ids = set(self.eng.get_vehicles(include_waiting=True))
            
            # ================== Update flow statistics ==================
            self._update_intersection_flow()
            
            # ================== Compute reward ==================
            reward = self._get_reward()
            
            # Update prev_vehicle_ids
            self.prev_vehicle_ids = current_vehicle_ids
            
            # ================== Get observation ==================
            obs = self._get_obs()
            
            # ================== Check termination ==================
            terminated, truncated = self._check_done()
            
            # ================== Info ==================
            info = self._get_info()
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            # Fail-safe
            print(f"[ERROR] step() exception: {e}")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = self._get_info()
            info["error"] = str(e)
            return obs, 0.0, True, False, info
    
    def render(self):
        """Render the environment (CityFlow visualizes mainly via replay files)."""
        if self.render_mode == "human":
            print(f"Time: {self.current_time:.1f}s, "
                  f"Vehicles: {self.eng.get_vehicle_count()}, "
                  f"Avg Travel Time: {self.eng.get_average_travel_time():.2f}s")
    
    def close(self):
        """Close the environment and release resources."""
        if self.eng is not None:
            del self.eng
            self.eng = None


# ================== Helper functions ==================

def make_env(config: Dict[str, Any]) -> CityFlowMultiIntersectionEnv:
    """
    Factory function to create the environment.
    
    Args:
        config: environment config dict
        
    Returns:
        CityFlowMultiIntersectionEnv instance
    """
    return CityFlowMultiIntersectionEnv(config)


def get_default_config(config_path: str = "./config.json") -> Dict[str, Any]:
    """
    Get default config.
    
    Args:
        config_path: CityFlow config file path
        
    Returns:
        Default config dict
    """
    return {
        "cityflow_config_file": config_path,
        "thread_num": 1,
        "ctrl_interval": 10,
        "episode_length": 3600,
        "min_green": 10,
        "max_phase_time": 60,
        "max_queue_per_dir": 50,
        "min_duration": 10,   # min green duration (s)
        "max_duration": 60,   # max green duration (s); options: 10,11,12,...,60
        "alpha": 1.0,
        "beta": 5.0,
        "gamma": 0.1,
        "delta": 0.5,
    }


if __name__ == "__main__":
    # Simple smoke test
    import os
    
    # Config path (adjust to your setup)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    config = get_default_config(config_path)
    config["episode_length"] = 100  # short test
    
    env = CityFlowMultiIntersectionEnv(config, render_mode="human")
    
    print("\nEnvironment info:")
    print(f"  - Intersections: {env.num_intersections}")
    print(f"  - Phases: {env.num_phases}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    
    # Random policy smoke test
    obs, info = env.reset(seed=42)
    print(f"\nInitial obs shape: {obs.shape}")
    
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        
        if terminated or truncated:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()

