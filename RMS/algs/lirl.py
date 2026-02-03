import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import math
from scipy.optimize import linear_sum_assignment
import sys
import os
import datetime
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
# Robust import for Env: handle both `env` package and `env/env.py` module
try:
    import importlib
    ENV = importlib.import_module('env')
    if not hasattr(ENV, 'Env'):
        # If the top-level env is a package without Env, try submodule
        ENV = importlib.import_module('env.env')
except Exception:
    from env import env as ENV

# =======================
# HYPERPARAMETERS CONFIG
# =======================
CONFIG = {
    # Learning parameters
    'lr_mu': 0.0005,
    'lr_q': 0.001,
    'gamma': 0.98,
    'batch_size': 128,
    'buffer_limit': 1000000,
    'tau': 0.005,  # for target network soft update
    
    # Environment parameters
    'num_of_jobs':100,
    'num_of_robots': 5,
    'alpha': 0.5,
    'beta': 0.5,
    'num_of_episodes': 1000,
    
    # Network architecture
    'hidden_dim1': 128,
    'hidden_dim2': 64,
    'critic_hidden': 32,
    
    # Training parameters
    'memory_threshold': 500,
    'training_iterations': 20,
    'noise_params': {'theta': 0.1, 'dt': 0.05, 'sigma': 0.1},
    
    # Multi-run training parameters
    'enable_multi_run': True,  # Enable multi-run training by default
    'seeds': [3047,294,714,1092,1386,2856,42,114514,2025,1993],  # Multiple random seeds for training
    'num_runs': 10,  # Number of training runs (usually equals len(seeds))
    
    # Testing parameters
    'max_test_steps': 100,
    
    # Output parameters
    'print_interval': 10,
    'enable_gantt_plots': False,  # Set to True to enable real-time plotting
    'plot_training_curve': True,
    'save_models': True,
}


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=CONFIG['buffer_limit'])

    def put(self, transition):
        self.buffer.append(transition)
        
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
    # Convert to numpy array first, then to tensor for efficiency
        s_lst_= torch.FloatTensor(np.array(s_lst))
        a_lst_= torch.tensor(np.array(a_lst), dtype=torch.float)
        r_lst_= torch.tensor(np.array(r_lst), dtype=torch.float)
        s_prime_lst_ = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_mask_lst_ = torch.tensor(np.array(done_mask_lst), dtype=torch.float)

        return s_lst_,a_lst_,r_lst_,s_prime_lst_,done_mask_lst_
    
    def size(self):
        return len(self.buffer)
    
class MuNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_size, CONFIG['hidden_dim1'])
        self.fc2 = nn.Linear(CONFIG['hidden_dim1'], CONFIG['hidden_dim2'])
        self.fc_mu = nn.Linear(CONFIG['hidden_dim2'], action_size)
        self.outlayer = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.outlayer(self.fc_mu(x)) 
        return mu

class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_size, CONFIG['hidden_dim2'])
        self.fc_a = nn.Linear(action_size, CONFIG['hidden_dim2'])
        self.fc_q = nn.Linear(CONFIG['hidden_dim2'] * 2, CONFIG['critic_hidden'])
        self.fc_out = nn.Linear(CONFIG['critic_hidden'], 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta = CONFIG['noise_params']['theta']
        self.dt = CONFIG['noise_params']['dt']
        self.sigma = CONFIG['noise_params']['sigma']
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def action_choose(env,a):
    a_ = a.detach().numpy()
    job_i = a_[0]
    robot_i = a_[1]
    valid_job =[]
    valid_robot = []
    task_set = env.task_set
    for job_id in range(len(task_set)):
        job = task_set[job_id]
        finished = False
        for op in range(len(task_set[job_id])):
            task = job[op]
            if task.state:
                finished = True
            else:
                finished = False
                break
        if not finished:
            valid_job.append(job_id)

    for robot_id in range(len(env.robot_state)):
        if env.robot_state[robot_id]==1:
            valid_robot.append(robot_id)
   
    # Selection rule: divide [0,1] into n segments, observe where the selection factor is located
    if len(valid_job)>0:
        delta = 1.00/len(valid_job)
        r = math.floor(job_i/delta)
        if job_i ==1:
            job_i = valid_job[-1]
        else:
            job_i = valid_job[r]
    if len(valid_robot)>0:
        delta = 1.00/len(valid_robot)
        r = math.floor(robot_i/delta)
        if robot_i ==1:
            robot_i =valid_robot[-1]
        else:
            robot_i =valid_robot[r] 
    

    return [job_i, robot_i, a_[2]]

def action_projection(env, a):
    """
    env: environment object
    a: action vector output by neural network
    return: [job_id, robot_id, param]
    """
    a_ = a.detach().cpu().numpy()
    
    # Get all available jobs and robots
    valid_jobs = []
    for job_id, job in enumerate(env.task_set):
        finished = False
        for op in range(len(job)):
            task = job[op]
            if task.state:
                finished = True
            else:
                finished = False
                break
        if not finished:
            valid_jobs.append(job_id)
    
    valid_robots = [i for i, state in enumerate(env.robot_state) if state == 1]
    
    # If no available jobs or robots, return default value
    if len(valid_jobs) == 0 or len(valid_robots) == 0:
        return [0, 0, a_[2] if len(a_) > 2 else 0.0]
    
    # Construct cost matrix: use network output to build job-robot assignment cost
    cost_matrix = np.zeros((len(valid_jobs), len(valid_robots)))
    
    # Method 1: use first two elements of action vector as weights
    job_preference = a_[0]  # job preference
    robot_preference = a_[1]  # robot preference
    
    for i, job_id in enumerate(valid_jobs):
        for j, robot_id in enumerate(valid_robots):
            # Construct cost based on environment reward function: reward = alpha*delta_time/C_duration + beta*delta_energy/(E_duration+robot_idle_time_now*5.00)
            # Get basic info of the task
            current_job = env.task_set[job_id]
            current_op_idx = 0
            for op_idx, task in enumerate(current_job):
                if not task.state:
                    current_op_idx = op_idx
                    break
            current_task = current_job[current_op_idx] if current_op_idx < len(current_job) else current_job[0]
            
            # Estimate time cost - based on task processing time and robot status
            try:
                
                # Get task processing time (C_duration)
                C_duration = getattr(current_task, 'duration', 1.0)
                # Get current idle time of robot
                robot_idle_time = max(0, env.current_time - env.robot_timeline[robot_id])
                # Estimate energy consumption (based on task complexity and robot efficiency)
                E_duration = getattr(current_task, 'energy', C_duration * 0.8)
                
                # Time cost component
                time_cost_factor = 1.0 / (C_duration + 1e-6)  # avoid division by zero
                # Energy cost component
                energy_cost_factor = 1.0 / (E_duration + robot_idle_time * 5.0 + 1e-6)
                
                # Combine network preference and environment cost
                preference_cost = abs(job_preference - (job_id / len(env.task_set))) + \
                                abs(robot_preference - (robot_id / len(env.robot_state)))
                
                # Total cost (negative reward) - smaller is better
                env_cost = -(env.alpha * time_cost_factor + env.beta * energy_cost_factor)
                cost_matrix[i, j] = preference_cost + env_cost
                # print("Estimating cost for Job {}, Robot {}".format(job_id, robot_id))  
            except (AttributeError, IndexError):
                # If unable to get environment info, fallback to simple preference cost
                job_cost = abs(job_preference - (job_id / len(env.task_set)))
                robot_cost = abs(robot_preference - (robot_id / len(env.robot_state)))
                cost_matrix[i, j] = job_cost + robot_cost
    
    # Use Hungarian algorithm to solve optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # print("Optimal assignment pairs (row_ind, col_ind):", list(zip(row_ind, col_ind)))
    
    # Select the first assignment result
    if len(row_ind) > 0:
        job_id = valid_jobs[row_ind[0]]
        robot_id = valid_robots[col_ind[0]]
    else:
        job_id = valid_jobs[0]
        robot_id = valid_robots[0]
    
    # Continuous parameter solved by quadratic programming (QP)
    if len(a_) > 2:
    # Extract initial value of continuous parameter
        v = a_[2:]  #
        
    # Construct constraint matrix A and vector b
    # Constraints based on selected job and robot
        A, b = construct_constraints_for_qp(env, job_id, robot_id, v)
        
    # Solve quadratic program: min_x 0.5*||x-v||^2 s.t. Ax <= b
        param = solve_quadratic_program(v, A, b)
    else:
        param = 0.0
    
    # print("Projected Action: Job {}, Robot {}, Param {:.3f}".format(job_id, robot_id, param))
    
    return [job_id, robot_id, param]

def construct_constraints_for_qp(env, job_id, robot_id, v):
    """
    Construct constraints for continuous parameters
    env: environment object
    job_id: selected job ID
    robot_id: selected robot ID
    v: continuous parameter output by network
    return: A, b (constraint Ax <= b)
    """
    # Example constraint: continuous parameter should be in [0, 1] range
    n_params = len(v)
    A = np.vstack([np.eye(n_params), -np.eye(n_params)]).astype(np.float64)  # upper and lower bound constraints
    b = np.hstack([np.ones(n_params), np.zeros(n_params)]).astype(np.float64)  # 0 <= x <= 1
    
    # More constraints based on job and robot can be added here
    # For example: kinematic constraints, collision avoidance, etc.
    
    return A, b

def solve_quadratic_program(v, A, b):
    """
    Solve quadratic programming problem: min_x 0.5*||x-v||^2 s.t. Ax <= b
    v: target value
    A, b: constraint matrix and vector
    return: optimal solution
    """
    try:
        from scipy.optimize import minimize
        
        # Ensure all data types are float64
        v = np.asarray(v, dtype=np.float64)
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        
        # Objective: 0.5*||x-v||^2
        def objective(x):
            x = np.asarray(x, dtype=np.float64)
            return 0.5 * np.sum((x - v)**2)
        
        # Constraint function
        def constraint_fun(x):
            x = np.asarray(x, dtype=np.float64)
            return b - A @ x
            
        constraints = {'type': 'ineq', 'fun': constraint_fun}
        
        # Initial value
        x0 = np.clip(v, 0.0, 1.0).astype(np.float64)
        
        # Solve
        result = minimize(objective, x0, constraints=constraints, method='SLSQP')
        
        if result.success:
            return float(result.x[0]) if len(result.x) > 0 else float(v[0])
        else:
            # If the solver fails, return the value projected to the feasible region
            return float(np.clip(v[0], 0.0, 1.0))
            
    except (ImportError, Exception) as e:
        # If any error occurs, fall back to simple projection
        v_safe = np.asarray(v, dtype=np.float64)
        return float(np.clip(v_safe[0] if len(v_safe) > 0 else 0.0, 0.0, 1.0))



def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s, a, r, s_prime, done_mask = memory.sample(CONFIG['batch_size'])
    
    target = torch.unsqueeze(r, dim=1) + CONFIG['gamma'] * q_target(s_prime, mu_target(s_prime)).mul(done_mask) 
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s, mu(s)).mean()  # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - CONFIG['tau']) + param.data * CONFIG['tau'])
    
def main(config=None):
    """Main training function"""
    if config is None:
        config = CONFIG
    
    # Environment setup
    env = ENV.Env(config['num_of_jobs'], config['num_of_robots'], config['alpha'], config['beta'])
    state_size = len(env.state)
    action_size = len(env.action)
    
    # Network initialization
    q, q_target = QNet(state_size, action_size), QNet(state_size, action_size)
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(state_size, action_size), MuNet(state_size, action_size)
    mu_target.load_state_dict(mu.state_dict())

    # Optimizer setup
    mu_optimizer = optim.Adam(mu.parameters(), lr=config['lr_mu'])
    q_optimizer = optim.Adam(q.parameters(), lr=config['lr_q'])
    
    # Training components
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size))
    memory = ReplayBuffer()
    
    # Training variables
    action_restore = []
    score_record = []
    reward = 0.0

    print(f"Starting DDPG-LIRL training:")
    print(f"Jobs: {config['num_of_jobs']}, Robots: {config['num_of_robots']}")
    print(f"Alpha: {config['alpha']}, Beta: {config['beta']}, Episodes: {config['num_of_episodes']}")

    for n_epi in range(config['num_of_episodes']):
        s = env.reset()
        done = False
        action_eps = []
        step = 0
        
        while not done:
            a = mu(torch.from_numpy(s).float()) 
            a = torch.clamp(a, 0, 1)
            a = a.to(torch.float32)
            action_eps.append(a.detach().numpy())
            action = action_projection(env, a)

            s_prime, r, done = env.step(action)
            
            # Optional: Real-time Gantt chart plotting
            if config['enable_gantt_plots'] and n_epi % 10 == 0:
                try:
                    env.render(f"Training - Episode {n_epi}, Step {len(action_eps)}")
                except:
                    pass
            
            memory.put((s, a.detach().numpy(), r, s_prime, done))
            s = s_prime
            step += 1
            reward+= r
            # Print step
            # print(f"Episode {n_epi}, Step {step}, Action: {action}, Reward: {r:.4f}")
            
        action_restore.append(action_eps)    
        score_record.append(reward)
        reward = 0.0  
        
        # Training update
        # Training update
        if memory.size() > config['memory_threshold']:   
            for i in range(config['training_iterations']):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)
        
        if n_epi % config['print_interval'] == 0 and n_epi != 0:
            print(f"Episode {n_epi}: Average Score = {np.mean(score_record[-config['print_interval']:]):.4f}")
    
    return score_record, action_restore, [mu, mu_target, q, q_target]


def multi_run_training(config=None):
    """Execute multiple training runs with different seeds"""
    if config is None:
        config = CONFIG
    
    all_score_records = []
    all_action_restores = []
    all_models = []
    
    print(f"\n{'='*80}")
    print(f"Starting Multi-Run DDPG-LIRL Training")
    print(f"Seeds: {config['seeds']}")
    print(f"Total runs: {len(config['seeds'])}")
    print(f"{'='*80}")
    
    for run_idx, seed in enumerate(config['seeds']):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{len(config['seeds'])} - Seed: {seed}")
        print(f"{'='*60}")
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Run training
        score_record, action_restore, models = main(config)
        
        # Store results
        all_score_records.append(score_record)
        all_action_restores.append(action_restore)
        all_models.append(models)
        
        print(f"Run {run_idx + 1} completed - Final Score: {score_record[-1]:.4f}")
        
    print(f"\n{'='*60}")
    print(f"All {len(config['seeds'])} runs completed!")
    print(f"{'='*60}")
    
    return all_score_records, all_action_restores, all_models


def evaluate_multi_run_results(all_score_records, config=None):
    """Evaluate and analyze results from multiple runs"""
    if config is None:
        config = CONFIG
    
    print(f"\n{'='*60}")
    print(f"Multi-Run Training Results Analysis")
    print(f"{'='*60}")
    
    # Calculate statistics
    final_scores = [scores[-1] for scores in all_score_records]
    mean_scores = [np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores) for scores in all_score_records]
    
    print(f"Final Episode Scores:")
    for i, (seed, score) in enumerate(zip(config['seeds'], final_scores)):
        print(f"  Run {i+1} (Seed {seed}): {score:.4f}")
    
    print(f"\nLast 20 Episodes Average Scores:")
    for i, (seed, score) in enumerate(zip(config['seeds'], mean_scores)):
        print(f"  Run {i+1} (Seed {seed}): {score:.4f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean Final Score: {np.mean(final_scores):.4f} ± {np.std(final_scores):.4f}")
    print(f"  Best Final Score: {np.max(final_scores):.4f}")
    print(f"  Worst Final Score: {np.min(final_scores):.4f}")
    print(f"  Mean of Last 20 Episodes: {np.mean(mean_scores):.4f} ± {np.std(mean_scores):.4f}")
    
    return {
        'final_scores': final_scores,
        'mean_scores': mean_scores,
        'overall_mean': np.mean(final_scores),
        'overall_std': np.std(final_scores),
        'best_score': np.max(final_scores),
        'worst_score': np.min(final_scores)
    }
def test_and_visualize(config=None, model_path=None):
    """Test trained model and visualize scheduling process"""
    if config is None:
        config = CONFIG
        
    print("\n=== Starting DDPG-LIRL Testing and Visualization ===")
    
    # Create environment
    env = ENV.Env(config['num_of_jobs'], config['num_of_robots'], config['alpha'], config['beta'])
    state_size = len(env.state)
    action_size = len(env.action)
    
    # Load trained model
    mu = MuNet(state_size, action_size)
    if model_path and os.path.exists(model_path):
        try:
            mu.load_state_dict(torch.load(model_path))
            mu.eval()
            print(f"Successfully loaded model: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using randomly initialized model")
    else:
        print("Warning: Model path not provided or file not found, using random initialization")
    
    # Reset environment
    s = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    print(f"\nStarting scheduling - Jobs: {config['num_of_jobs']}, Robots: {config['num_of_robots']}")
    print(f"Alpha: {config['alpha']}, Beta: {config['beta']}")
    print("-" * 50)
    
    # Execute scheduling process
    while not done and step < config['max_test_steps']:
        # Use trained policy network to select action
        with torch.no_grad():
            a = mu(torch.from_numpy(s).float())
            a = torch.clamp(a, 0, 1)
            
        # Use action projection method
        action = action_projection(env, a)
        
        print(f"Step {step+1}: Job{action[0]}, Robot{action[1]}, Param{action[2]:.3f}")
        
        # Execute action
        s_prime, reward, done = env.step(action)
        
        print(f"  Reward: {reward:.4f}")
        total_reward += reward
        s = s_prime
        step += 1
        
        # Real-time Gantt chart update
        if config['enable_gantt_plots']:
            try:
                env.render(f"DDPG-LIRL Scheduling - Step {step} (Jobs:{config['num_of_jobs']}, Robots:{config['num_of_robots']})")
                print(f"  Gantt chart updated (Step {step})")
            except Exception as e:
                print(f"  Error plotting Gantt chart: {e}")
        
        if done:
            print(f"\nAll tasks completed! Total steps: {step}")
            break
    
    # Print final results
    print(f"\n=== Scheduling Results Summary ===")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Average reward: {total_reward/step:.4f}")
    print(f"Makespan: {env.future_time:.2f}")
    print(f"Current time: {env.current_time:.2f}")
    print(f"Robot timeline: {[f'{t:.2f}' for t in env.robot_timeline]}")
    
    # Final Gantt chart
    # print(f"\n=== Drawing Final Gantt Chart ===")
    # try:
    #     env.render(f"DDPG-LIRL Final Results (Jobs:{config['num_of_jobs']}, Robots:{config['num_of_robots']})")
    #     print("Final Gantt chart generated successfully!")
    # except Exception as e:
    #     print(f"Error generating final Gantt chart: {e}")
    
    return total_reward, step

def save_results(score_records, action_restores, models_restore, config):
    """Save training results and models"""
    if not config['save_models']:
        return None, None
        
    # Create save directory with timestamp
    import datetime
    alg_name = "ddpg_lirl_pi"
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{alg_name}_{now_str}"
    os.makedirs(save_dir, exist_ok=True)

    # Save training data
    np.save(os.path.join(save_dir, f"{alg_name}_scores_{now_str}.npy"), score_records)
    np.save(os.path.join(save_dir, f"{alg_name}_actions_{now_str}.npy"), action_restores)

    # Save models
    model_paths = []
    for idx, models in enumerate(models_restore):
        mu, mu_target, q, q_target = models
        mu_path = os.path.join(save_dir, f"{alg_name}_mu_{idx}_{now_str}.pth")
        torch.save(mu.state_dict(), mu_path)
        torch.save(mu_target.state_dict(), os.path.join(save_dir, f"{alg_name}_mu_target_{idx}_{now_str}.pth"))
        torch.save(q.state_dict(), os.path.join(save_dir, f"{alg_name}_q_{idx}_{now_str}.pth"))
        torch.save(q_target.state_dict(), os.path.join(save_dir, f"{alg_name}_q_target_{idx}_{now_str}.pth"))
        model_paths.append(mu_path)
    
    print(f"Results saved to directory: {save_dir}")
    return save_dir, model_paths[0] if model_paths else None

def plot_training_curve(score_records):
    """Plot training curve"""
    x = range(len(score_records))
    y = score_records
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title('DDPG-LIRL Training Score over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

def save_multi_run_results(all_score_records, all_action_restores, all_models, config):
    """Save results from multiple training runs"""
    if not config['save_models']:
        return None, None
        
    # Create save directory with timestamp
    alg_name = "ddpg_lirl_pi"
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{alg_name}_multi_run_{now_str}"
    os.makedirs(save_dir, exist_ok=True)

    # Save training data for all runs
    np.save(os.path.join(save_dir, f"{alg_name}_all_scores_{now_str}.npy"), all_score_records)
    np.save(os.path.join(save_dir, f"{alg_name}_all_actions_{now_str}.npy"), all_action_restores)

    # Save models from all runs
    model_paths = []
    for run_idx, models in enumerate(all_models):
        mu, mu_target, q, q_target = models
        run_save_dir = os.path.join(save_dir, f"run_{run_idx+1}_seed_{config['seeds'][run_idx]}")
        os.makedirs(run_save_dir, exist_ok=True)
        
        mu_path = os.path.join(run_save_dir, f"{alg_name}_mu_{now_str}.pth")
        torch.save(mu.state_dict(), mu_path)
        torch.save(mu_target.state_dict(), os.path.join(run_save_dir, f"{alg_name}_mu_target_{now_str}.pth"))
        torch.save(q.state_dict(), os.path.join(run_save_dir, f"{alg_name}_q_{now_str}.pth"))
        torch.save(q_target.state_dict(), os.path.join(run_save_dir, f"{alg_name}_q_target_{now_str}.pth"))
        model_paths.append(mu_path)
    
    # Save configuration
    config_path = os.path.join(save_dir, f"config_{now_str}.json")
    import json
    with open(config_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_config = {}
        for key, value in config.items():
            if isinstance(value, np.ndarray):
                json_config[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.integer):
                json_config[key] = [int(v) for v in value]
            else:
                json_config[key] = value
        json.dump(json_config, f, indent=2)
    
    print(f"Multi-run results saved to directory: {save_dir}")
    return save_dir, model_paths

def plot_multi_run_training_curves(all_score_records, config=None):
    """Plot training curves for multiple runs"""
    if config is None:
        config = CONFIG
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    for i, scores in enumerate(all_score_records):
        x = range(len(scores))
        plt.plot(x, scores, alpha=0.6, label=f'Run {i+1} (Seed {config["seeds"][i]})')
    
    # Plot mean curve
    min_length = min(len(scores) for scores in all_score_records)
    mean_scores = np.mean([scores[:min_length] for scores in all_score_records], axis=0)
    std_scores = np.std([scores[:min_length] for scores in all_score_records], axis=0)
    
    x = range(min_length)
    plt.plot(x, mean_scores, 'k-', linewidth=2, label='Mean')
    plt.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, color='black')
    
    plt.title('DDPG-LIRL Multi-Run Training Curves')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parse command line arguments or use default configuration
    import argparse
    
    parser = argparse.ArgumentParser(description='DDPG-LIRL Policy for Robot Task Scheduling')
    parser.add_argument('--jobs', type=int, default=CONFIG['num_of_jobs'], help='Number of jobs')
    parser.add_argument('--robots', type=int, default=CONFIG['num_of_robots'], help='Number of robots')
    parser.add_argument('--alpha', type=float, default=CONFIG['alpha'], help='Alpha parameter')
    parser.add_argument('--beta', type=float, default=CONFIG['beta'], help='Beta parameter')
    parser.add_argument('--episodes', type=int, default=CONFIG['num_of_episodes'], help='Number of episodes')
    parser.add_argument('--test-only', action='store_true', help='Run test only (skip training)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to saved model for testing')
    parser.add_argument('--multi-run', action='store_true', default=CONFIG['enable_multi_run'], 
                       help='Run multiple training sessions with different seeds')
    parser.add_argument('--single-run', action='store_true', help='Force single run training (override config)')
    parser.add_argument('--seeds', nargs='+', type=int, default=CONFIG['seeds'], help='Random seeds for multi-run training')
    
    args = parser.parse_args()
    
    # Update CONFIG with command line arguments
    config = CONFIG.copy()
    config.update({
        'num_of_jobs': args.jobs,
        'num_of_robots': args.robots,
        'alpha': args.alpha,
        'beta': args.beta,
        'num_of_episodes': args.episodes,
        'seeds': args.seeds,
        'enable_multi_run': args.multi_run and not args.single_run  # Allow override with --single-run
    })
    
    print(f"\n{'='*60}")
    print(f"DDPG-LIRL Policy for Robot Task Scheduling")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Jobs: {config['num_of_jobs']}")
    print(f"  Robots: {config['num_of_robots']}")
    print(f"  Alpha: {config['alpha']}")
    print(f"  Beta: {config['beta']}")
    print(f"  Episodes: {config['num_of_episodes']}")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Test only: {args.test_only}")
    print(f"  Multi-run enabled (config): {CONFIG['enable_multi_run']}")
    print(f"  Multi-run mode (final): {config['enable_multi_run']}")
    if args.model_path:
        print(f"  Model path: {args.model_path}")
    print(f"{'='*60}")
    
    if args.test_only:
        # Test only mode
        test_and_visualize(config, args.model_path)
    elif config['enable_multi_run']:
        # Multi-run training mode (based on config or command line)
        all_score_records, all_action_restores, all_models = multi_run_training(config)
        
        # Evaluate results
        stats = evaluate_multi_run_results(all_score_records, config)
        
        # Save multi-run results if enabled
        if config['save_models']:
            save_dir, model_paths = save_multi_run_results(all_score_records, all_action_restores, all_models, config)
        
        # Plot multi-run training curves
        if config['plot_training_curve']:
            plot_multi_run_training_curves(all_score_records, config)
        
        # Test with the best performing model
        if config['save_models'] and model_paths:
            best_run_idx = np.argmax([scores[-1] for scores in all_score_records])
            best_model_path = model_paths[best_run_idx]
            print(f"\n{'='*40}")
            print(f"Testing with best model (Run {best_run_idx+1})...")
            print(f"{'='*40}")
            test_and_visualize(config, best_model_path)
    else:
        # Single run training mode
        # Set random seed
        if config['seeds']:
            seed = config['seeds'][0]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            print(f"Using random seed: {seed}")
        
        score_record, action_restore, models = main(config)
        
        # Save results if enabled
        save_dir, model_path = save_results([score_record], [action_restore], [models], config)
        
        # Plot training curve
        # if config['plot_training_curve']:
        #     plot_training_curve(score_record)
        
        # Test with trained model
        # if model_path:
        #     print(f"\n{'='*40}")
        #     print(f"Testing with trained model...")
        #     print(f"{'='*40}")
        #     test_and_visualize(config, model_path)