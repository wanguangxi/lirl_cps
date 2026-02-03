#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :environment.py
@Description :
@Date        :2023/08/23 20:27:01
@Author      :wgx
@Version     :1.0
'''
import random
import time
import energy_model as EM
import task_set as TS
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



 # Fully flexible manufacturing system
class Env:
    def __init__(self, num_of_jobs, num_of_robots,alpha,beta):
       
        self.num_of_jobs = num_of_jobs
        self.num_of_robots = num_of_robots
        self.task_set = TS.TaskSet(num_of_jobs).task_set
        self.task_state = np.zeros((5*num_of_jobs,), dtype = int) # 0-unfinished/unassigned, 1-assigned
        self.task_prcoessing_time_state = np.zeros((5*num_of_jobs,), dtype = float) # 0-unfinished/unassigned, 1-assigned

        self.robot_state = np.ones((num_of_robots,), dtype = int)   # 0-unavailable, 1-available
        self.last_action_state = np.ones((2,), dtype = int) # 11 means normal
        self.state = np.concatenate((self.task_state,self.robot_state)) # State of all operations
        self.state = np.concatenate((self.state,self.task_prcoessing_time_state))
        self.state = np.concatenate((self.state,self.last_action_state))
        self.action = np.array([0,0,0.0]) # Three dimensions: operation, robot, and Parameter
        self.robot_timeline = np.zeros((num_of_robots,), dtype = float)
        self.current_time = 0.0
        self.future_time = 0.0
        self.done = False
        self.reward = 0.0
       
       
        self.alpha = alpha
        self.beta = beta

        self.step_count = 0
        
        # Track each robot's task execution history for idle time calculation
        self.robot_task_history = [[] for _ in range(num_of_robots)]  # Task history for each robot

    def reset(self):
        self.task_set = TS.TaskSet(self.num_of_jobs).task_set
        self.task_state = np.zeros((5*self.num_of_jobs,), dtype = int) # 0-unfinished/unassigned, 1-assigned
        self.robot_state = np.ones((self.num_of_robots,), dtype = int)   # 0-unavailable, 1-available
        self.task_prcoessing_time_state = np.zeros((5*self.num_of_jobs,), dtype = float) # 0-unfinished/unassigned, 1-assigned
        self.last_action_state = np.ones((2,), dtype = int) # 11 means normal
        self.state = np.concatenate((self.task_state,self.robot_state)) # State of all operations
        self.state = np.concatenate((self.state,self.task_prcoessing_time_state))        
        self.state = np.concatenate((self.state,self.last_action_state))

        self.action = np.array([0,0,0.0])  # Three dimensions: job, robot, and parameter
        self.done = False
        self.robot_timeline = np.zeros((self.num_of_robots,), dtype = float)
        self.current_time = 0.0
        self.future_time = 0.0
        self.reward = 0.0
        self.step_count = 0
      
        # Reset robot task history
        self.robot_task_history = [[] for _ in range(self.num_of_robots)]
        return self.state

    def step(self,action):

        #
        robot_idle_time_sum = self.calculate_robot_idle_times() 
        robot_idle_time_last = robot_idle_time_sum['summary']['total_idle_time']

        for i in range (len(self.task_state)):
            if self.task_state[i] == 1:
                self.done = True
            else:
                self.done = False
                break
        if self.done:
            self.reward = len(self.task_state)
            return self.state,self.reward,self.done  
      
        # Determine selected operation
        job_id = round(action[0])  # job id
        if job_id >=self.num_of_jobs:
            job_id = self.num_of_jobs -1
        
        operations = self.task_set[job_id]   
        assigned_num = 0              
        for iter in range(len(operations)):
            task = operations[iter]
            if task.state:
                assigned_num = assigned_num+1
            else:
                break                  
                
        robot_id = round(action[1])      # int
        if robot_id>=len(self.robot_state)-1:
            robot_id = len(self.robot_state)-1
        param = action[2]         # float[0,1]
        # print("param:",param)

        # Check action validity
    # 1-selected a completed job; 2-correct operation but wrong robot, robot unavailable -10; 3-selected task, robot does not support processing
        if assigned_num == len(operations) and self.robot_state[robot_id] == 0:                    # 1 - selected an already completed job
            self.reward = -1
            self.state[-2] = 0
            self.state[-1] = 0
            print("Selected a completed job and robot is unavailable")
            return self.state,self.reward,self.done
        
        if assigned_num == len(operations) and self.robot_state[robot_id] == 1:                    # 1 - selected an already completed job
            self.reward = -1
            self.state[-2] = 0
            self.state[-1] = 1
            print("Selected a completed job")
            return self.state,self.reward,self.done        

        if self.robot_state[robot_id] == 0:                    # 2 - correct operation but wrong robot / robot unavailable
            self.reward = -1
            self.state[-1]  = 0
            self.state[-2]  = 1
            print("Selected robot is unavailable")
            return self.state,self.reward,self.done  
        
        if not robot_id in task.available_modules:
            self.reward = -1
            self.state[-2] = 0
            self.state[-1] = 0
            print("Selected robot does not support this task processing")
            return self.state,self.reward,self.done   


        task.processing_robot = robot_id
        # Action is valid -> compute reward
        self.last_action_state[0] = 1
        self.last_action_state[1] = 1
 
    # Calculate processing time
        C_duration = task.ref_time[1]
        E_duration = task.ref_energy[1]
        task.processing_time = task.ref_time[0]+(task.ref_time[1]-task.ref_time[0])*param  # task processing time (LIRL)
        # task.processing_time = task.ref_time[0]+ param  # task processing time (alt)
    # Change in timeline as time reward
        if assigned_num == 0: # First operation
            task.start_time = max(self.robot_timeline[robot_id],self.current_time)                       # Task start time
        else: # Not first operation
            pre_task = operations[assigned_num-1]
            task.start_time = max(self.current_time,pre_task.end_time,self.robot_timeline[robot_id]) 

        task.state = True   

    # Update robot timeline
        self.robot_timeline[robot_id] = task.start_time+task.processing_time
        task.end_time = task.start_time+task.processing_time
        
    # Record task history to robot's task history
        task_record = {
            'job_id': job_id,
            'operation_id': assigned_num,
            'start_time': task.start_time,
            'end_time': task.end_time,
            'processing_time': task.processing_time
        }
        self.robot_task_history[robot_id].append(task_record)
        
    # Update future maximum completion time
        future_time_new = np.max(self.robot_timeline)
        delta_time = future_time_new-self.future_time
        self.future_time = future_time_new
    # Update current timeline: = timeline of the earliest available robot
        self.current_time = np.min(self.robot_timeline)

        robot_idle_time_sum = self.calculate_robot_idle_times() 
        robot_idle_time_now = robot_idle_time_sum['summary']['total_idle_time']

    # Calculate newly added energy consumption
        delta_energy = EM.energy_dynamic(task.target_position,task.mass,task.processing_time)
       
        delta_energy = delta_energy + (robot_idle_time_now - robot_idle_time_last)*5.00  # Consider energy consumption during idle time, assume idle energy is 5 units per time unit

    # Calculate reward
        self.reward = self.alpha*delta_time/C_duration + self.beta*delta_energy/(E_duration+robot_idle_time_now*5.00)
        self.reward = -1*self.reward


    # State update
        self.task_state[job_id*5+assigned_num] = 1
        self.task_prcoessing_time_state[job_id*5+assigned_num] = task.processing_time
        self.robot_state[robot_id] = 0
        for robot in range(len(self.robot_state)):
            if self.robot_timeline[robot]<= self.current_time:
                self.robot_state[robot] =1
        self.state = np.concatenate((self.task_state,self.robot_state))
        self.state = np.concatenate((self.state,self.task_prcoessing_time_state))
        self.state = np.concatenate((self.state,self.last_action_state))

        for i in range (len(self.task_state)):
            if self.task_state[i] == 1:
                self.done = True
            else:
                self.done = False
                break       
        return self.state, self.reward,self.done

    def calculate_robot_idle_times(self, reference_time=None):
        """
        Calculate idle time for all robots before the current timeline
        Args:
            reference_time (float): reference time point, default is current time
        Returns:
            dict: dictionary containing idle time statistics for each robot
        """
        if reference_time is None:
            reference_time = self.current_time
            
        idle_stats = {}
        total_idle_time = 0.0
        
        for robot_id in range(self.num_of_robots):
            # Get the robot's task history and sort by start time
            tasks = sorted(self.robot_task_history[robot_id], key=lambda x: x['start_time'])
            
            robot_idle_time = 0.0
            robot_idle_periods = []  # Record idle time periods
            
            if not tasks:
                # If the robot has not performed any tasks, the entire time is idle
                robot_idle_time = reference_time
                if reference_time > 0:
                    robot_idle_periods.append({'start': 0.0, 'end': reference_time, 'duration': reference_time})
            else:
                # Calculate idle time before the first task
                first_task_start = tasks[0]['start_time']
                if first_task_start > 0:
                    idle_duration = first_task_start
                    robot_idle_time += idle_duration
                    robot_idle_periods.append({
                        'start': 0.0, 
                        'end': first_task_start, 
                        'duration': idle_duration
                    })
                
                # Calculate idle time between tasks
                for i in range(len(tasks) - 1):
                    current_task_end = tasks[i]['end_time']
                    next_task_start = tasks[i + 1]['start_time']
                    
                    if next_task_start > current_task_end:
                        idle_duration = next_task_start - current_task_end
                        robot_idle_time += idle_duration
                        robot_idle_periods.append({
                            'start': current_task_end,
                            'end': next_task_start,
                            'duration': idle_duration
                        })
                
                # Calculate idle time from end of last task to reference time
                last_task_end = tasks[-1]['end_time']
                if reference_time > last_task_end:
                    idle_duration = reference_time - last_task_end
                    robot_idle_time += idle_duration
                    robot_idle_periods.append({
                        'start': last_task_end,
                        'end': reference_time,
                        'duration': idle_duration
                    })
            
            # Calculate work time
            work_time = sum([task['processing_time'] for task in tasks])
            
            # Calculate utilization
            utilization = (work_time / reference_time * 100) if reference_time > 0 else 0.0
            
            idle_stats[f'robot_{robot_id}'] = {
                'idle_time': robot_idle_time,
                'work_time': work_time,
                'total_time': reference_time,
                'utilization_rate': utilization,
                'idle_periods': robot_idle_periods,
                'task_count': len(tasks)
            }
            
            total_idle_time += robot_idle_time
        
    # Add overall statistics
        idle_stats['summary'] = {
            'total_idle_time': total_idle_time,
            'average_idle_time': total_idle_time / self.num_of_robots,
            'reference_time': reference_time,
            'total_robots': self.num_of_robots
        }
        
        return idle_stats
    
    def print_idle_time_report(self, reference_time=None):
        """
        Print idle time report
        Args:
            reference_time (float): reference time point, default is current time
        """
        idle_stats = self.calculate_robot_idle_times(reference_time)
        
        print("\n" + "="*60)
        print("           Robot Idle Time Statistics Report")
        print("="*60)
        print(f"Reference time point: {idle_stats['summary']['reference_time']:.2f}")
        print(f"Current time: {self.current_time:.2f}")
        print(f"Future completion time: {self.future_time:.2f}")
        print("-"*60)
            
        for robot_id in range(self.num_of_robots):
            robot_key = f'robot_{robot_id}'
            stats = idle_stats[robot_key]
            
            print(f"Robot {robot_id}:")
            print(f"  Idle time: {stats['idle_time']:.2f}")
            print(f"  Work time: {stats['work_time']:.2f}")
            print(f"  Utilization: {stats['utilization_rate']:.1f}%")
            print(f"  Number of tasks executed: {stats['task_count']}")
            
            if stats['idle_periods']:
                print(f"  Idle periods:")
                for i, period in enumerate(stats['idle_periods']):
                    print(f"    [{i+1}] {period['start']:.2f} - {period['end']:.2f} (Duration: {period['duration']:.2f})")
            print()
            
        print("-"*60)
        summary = idle_stats['summary']
        print(f"Total idle time: {summary['total_idle_time']:.2f}")
        print(f"Average idle time: {summary['average_idle_time']:.2f}")
        print(f"System total utilization: {((summary['reference_time'] * self.num_of_robots - summary['total_idle_time']) / (summary['reference_time'] * self.num_of_robots) * 100) if summary['reference_time'] > 0 else 0:.1f}%")
        print("="*60)

    def render(self, title="Gantt Chart"):
        """
        Draw Gantt chart to show scheduling results
        Args:
            title (str): chart title
        """
    # Set Chinese font and chart style
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # fallback fonts
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.size"] = 10
        
    # Collect all scheduled task information
        scheduled_tasks = []
        job_colors = {}
        
    # Assign different colors for each job
        colors = list(mcolors.TABLEAU_COLORS.values())
        if len(colors) < self.num_of_jobs:
            colors = colors * ((self.num_of_jobs // len(colors)) + 1)
        
        for job_id in range(self.num_of_jobs):
            job_colors[job_id] = colors[job_id]
            
    # Traverse all tasks, collect completed task information
        for job_id in range(self.num_of_jobs):
            operations = self.task_set[job_id]
            for op_id, task in enumerate(operations):
                if task.state:  # If task has been assigned
                    scheduled_tasks.append({
                        'job_id': job_id,
                        'operation_id': op_id,
                        'robot_id': task.processing_robot,
                        'start_time': task.start_time,
                        'end_time': task.end_time,
                        'processing_time': task.processing_time,
                        'color': job_colors[job_id]
                    })
        
        if not scheduled_tasks:
            print("No scheduled tasks to display")
            return
            
    # Sort by robot ID and start time
        scheduled_tasks.sort(key=lambda x: (x['robot_id'], x['start_time']))
        
    # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
    # Draw Gantt chart
        for task in scheduled_tasks:
            robot_id = task['robot_id']
            start_time = task['start_time']
            duration = task['processing_time']
            color = task['color']
            
            # Draw task bar
            ax.barh(robot_id, duration, left=start_time, height=0.6, 
                   color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add task label
            label = f"J{task['job_id']}.{task['operation_id']}"
            ax.text(start_time + duration/2, robot_id, label, 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
    # Set y-axis labels (robots)
        robot_labels = [f"Robot{i}" for i in range(self.num_of_robots)]
        ax.set_yticks(range(self.num_of_robots))
        ax.set_yticklabels(robot_labels)
        
    # Set chart properties
        ax.set_xlabel('Time')
        ax.set_ylabel('Robots')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
    # Add completion time reference line
        makespan = max([task['end_time'] for task in scheduled_tasks])
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(makespan, self.num_of_robots-0.5, f'Makespan: {makespan:.2f}', 
               ha='left', va='top', color='red', fontweight='bold')
        
    # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=job_colors[i], 
                          label=f'Job {i}') for i in range(self.num_of_jobs)]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.show()
        
        # Print scheduling statistics
        print(f"\n=== Scheduling Statistics ===")
        print(f"Total completion time (Makespan): {makespan:.2f}")
        print(f"Number of scheduled tasks: {len(scheduled_tasks)}")
        print(f"Robot utilization:")
        for robot_id in range(self.num_of_robots):
            robot_tasks = [t for t in scheduled_tasks if t['robot_id'] == robot_id]
            if robot_tasks:
                total_work_time = sum([t['processing_time'] for t in robot_tasks])
                utilization = (total_work_time / makespan) * 100
                print(f"  Robot{robot_id}: {utilization:.1f}% ({total_work_time:.2f}/{makespan:.2f})")
            else:
                print(f"  Robot{robot_id}: 0.0% (unused)")
        
        # Print detailed idle time report
        self.print_idle_time_report(makespan)


if __name__ == '__main__': 

    num_of_jobs = 5
    num_of_robots = 3
    alpha = 0.2
    beta = 0.8
    env = Env(num_of_jobs, num_of_robots,alpha,beta)
    Operation_action = 1

    # Random policy test
    import random
    from random import randint
    state = env.reset()
    step = 0
    max_steps = 1000  # Prevent infinite loop
    
    while not env.done and step < max_steps:
    # Randomly select job
        Operation_action = randint(0, num_of_jobs-1)
        Robot_action = randint(0, num_of_robots-1)
        Param_action = random.uniform(0, 1)
        action = np.array([Operation_action, Robot_action, Param_action])

        print("step:", step)
        print("action:", action)

        state_, reward, done = env.step(action)
        print("*"*20)
        print("state:", state_)
        print("reward:", reward)
        print("done:", done)
        print()
        
        step += 1
        state = state_
        
    # If all tasks are completed, draw Gantt chart
        if done:
            print(f"All tasks completed at step {step}!")
            env.render("Random policy scheduling result")
            break
    
    if step >= max_steps:
        print("Reached maximum step limit, some tasks may not be completed")
        env.render("Partial scheduling result")