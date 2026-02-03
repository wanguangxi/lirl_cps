#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :task_set.py
@Description :
@Date        :2021/12/27 14:36:57
@Author      :wgx
@Version     :1.0
'''
import numpy as np
from math import pi
import copy


class Task:
    def __init__(self,type,available_modules,target_position, mass, ref_time,ref_energy,start_time,end_time, id):
        """[summary]

        Args:
            type (string): the type of task, t0 t1 t2 t3 t4 
            available_modules(array[1,2]): the avaible plant modules that can process task
                t0.avaible=[0], t1-t4.avaible=[1,2]
                0-- polishing plant module
                1,2 -- assembly plant module
            target_position (array[q0,q1,...,q5]): robot joints
            mass (float): the mass of load, e.g., 0.5kg
            ref_time ([0.22, 0.8]): reference procssing time
            ref_energy: reference energy consuming
        """        
        self.type = type
        self.available_modules = available_modules
        self.target_position  = target_position
        self.mass = mass
        self.ref_time = ref_time
        self.ref_energy = ref_energy
        self.start_time = start_time
        self.end_time = end_time
        self.processing_robot = 0
        self.processing_time = 0
        self.state = False
        self.id = id
        self.theta = 0
        self.energy_consuming = 0
    def get_state(self):
        return  self.state
    def set_state(self, state):
        """[summary]

        Args:
            state (bool): False- task unprocessed; True - task assigned
        """        
        self.state = state
    def set_theta(self, theta):
        self.theta = theta
    def set_energy_consuming(self, energy):
        self.energy_consuming = energy
 
class TaskSet:
    def __init__(self, product_num):
        """[summary]

        Args:
            product_num (int): one product has 5 operations, one operation is one task 
        """        
        self.task_num = product_num*5
        self.product_num = product_num
        self.task_set = self.creat_task_set()

    def creat_task_set(self):
        """
            t0   t1  t2  t3  t4  
        p0 |   |   |   |   |   |   
        p1 |   |   |   |   |   |   
        ...
        """    
        q0 = [pi/2, pi/6,-pi/4, 0, -pi/2, pi/2]
        q1 = [pi/2, pi/4,-pi/4, 0, -pi/4, pi/2]
        q2 = [pi/2, pi/6,-pi/4, 0, -pi/2, 0]
        q3 = [pi/2, pi/6,-pi/6, 0, -pi/2, 0]
        q4 = [pi/2, pi/4,-pi/4, 0, -pi/2, pi/2]

        am0 = list(range(501))
        am1 = list(range(501))
        am2 = list(range(501))
        am3 = list(range(501))
        am4 = list(range(501))

        m0 = 0.5
        m1 = 1.2
        m2 = 0.8
        m3 = 0.5
        m4 = 0.3





        # ref_time0 = [4.0, 7.2]
        # ref_time1 = [2.00, 14.20]
        # ref_time2 = [2.50, 16.50]
        # ref_time3 = [2.10, 18.00]
        # ref_time4 = [2.40, 16.80]

        # ref_energy0 = [94.75, 103.54]
        # ref_energy1 = [95.45, 135.31]
        # ref_energy2 = [53.01, 92.83]
        # ref_energy3 = [42.75, 88.10]
        # ref_energy4 = [105.22,161.68]

        # ref_time0 = [4.0, 4.0]
        # ref_time1 = [2.00, 2.00]
        # ref_time2 = [2.50, 2.50]
        # ref_time3 = [2.10, 2.10]
        # ref_time4 = [2.40, 2.40]

        ref_energy0 = [94.75, 103.54]
        ref_energy1 = [95.45, 135.31]
        ref_energy2 = [53.01, 92.83]
        ref_energy3 = [42.75, 88.10]
        ref_energy4 = [105.22,135.386]

        ref_time0 = [4.30, 4.30]
        ref_time1 = [5.40, 5.40]
        ref_time2 = [5.42, 5.42]
        ref_time3 = [5.10, 5.10]
        ref_time4 = [4.85, 4.85]

        # task_set = [[copy.deepcopy(t0),copy.deepcopy(t1),copy.deepcopy(t2),copy.deepcopy(t3),copy.deepcopy(t4) ]]*self.product_num
        task_set =[[[] for i in range(5)] for j in range(self.product_num)]
        # coding with id
        cnt = 0
        start_time = 0.0
        end_time = 0.0
        for i in range(self.product_num):
            task_set[i][0] = Task("t0",am0,q0,m0,ref_time0,ref_energy0,start_time,end_time, cnt)
            task_set[i][1] = Task("t1",am1,q1,m1,ref_time1,ref_energy1,start_time,end_time,cnt+1)
            task_set[i][2] = Task("t2",am2,q2,m2,ref_time2,ref_energy2,start_time,end_time,cnt+2)
            task_set[i][3] = Task("t3",am3,q3,m3,ref_time3,ref_energy3,start_time,end_time,cnt+3)
            task_set[i][4] = Task("t4",am4,q4,m4,ref_time4,ref_energy4,start_time,end_time,cnt+4)
            cnt = cnt+5
        # print("Task set initial success!")
        return task_set

    def task_set_update(self,observation):
        pass




if __name__ == '__main__': 

    task_set = TaskSet(10)

    for i in range(10):
        print("*"*20, i)
        for j in range(5):
            task_type = task_set.task_set[i][j].id
            print(task_type)
    

    # print(task_set.task_set[1])
   
    # print(task_set.task_set[2][1])    
    