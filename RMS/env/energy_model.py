import roboticstoolbox as rtb
from rokae import ROKAE
from spatialmath import SE3
from quintic_polynomial import QuinticPolynomial
import  numpy as np
from math import pi
import matplotlib.pyplot as plt
import time

robot = ROKAE()
 # Stator resistance of the motor
MotorR = np.array([1.3, 1.3, 5.1, 7.5, 20.3, 20.3])

 # Motor torque constant
MotorT = np.array([0.5, 0.5, 0.48, 0.39, 0.32, 0.32])

 # Motor gear reduction ratio
MotorGear = np.array([81, 101, 81, 51, 101, 51])

waypionts_num = 10
 # q_pickup = [pi/2, pi/6, -pi/4,0,-pi/2,pi/2]

def energy_dynamic(q_pickup,load,exe_time):
    '''
    q_pickup: target joint position
    load: payload
    exe_time: execution time
    '''
    q,qd,qdd = QuinticPolynomial(robot.q, q_pickup,exe_time, waypionts_num)

    delta_time = exe_time/waypionts_num
    # External load
    W = [0,0,0,0,0,9.8*load]
    tau = robot.rne(q, qd, qdd,fext=W)
    qd_mat = np.array(qd)
    tau_mat = np.array(tau)
    
    # Power for motion work
    P_d = tau_mat*qd_mat
    # Power consumed by resistance
    P_s = (MotorR*(tau_mat*tau_mat))/((MotorT*MotorT)*(MotorGear*MotorGear))
    E = (P_d+P_s)*delta_time
    # print(E)
    Enegy = np.sum(E)
    return Enegy

def energy_static(q_configure,load,idle_time):
    '''
    q_configure: current joint angle
    load: payload
    idle_time: idle time
    '''
    tau = robot.rne(q_configure, robot.qz, robot.qz,fext=load)
    tau_mat = np.array(tau)
    P_s = (MotorR*(tau_mat*tau_mat))/((MotorT*MotorT)*(MotorGear*MotorGear))
    return np.sum(P_s*idle_time)


if __name__ == '__main__': 
    Energy_list =[]

    q0 = [pi/2, pi/6,-pi/4, 0, -pi/2, pi/2]
    q1 = [pi/2, pi/4,-pi/4, 0, -pi/4, pi/2]
    q2 = [pi/2, pi/6,-pi/4, 0, -pi/2, 0]
    q3 = [pi/2, pi/6,-pi/6, 0, -pi/2, 0]
    q4 = [pi/2, pi/4,-pi/4, 0, -pi/2, pi/2]
    m0 = 0.5
    m1 = 1.2
    m2 = 0.8
    m3 = 0.5
    m4 = 0.4

    m0 = 0.5
    m1 = 0.8
    m2 = 1.1
    m3 = 1.4
    m4 = 1.7



    x1 = np.arange(1,5,0.25)
    x2 = np.arange(5,20,1)
    x = np.append(x1,x2)

    ref_time0 = [4.0, 7.2]
    ref_time1 = [2.00, 14.20]
    ref_time2 = [2.50, 16.50]
    ref_time3 = [2.10, 18.00]
    ref_time4 = [2.40, 16.80]
    # t = ref_time4[0]
    # q = q4
    # RESULT = []
    # result = energy_dynamic(q,m0,t)
    # RESULT.append(result)
    # result = energy_dynamic(q,m1,t)
    # RESULT.append(result)
    # result = energy_dynamic(q,m2,t)
    # RESULT.append(result)
    # result = energy_dynamic(q,m3,t)
    # RESULT.append(result)
    # result = energy_dynamic(q,m4,t)
    # RESULT.append(result)
    # print(max(RESULT))


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

    Energy_list0 =[]
    Energy_list1 =[]
    Energy_list2 =[]
    Energy_list3 =[]
    Energy_list4 =[]
    q = q0
    for i in range(len(x)):
    # time_start = time.time()  # Record start time

        Energy_list0.append(energy_dynamic(q,m0,x[i])) 
        Energy_list1.append(energy_dynamic(q,m1,x[i])) 
        Energy_list2.append(energy_dynamic(q,m2,x[i])) 
        Energy_list3.append(energy_dynamic(q,m3,x[i])) 
        Energy_list4.append(energy_dynamic(q,m4,x[i])) 

    # function()   execute the program
    # time_end = time.time()  # Record end time
    # time_sum = time_end - time_start  # Calculate
    # print(time_sum)
    

    # print(Energy_list0.index(min(Energy_list0))*0.05+1,min(Energy_list0))
    # print(Energy_list1.index(min(Energy_list1))*0.05+1, min(Energy_list1))
    # print(Energy_list2.index(min(Energy_list2))*0.05+1, min(Energy_list2))
    # print(Energy_list3.index(min(Energy_list3))*0.05+1, min(Energy_list3))
    # print(Energy_list4.index(min(Energy_list4))*0.05+1, min(Energy_list4))
   
    ref_time0 = [4.30, 7.2]
    ref_time1 = [5.4, 14.20]
    ref_time2 = [5.4, 16.50]
    ref_time3 = [5.1, 18.00]
    ref_time4 = [4.8, 16.80]

    plt.plot(x,Energy_list0)
    plt.plot(x,Energy_list1)
    plt.plot(x,Energy_list2)
    plt.plot(x,Energy_list3)
    plt.plot(x,Energy_list4)
    data_save0 = np.array(Energy_list0)
    data_save1 = np.array(Energy_list1)
    data_save2 = np.array(Energy_list2)
    data_save3 = np.array(Energy_list3)
    data_save4 = np.array(Energy_list4)
    np.save("Energy_list0.npy",data_save0)
    np.save("Energy_list1.npy",data_save1)
    np.save("Energy_list2.npy",data_save2)
    np.save("Energy_list3.npy",data_save3)
    np.save("Energy_list4.npy",data_save4)
    plt.show()