import numpy as np

def QuinticPolynomial(q0,q1,tf,waypionts_num =10):
    n = len(q0)
    YY = []
    YYD = []
    YYDD = []
    g_vs = [0 for i in range(n)]
    g_as =[0 for i in range(n)]
    g_ve = [0 for i in range(n)]
    g_ae =[0 for i in range(n)]
    for i in range(n):
        S = q1[i] - q0[i]
        a0 = 0
        a1 = g_vs[i]
        a2 = g_as[i]
        a3 = (20*S - (8*g_ve[i]+12*g_vs[i])*tf -(3*g_as[i]- g_ae[i])*(tf**2))/(2*(tf**3))
        a4 = (-30*S + (14*g_ve[i]+16*g_vs[i])*tf +(3*g_as[i] - 2*g_ae[i])*(tf**2))/(2*(tf**4))
        a5 = (12*S - 6*(g_ve[i]+g_vs[i])*tf +(g_ae[i] - g_as[i])*(tf**2))/(2*(tf**5))
        t = np.arange(0,tf,tf/waypionts_num)
        Y = a0 + a1*t + a2*(t**2) + a3*(t**3) + a4*(t**4) + a5*(t**5)
        YD = a1 + 2*a2*t + 3*a3*(t**2)+ 4*a4*(t**3)+ 5*a5*(t**4)
        YDD =  2*a2 + 6*a3*t+ 12*a4*(t**2)+ 20*a5*(t**3)
        YY.append(Y)
        YYD.append(YD)
        YYDD.append(YDD)
    return np.array(YY).T,np.array(YYD).T, np.array(YYDD).T 