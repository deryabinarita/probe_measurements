# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 20:51:01 2022

@author: Rita
"""

import numpy as np
import matplotlib.pyplot as plt

def smooth_osc(t_window,dt,s):
    #Moving average
    #SOURCE: https://numba.pydata.org/numba-doc/latest/user/examples.html
    a=s
    window_width=int(np.round(t_window/dt))
    out=np.zeros(s.shape)
    
    # window_width = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count
    
    return out

def extract_signal(path='./20190220/',filename='25ez3.dat'):
    #number of the data column
    prorez=100
    
    #open file
    f=open(path+filename,'rb')
    sig=np.fromfile(f, dtype=np.float32)[::prorez]
    f.close()
    
    #discretization
    dt=0.02*prorez #mks
    
    #time array
    time=np.linspace(0,len(sig)*dt,len(sig)) #mks
    
    #subtraction of the constant component
    num_const_data=500
    const_data=sig[num_const_data]
    sig=sig-const_data
    
    #smoothing with moving average method
    t_window=5*1e3 #mks
    t_window_int=int(t_window/dt/2)
    sig_smooth=smooth_osc(t_window_int,dt,sig)
    
    return time, sig_smooth

def time_average(time,time_int,t_window,arr):
    #function for data averaging near time point
    
    #array number with time
    num_time_int=np.argmin(np.abs((time-time_int)))
    
    #dicretization
    dt=np.diff(time)[0]
    
    #number of time window points
    num_tw=int(t_window/dt/2)
    
    return np.mean(arr[num_time_int-num_tw:num_time_int+num_tw])


def read_shots_group (shots_group, k1k2, path):
    #type B
    shotn=shots_group[0]
    time,i4=extract_signal(path=path,filename=str(shotn).zfill(2)+'ez4.dat')
    time,i2=extract_signal(path=path,filename=str(shotn).zfill(2)+'ez5.dat')

    #type A
    shotn=shots_group[1]
    time,t2a=extract_signal(path=path,filename=str(shotn).zfill(2)+'ez4.dat')

    #type A dop
    shotn=shots_group[2]
    time,t2a_dop=extract_signal(path=path,filename=str(shotn).zfill(2)+'ez4.dat')

    #T2 temperature [eV]
    T2=50/k1k2*(t2a-t2a_dop)/np.log(1+i4/i2)

    #type C
    shotn=shots_group[3]
    time,t4c=extract_signal(path=path,filename=str(shotn).zfill(2)+'ez4.dat')

    #type C dop
    shotn=shots_group[4]
    time,t4c_dop=extract_signal(path=path,filename=str(shotn).zfill(2)+'ez4.dat')

    #T4 temperature [eV]
    T4=50/k1k2*(t4c-t4c_dop)/np.log(1+i2/i4)

    #density type 2 and 4 [cm^(-3)]
    n2=4.26e10*i2*1e3/k1k2/np.sqrt(T2)
    n4=4.26e10*i4*1e3/k1k2/np.sqrt(T4)


    return dict(
        n2=n2,
        n4=n4,
        T2=T2,
        T4=T4,
        time=time
        ) #

path='./20190220/'
#path = './'

#transmission ratio
k1k2=0.5

#[B, A, A_dop, C, C_dop]
shots = [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14],\
         [15, 18, 19, 20, 21],\
         [38, 39, 40, 41, 42], [43, 44, 45, 46, 47],\
         [48, 49, 50, 51, 52], [53, 54, 55, 56, 57],\
         [61, 62, 63, 64, 65]]
    
radius_upper = [82, 80, 78] 
radius_lower = [88, 86, 84, 82, 76]
out=np.loadtxt('radius_upper_probe.txt')
out_r2 = out[:,0] 
out_r3 = out[:,1]
out_r4 = out[:,2]

r2=list()
r4=list()
for r_buf in radius_upper:
    ii=np.argmin( np.abs(r_buf-out_r3) )
    r2.append( out_r2[ii] )
    r4.append( out_r4[ii] )
    
list_=list()

shots_upper = shots[:3]
for shots_group in shots_upper:
    dict_buf = read_shots_group(shots_group, k1k2, path) 
    #словарь с буферными n2, n4, T2, T4, time
    list_.append(dict_buf)
    #лист словарей 

#data averaging near time_int with time windows - t_windows
t_window=10*1e3 #mks
time_int=27*1e3 #mks

list_T2av_upper=list()
for dict1 in list_:
    T2_av_buf=time_average(dict1['time'],time_int,t_window,dict1['T2'])
    list_T2av_upper.append(T2_av_buf)
    




list_T4av_upper=list()
for dict1 in list_:
    T4_av_buf=time_average(dict1['time'],time_int,t_window,dict1['T4'])
    list_T4av_upper.append(T4_av_buf)

list_n2av_upper=list()
for dict1 in list_:
    n2_av_buf=time_average(dict1['time'],time_int,t_window,dict1['n2'])
    list_n2av_upper.append(n2_av_buf)
    
list_n4av_upper=list()
for dict1 in list_:
    n4_av_buf=time_average(dict1['time'],time_int,t_window,dict1['n4'])
    list_n4av_upper.append(n4_av_buf)
    
   
    

#sig
fig, ax = plt.subplots()

ax.plot(r2,list_T2av_upper,'ob')

ax.plot(r4,list_T4av_upper,'or')

ax.set_title('T2_blue, T4_red, upper vs R')
ax.set_ylabel('T2, T4')
ax.set_xlabel('radius, mm')

plt.show()


fig, ax = plt.subplots()

ax.plot(r2,list_n2av_upper,'ob')

ax.plot(r4,list_n4av_upper,'or')

ax.set_title('n2_blue, n4_red, upper vs R')
ax.set_ylabel('n2, n4')
ax.set_xlabel('radius, mm')

plt.show()