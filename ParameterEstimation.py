#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:34:37 2017

@author: matgilson
"""

import os
import numpy as np
import scipy.signal as spsg
import WBLECmodel


#%% create a folder for saving the data and model parameters

res_dir = 'model_param/'
if not os.path.exists(res_dir):
    print('create directory:',res_dir)
    os.makedirs(res_dir)


#%% general properties
    
# number of subjects
n_sub = 22
# fMRI sessions: first 2 rest + last 3 movie
n_run = 5
# number of ROIs (brain regions)
N = 66
# number of time points (in TR) of the fMRI/BOLD 
T = 300


# time shifts for spatiotemporal covariances in our FC: 0, 1 and 2 TR
v_tau = np.arange(3,dtype=float)
n_tau = v_tau.size


#%% functional data

# load fMRI data
ts_emp = np.load('ts_emp.npy')
print(ts_emp.shape)

# filtering between 0.01 and 0.1 Hz
n_order = 3
Nyquist_freq = 0.5 / 2
low_f = 0.01 / Nyquist_freq
high_f = 0.1 / Nyquist_freq
b,a = spsg.iirfilter(n_order,[low_f,high_f],btype='bandpass',ftype='butter')

# calculate FC
FC_emp = np.zeros([n_sub,n_run,n_tau,N,N])
for i_sub in range(n_sub):
    for i_run in range(n_run):
        # appply filter
        ts_emp[i_sub,i_run,:,:] = spsg.filtfilt(b,a,ts_emp[i_sub,i_run,:,:],axis=1)
        # center the time series
        ts_emp[i_sub,i_run,:,:] -= np.outer(ts_emp[i_sub,i_run,:,:].mean(1),np.ones(T))
for i_sub in range(n_sub):
    for i_run in range(n_run):
        # calculate BOLD covariances for all time shifts
        for i_tau in range(n_tau):
            FC_emp[i_sub,i_run,i_tau,:,:] = np.tensordot(ts_emp[i_sub,i_run,:,0:T-n_tau+1],ts_emp[i_sub,i_run,:,i_tau:T-n_tau+1+i_tau],axes=(1,1)) / float((T-n_tau))

# renormalization of FC (homogeneous for all sessions)
FC_emp *= 0.5/FC_emp[:,0,0,:,:].mean()
print('max FC value (most of the distribution should be between 0 and 1):',FC_emp.mean())


#%% structural data

# load DTI data
SC_anat = np.load('SC_anat.npy')

# mask for existing connections in EC
mask_EC = np.zeros([N,N],dtype=bool)
# limit DTI value to determine SC (only connections with larger values are tuned)
lim_SC = 0.
# threshold DTI to obtain EC mask
mask_EC[SC_anat>lim_SC] = True
for i in range(N):
    # no self connection
    mask_EC[i,i] = False
    # additional interhemispheric connections (known to be missed by DTI)
    mask_EC[i,N-1-i] = True
print('EC density:',mask_EC.sum()/float(N*(N-1)))

# diagonal mask for input noise matrix (here, no input cross-correlation)
mask_Sigma = np.eye(N,dtype=bool)


#%%  model optimization
J_mod = np.zeros([n_sub,n_run,N,N]) # Jacobian (off-diagonal elements = EC)
Sigma_mod = np.zeros([n_sub,n_run,N,N]) # local variance

for i_sub in range(n_sub):
    for i_run in range(n_run):
        print('sub',i_sub,'; run',i_run)

        # objective FC matrices (empirical)
        FC0_obj = FC_emp[i_sub,i_run,0,:,:]
        FC1_obj = FC_emp[i_sub,i_run,1,:,:]

        # autocovariance for time shifts in v_tau; with lower bound to avoid negative values (cf. log)
        ac_tmp = np.maximum(FC_emp[i_sub,i_run,:,:,:].diagonal(axis1=1,axis2=2),1e-6)
        # time constant for BOLD autocovariances (calculated with lag from 0 to 2 TRs)
        # it relates to inverse of negative slope of autocovariance over all ROIs
        tau_x = -1. / np.polyfit(v_tau,np.log(ac_tmp).mean(1),1)[0]

        # perform model inversion
        (J_mod_tmp,Sigma_mod_tmp) = WBLECmodel.optimize(FC0_obj,FC1_obj,tau_x,mask_EC,mask_Sigma)

        # store results
        J_mod[i_sub,i_run,:,:] = J_mod_tmp
        Sigma_mod[i_sub,i_run,:,:] = Sigma_mod_tmp


# save results
np.save(res_dir+'FC_emp.npy',FC_emp) # empirical spatiotemporal FC
np.save(res_dir+'mask_EC.npy',mask_EC) # mask of optimized connections
np.save(res_dir+'mask_Sigma.npy',mask_Sigma) # mask of optimized Sigma elements

np.save(res_dir+'J_mod.npy',J_mod) # estimated Jacobian matrices (EC + inverse time constant on diagonal)
np.save(res_dir+'Sigma_mod.npy',Sigma_mod) # estimated Sigma matrices
    
