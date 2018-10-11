#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:34:37 2017

@author: matgilson 
Modified by: Natalia Esteve 
"""
 

def main():
    import os
    import numpy as np
    import WBLECmodel
    import scipy.io as sio
    
    os.system('clear')
    
    res_dir = 'model_param/'
    if not os.path.exists(res_dir):
        print('create directory:',res_dir)
        os.makedirs(res_dir)
    
    
    ################# 
    # fMRI properties    
    n_sub = 18 # number of subjects
    n_stage = 4 # vigilance stages: 0=awake, 1=n1 sleep, 2=n2 sleep, 3=n3 sleep
    N = 90 # number of ROIs
    
    
    # time shifts for FC: 0, 1 and 2 TR
    v_tau = np.arange(3,dtype=float)
    n_tau = v_tau.size
    
    
    #################
    # functional data
    ts_awake = dict() # awake fMRI data: row vector containing n-sub fMRI data matrices 
    ts_n1 = dict() # n1 sleep fMRI data: row vector containing n-sub fMRI data matrices 
    ts_n2 = dict() # n2 sleep fMRI data: row vector containing n-sub fMRI data matrices 
    ts_n3 = dict() # n3 sleep fMRI data: row vector containing n-sub fMRI data matrices 
    T=np.zeros((n_sub, n_stage) ,dtype=int)   # matrix with number of TRs of the recording per subject for all stages

    # loading fMRI data and calculating Time vectors  
    for i_sub in range(n_sub): 
        ts_awake[i_sub] = sio.loadmat('data_Awake')['X'][0,i_sub] 
        ts_n1[i_sub] = sio.loadmat('data_N1')['X'][0,i_sub] 
        ts_n2[i_sub] = sio.loadmat('data_N2')['X'][0,i_sub] 
        ts_n3[i_sub] = sio.loadmat('data_N3')['X'][0,i_sub] 
        T[i_sub,0]=ts_awake[i_sub].shape[1]
        T[i_sub,1]=ts_n1[i_sub].shape[1]
        T[i_sub,2]=ts_n2[i_sub].shape[1]
        T[i_sub,3]=ts_n3[i_sub].shape[1]
    
    # centering the time series (i.e substracting mean)
    for i_sub in range(n_sub):
        ts_awake[i_sub] -= np.outer(ts_awake[i_sub].mean(1),np.ones(T[i_sub,0])) 
        ts_n1[i_sub] -= np.outer(ts_n1[i_sub].mean(1),np.ones(T[i_sub,1]))
        ts_n2[i_sub] -= np.outer(ts_n2[i_sub].mean(1),np.ones(T[i_sub,2]))
        ts_n3[i_sub] -= np.outer(ts_n3[i_sub].mean(1),np.ones(T[i_sub,3]))  

    # Calculating Empirical FC (FC = spatiotemporal covariances of BOLD signals)
    FC_emp= np.zeros([n_sub,n_stage,n_tau,N,N]) 
    for i_sub in range(n_sub):
        for i_tau in range(n_tau):
            FC_emp[i_sub,0,i_tau,:,:] = np.tensordot(ts_awake[i_sub][:,0:T[i_sub,0]-n_tau+1],ts_awake[i_sub][:,i_tau:T[i_sub,0]-n_tau+1+i_tau],axes=(1,1))/float((T[i_sub,0]-n_tau))
            FC_emp[i_sub,1,i_tau,:,:] = np.tensordot(ts_n1[i_sub][:,0:T[i_sub,1]-n_tau+1],ts_n1[i_sub][:,i_tau:T[i_sub,1]-n_tau+1+i_tau],axes=(1,1))/float((T[i_sub,1]-n_tau))    
            FC_emp[i_sub,2,i_tau,:,:] = np.tensordot(ts_n2[i_sub][:,0:T[i_sub,2]-n_tau+1],ts_n2[i_sub][:,i_tau:T[i_sub,2]-n_tau+1+i_tau],axes=(1,1))/float((T[i_sub,2]-n_tau))
            FC_emp[i_sub,3,i_tau,:,:] = np.tensordot(ts_n3[i_sub][:,0:T[i_sub,3]-n_tau+1],ts_n3[i_sub][:,i_tau:T[i_sub,3]-n_tau+1+i_tau],axes=(1,1))/float((T[i_sub,3]-n_tau))
                                 
    FC_emp*= 0.5/FC_emp[:,0,0,:,:].mean() # renormalization of FC

   
    #################
    # structural data
    SC_anat = np.load('SC.npy')
    lim_SC = 0. # limit DTI value to determine SC (only connections with larger values are tuned)
    
    # mask for existing connections for EC
    mask_EC = np.zeros([N,N],dtype=bool) # EC weights to tune
    mask_EC[SC_anat>lim_SC] = True
    for i in range(N):
        mask_EC[i,i] = False # no self connection
        mask_EC[i,N-1-i] = True # additional interhemispheric connections
    print('EC density:',mask_EC.sum()/float(N*(N-1))) #EC density: 0.3995006242197253
    
    # diagonal mask for input noise matrix (here, no input cross-correlation)
    mask_Sigma = np.eye(N,dtype=bool)
    
    
    ####################
    # model optimization
    J_mod = np.zeros([n_sub,n_stage,N,N]) # Jacobian (off-diagonal elements = EC)
    Sigma_mod = np.zeros([n_sub,n_stage,N,N]) # local variance
    
    for i_sub in range(n_sub):
        for i_stage in range(n_stage):
            print('sub',i_sub,'; stage',i_stage)
    
            # objective FC matrices (empirical)
            FC0_obj = FC_emp[i_sub,i_stage,0,:,:]
            FC1_obj = FC_emp[i_sub,i_stage,1,:,:]
    
            # time constant for BOLD autocovariances (calculated with lag from 0 to 2 TRs)
            ac_tmp = np.maximum(FC_emp[i_sub,i_stage,:,:,:].diagonal(axis1=1,axis2=2),1e-6) # autocovariance for time shifts in v_tau; with lower bound to avoid negative values (cf. log)
            tau_x = -1. / np.polyfit(v_tau,np.log(ac_tmp).mean(1),1)[0] # inverse of negative slope of autocovariance over all ROIs
    
            #callling model optimization 
            (J_mod_tmp,Sigma_mod_tmp) = WBLECmodel.optimize(FC0_obj,FC1_obj,tau_x,mask_EC,mask_Sigma)
    
            J_mod[i_sub,i_stage,:,:] = J_mod_tmp
            Sigma_mod[i_sub,i_stage,:,:] = Sigma_mod_tmp    
    
    
    # save results
    np.save(res_dir+'FC_emp.npy',FC_emp) # empirical spatiotemporal FC
    np.save(res_dir+'mask_EC.npy',mask_EC) # mask of optimized connections
    np.save(res_dir+'mask_Sigma.npy',mask_Sigma) # mask of optimized Sigma elements   
    np.save(res_dir+'J_mod.npy',J_mod) # estimated Jacobian matrices (EC + inverse time constant on diagonal)
    np.save(res_dir+'Sigma_mod.npy',Sigma_mod) # estimated Sigma matrices
        
    
if __name__=="__main__":
    main()