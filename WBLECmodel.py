#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:34:37 2017

@author: matgilson
"""

import numpy as np
import scipy.linalg as spl
import scipy.stats as stt


def optimize(FC0_obj,FC1_obj,tau_x,mask_EC,mask_Sigma):

    N = FC0_obj.shape[0]
    
    # optimzation rates (to avoid explosion of activity, Sigma is tuned quicker)
    epsilon_EC = 0.0005
    epsilon_Sigma = 0.05
    
    min_val_EC = 0. # minimal value for tuned EC elements
    max_val_EC = 1. # maximal value for tuned EC elements
    min_val_Sigma = 0.01 # minimal value for tuned Sigma elements
    
    # initial EC
    EC = np.zeros([N,N]) # initial connectivity
    Sigma = np.eye(N)  # initial noise

    # record best fit (matrix distance between model and empirical FC)
    best_dist = 1e10
    
    # scaling coefs for FC0 and FC1
    a0 = np.linalg.norm(FC1_obj) / (np.linalg.norm(FC0_obj) + np.linalg.norm(FC1_obj))
    a1 = 1. - a0

    stop_opt = False
    i_opt = 0
    while not stop_opt:

        # calculate Jacobian of dynamical system
        J = -np.eye(N)/tau_x + EC

        # calculate FC0 and FC1 for model
        FC0 = spl.solve_lyapunov(J,-Sigma)
        FC1 = np.dot(FC0,spl.expm(J.T))

        # matrices of model error
        Delta_FC0 = FC0_obj-FC0
        Delta_FC1 = FC1_obj-FC1

        # calculate error between model and empirical data for FC0 and FC_tau (matrix distance)
        dist_FC_tmp = 0.5*(np.linalg.norm(Delta_FC0)/np.linalg.norm(FC0_obj)+np.linalg.norm(Delta_FC1)/np.linalg.norm(FC1_obj))

        # calculate Pearson correlation between model and empirical data for FC0 and FC_tau
        Pearson_FC_tmp = 0.5*(stt.pearsonr(FC0.reshape(-1),FC0_obj.reshape(-1))[0]+stt.pearsonr(FC1.reshape(-1),FC1_obj.reshape(-1))[0])

        # record best model parameters
        if dist_FC_tmp<best_dist:
            best_dist = dist_FC_tmp
            best_Pearson = Pearson_FC_tmp
            i_best = i_opt
            J_mod_best = np.array(J)
            Sigma_mod_best = np.array(Sigma)
        else:
            stop_opt = i_opt>50

        # Jacobian update
        Delta_J = np.dot(np.linalg.pinv(FC0),a0*Delta_FC0+np.dot(a1*Delta_FC1,spl.expm(-J.T))).T

        # update EC (recurrent connectivity)
        EC[mask_EC] += epsilon_EC * Delta_J[mask_EC]
        EC[mask_EC] = np.clip(EC[mask_EC],min_val_EC,max_val_EC)

        # update Sigma (input variances)
        Delta_Sigma = -np.dot(J,Delta_FC0)-np.dot(Delta_FC0,J.T)
        Sigma[mask_Sigma] += epsilon_Sigma * Delta_Sigma[mask_Sigma]
        Sigma[mask_Sigma] = np.maximum(Sigma[mask_Sigma],min_val_Sigma)

        # check for stop
        if not stop_opt:
            i_opt += 1
        else:
            print('stop at step',i_best,'with best FC dist:',best_dist,'; best FC Pearson:',best_Pearson)

    return (J_mod_best,Sigma_mod_best)
    