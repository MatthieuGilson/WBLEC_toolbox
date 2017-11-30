#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:46:02 2017

@author: matgilson
"""


import numpy as np
import scipy.stats as stt



# load EC matrices, session labels and general parameters
EC = np.load('model_param/J_mod.npy')
Sigma = np.load('model_param/Sigma_mod.npy')
mask_EC = np.load('model_param/mask_EC.npy')
mask_Sigma = np.load('model_param/mask_Sigma.npy')

ROI_labels = np.load('ROI_labels.npy')

n_sub = 22
n_run = 5
N = 66

# remove invalid subjects
ind_valid_sub = np.arange(n_sub)
ind_valid_sub = np.delete(ind_valid_sub,[1,11,19])
n_sub = ind_valid_sub.size

EC = EC[ind_valid_sub,:,:,:]
Sigma = Sigma[ind_valid_sub,:,:,:]

# runs to compare
if True:
    # individual runs
    r1 = 0
    r2 = 2
else:
    # all rest versus all movie
    r1 = [0,1]
    r2 = [2,3,4]

# calculate p values for EC
pval_EC = np.ones([N,N])
for i in range(N):
    for j in range(N):
        if mask_EC[i,j]:
            t_tmp, p_tmp = stt.ttest_ind(np.ravel(EC[:,r1,i,j]),np.ravel(EC[:,r2,i,j]),equal_var=False)
            pval_EC[i,j] = p_tmp
            if p_tmp<0.05/mask_EC.sum():
                print('connection from',ROI_labels[j],'to',ROI_labels[i])
            
print()            
            
# calculate p values for Sigma
pval_Sigma = np.ones([N])
for i in range(N):
    t_tmp, p_tmp = stt.ttest_ind(np.ravel(Sigma[:,r1,i,i]),np.ravel(Sigma[:,r2,i,i]),equal_var=False)
    pval_Sigma[i] = p_tmp
    if p_tmp<0.05/mask_Sigma.sum():
        print('variance at',ROI_labels[i])
            
# surrogate distribution
n_surr = mask_EC.sum()
pval_surr = np.ones([n_surr])
rand_Gaussian_var = np.random.rand(n_surr,n_sub,2)
for i_surr in range(n_surr):
    t_tmp, p_tmp = stt.ttest_ind(rand_Gaussian_var[i_surr,:,0],rand_Gaussian_var[i_surr,:,1],equal_var=False)
    pval_surr[i_surr] = p_tmp
            
# compare distributions of p values
import matplotlib.pyplot as pp

v_bins = np.linspace(0,6,20)

pp.figure(figsize=[3,2])
pp.hist(-np.log10(pval_surr),bins=v_bins,histtype='step',normed=True,cumulative=True,color='k')
pp.hist(-np.log10(pval_EC[mask_EC]),bins=v_bins,histtype='step',normed=True,cumulative=True,color='r')
pp.hist(-np.log10(pval_Sigma),bins=v_bins,histtype='step',normed=True,cumulative=True,color='m')
pp.xticks(fontsize=8)
pp.yticks(fontsize=8)
pp.xlabel('-log$_10$(p)',fontsize=8)
#pp.savefig('comp_pval')
pp.show()