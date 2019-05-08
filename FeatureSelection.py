#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:46:02 2017

@author: matgilson
"""


import numpy as np
import scipy.stats as stt
import sklearn.linear_model as skllm
import sklearn.pipeline as skppl
import sklearn.preprocessing as skprp
import sklearn.feature_selection as skfs
import matplotlib.pyplot as pp


#%% load data and general parameters

# load EC matrices, ROI labels and general parameters
EC = np.load('model_param/J_mod.npy')
mask_EC = np.load('model_param/mask_EC.npy')

ROI_labels = np.load('ROI_labels.npy')

n_sub = 22 # number of subjects
n_run = 5  # number of runs
N = 66 # number of ROIs
n_EC = mask_EC.sum() # number of EC links

# perform z-scoring within each session to obtain the ranking between EC connections
for i_sub in range(n_sub):
    for i_run in range(n_run):
        EC[i_sub,i_run,mask_EC] = (EC[i_sub,i_run,mask_EC] - np.mean(EC[i_sub,i_run,mask_EC])) / np.std(EC[i_sub,i_run,mask_EC])
# vectorized EC matrices (only retaining existing connections)
vect_EC = EC[:,:,mask_EC]


# labels of sessions for classification: rest versus movie
RM_labels = np.repeat(np.array([0,0,1,1,1],dtype=np.int).reshape([1,-1]), n_sub, axis=0)
# labels of sessions for classification: rest + movie sessions taken individually (4 tasks)
#run_labels = np.repeat(np.array([0,0,1,2,3],dtype=np.int).reshape([1,-1]), n_sub, axis=0)



# classifier
class RFE_pipeline(skppl.Pipeline):
    def fit(self, X, y=None, **fit_params):
        """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
        """
        super(RFE_pipeline, self).fit(X, y, **fit_params)
        self.coef_ = self.steps[-1][-1].coef_
        return self

common_c = 10 # classifier optimization parameter

classifier = RFE_pipeline([('std_scal',skprp.StandardScaler()),('clf',skllm.LogisticRegression(C=common_c, penalty='l2', multi_class='multinomial', solver='lbfgs'))])


# number of repetitions
n_rep = 20

# save rankings for each split
rk_EC = np.zeros([n_rep,n_EC],dtype=np.int)


#%% perform recursive feature elimination (RFE)
print('perform recursive feature elimination (RFE)')
for i_rep in range(n_rep):
    print(i_rep)
    
    # train and test classifiers to discriminate sessions (rest versus movie, 4 tasks)
    # split samples in train and test sets (80% and 20% of subjects, respectively)
    train_ind = np.ones([n_sub,n_run],dtype=bool)
    while train_ind.sum()>0.8*n_sub*n_run:
        train_ind[np.random.randint(n_sub),:] = False
    test_ind = np.logical_not(train_ind)
    
    # tune classifier and perform RFE with cross-validation to obtain ranking
    RFE_clf = skfs.RFE(classifier,n_features_to_select=10)
    RFE_clf.fit(vect_EC[train_ind,:],RM_labels[train_ind])
    rk_EC[i_rep,:] += RFE_clf.ranking_


mean_rk_EC = rk_EC.mean(0)

surr_rk_EC = np.zeros(rk_EC.shape)
for i_rep in range(n_rep):
    surr_rk_EC[i_rep,:] = np.random.permutation(rk_EC[i_rep,:])

print('variability of ranking:', rk_EC.std(0).mean())
print('to be compared with:', surr_rk_EC.std(0).mean())

Pearson_rk = []
for i_rep1 in range(n_rep):
    for i_rep2 in range(i_rep1):
        Pearson_rk += [stt.pearsonr(rk_EC[i_rep1,:],rk_EC[i_rep2,:])[0]]

print('Pearson of rankings (mean, std):', np.mean(Pearson_rk), np.std(Pearson_rk))


#%% check classification performance to evaluate optimal number of features
n_conn = 40 # number of steps
step_conn = 1 # number of best features (connections) at each step
perf = np.zeros([n_conn,n_rep]) # classification performance
for i_conn in range(1,n_conn):
    print(i_conn)

    # choose best feeatures
    selected_features = np.argsort(mean_rk_EC)[:i_conn*step_conn]
    # perform classification for n_rep repetitions with random splits
    for i_rep in range(n_rep):
        # split samples in train and test sets (80% and 20% of subjects, respectively)
        train_ind = np.ones([n_sub,n_run],dtype=bool)
        while train_ind.sum()>0.8*n_sub*n_run:
            train_ind[np.random.randint(n_sub),:] = False
        test_ind = np.logical_not(train_ind)
        # train and test to evaluate classification accuracy
        classifier.fit(vect_EC[train_ind,:][:,selected_features],RM_labels[train_ind])
        perf[i_conn,i_rep] = classifier.score(vect_EC[test_ind,:][:,selected_features],RM_labels[test_ind])


# plot of evolution of classification accuracy with 
pp.figure()
pp.errorbar(np.arange(n_conn)*step_conn,perf.mean(1),yerr=perf.std(1))
pp.xlabel('number of best features')
pp.ylabel('accuracy')
            
            
#%% Comparison with statistical testing

# runs to compare: rest versus movie
runs_rest = [0,1]
runs_movie = [2,3,4]

# calculate p-values for EC
pval_EC = np.ones([N,N])
for i in range(N):
    for j in range(N):
        if mask_EC[i,j]:
            t_tmp, p_tmp = stt.ttest_ind(np.ravel(EC[:,runs_rest,i,j]),np.ravel(EC[:,runs_movie,i,j]),equal_var=False)
            pval_EC[i,j] = p_tmp
            if p_tmp<0.05/mask_EC.sum():
                print('connection from', ROI_labels[j], 'to', ROI_labels[i])
      
# surrogate distribution
n_surr = mask_EC.sum()
pval_surr = np.ones([n_surr])
rand_Gaussian_var = np.random.rand(n_surr,n_sub,2)
for i_surr in range(n_surr):
    t_tmp, p_tmp = stt.ttest_ind(rand_Gaussian_var[i_surr,:,0],rand_Gaussian_var[i_surr,:,1],equal_var=False)
    pval_surr[i_surr] = p_tmp

            
#%% plots

# histogram of p-values
    
v_bins = np.linspace(0,10,20)

pp.figure()
pp.hist(-np.log10(pval_surr),bins=v_bins,histtype='step',normed=True,cumulative=True,color='k')
pp.hist(-np.log10(pval_EC[mask_EC]),bins=v_bins,histtype='step',normed=True,cumulative=True,color='r')
pp.xlabel('-log$_10$(p)')
pp.ylabel('cumulative distribution')


# comparison of distributions of p values

pp.figure()
pp.plot(-np.log10(pval_EC[mask_EC]),mean_rk_EC,'xr')
pp.xlabel('-log$_10$(p)')
pp.ylabel('ranking')


pp.show()