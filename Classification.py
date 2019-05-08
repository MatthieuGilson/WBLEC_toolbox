#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:46:02 2017

@author: matgilson, ainsabato, vpallares
"""

import numpy as np
import sklearn.linear_model as skllm
import sklearn.preprocessing as skppr
import sklearn.pipeline as skppl
import sklearn.neighbors as sklnn
import matplotlib.pyplot as pp


#%% load data and general parameters
n_sub = 22 # number of subjects
n_run = 5  # number of sessions/runs
N = 66 # number of ROIs

# load EC matrices and perform z-scoring within each session to obtain the ranking between EC connections
mask_EC = np.load('model_param/mask_EC.npy')
EC = np.load('model_param/J_mod.npy')
for i_sub in range(n_sub):
    for i_run in range(n_run):
        EC[i_sub,i_run,mask_EC] = (EC[i_sub,i_run,mask_EC] - np.mean(EC[i_sub,i_run,mask_EC])) / np.std(EC[i_sub,i_run,mask_EC])
# vectorized EC matrices (only retaining existing connections)
vect_EC = EC[:,:,mask_EC]

# load FC matrices and calculate BOLD correlations
mask_FC = np.tri(N,N,-1,dtype=np.bool)
corrFC = np.load('model_param/FC_emp.npy')[:,:,0,:,:]
for i_sub in range(n_sub):
    for i_run in range(n_run):
        corrFC[i_sub,i_run,:,:] /= np.sqrt(np.outer(corrFC[i_sub,i_run,:,:].diagonal(),corrFC[i_sub,i_run,:,:].diagonal()))
# vectorized FC matrices (only retaining lower triangle of symmetric matrix)
vect_FC = corrFC[:,:,mask_FC]


# labels of sessions for classification
# movie versus rest
RM_labels = np.repeat(np.array([0,0,1,1,1],dtype=np.int).reshape([1,-1]), n_sub, axis=0)
# movie sessions taken individually + rest (4 tasks)
run_labels = np.repeat(np.array([0,0,1,2,3],dtype=np.int).reshape([1,-1]), n_sub, axis=0)
# subjects
sub_labels = np.repeat(np.arange(n_sub).reshape([-1,1]), n_run, axis=1)


#%% classifiers and learning parameters
c_MLR = skppl.make_pipeline(skppr.StandardScaler(),skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='lbfgs'))
c_1NN = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

# number of repetition of classification procedure
n_rep = 20
# record classification performance (2nd index: EC, FC; 3rd index: MRL, 1NN; 4th index: rest vs movie, 4 tasks, subject identification)
perf = np.zeros([n_rep,2,2,3])


#%% perform classification
for i_rep in range(n_rep):
    # train and test classifiers to discriminate sessions (rest versus movie, 4 tasks)
    # split samples in train and test sets (80% and 20% of subjects, respectively)
    train_ind = np.ones([n_sub,n_run],dtype=bool)
    while train_ind.sum()>0.8*n_sub*n_run:
        train_ind[np.random.randint(n_sub),:] = False
    test_ind = np.logical_not(train_ind)
#    print('train/test sets:',train_ind,test_ind)

    # rest versus movie    
    c_MLR.fit(vect_EC[train_ind,:], RM_labels[train_ind])
    perf[i_rep,0,0,0] = c_MLR.score(vect_EC[test_ind,:], RM_labels[test_ind])

    c_1NN.fit(vect_EC[train_ind,:], RM_labels[train_ind])
    perf[i_rep,0,1,0] = c_1NN.score(vect_EC[test_ind,:], RM_labels[test_ind])

    c_MLR.fit(vect_FC[train_ind,:], RM_labels[train_ind])
    perf[i_rep,1,0,0] = c_MLR.score(vect_FC[test_ind,:], RM_labels[test_ind])

    c_1NN.fit(vect_FC[train_ind,:], RM_labels[train_ind])
    perf[i_rep,1,1,0] = c_1NN.score(vect_FC[test_ind,:], RM_labels[test_ind])

    # 4 tasks
    c_MLR.fit(vect_EC[train_ind,:], run_labels[train_ind])
    perf[i_rep,0,0,1] = c_MLR.score(vect_EC[test_ind,:], run_labels[test_ind])

    c_1NN.fit(vect_EC[train_ind,:], run_labels[train_ind])
    perf[i_rep,0,1,1] = c_1NN.score(vect_EC[test_ind,:], run_labels[test_ind])

    c_MLR.fit(vect_FC[train_ind,:], run_labels[train_ind])
    perf[i_rep,1,0,1] = c_MLR.score(vect_FC[test_ind,:], run_labels[test_ind])

    c_1NN.fit(vect_FC[train_ind,:], run_labels[train_ind])
    perf[i_rep,1,1,1] = c_1NN.score(vect_FC[test_ind,:], run_labels[test_ind])


    # train and test classifiers to identify subjects
    # split samples in train and test sets (2 runs and 3 runs per subject, respectively)
    train_ind = np.zeros([n_sub,n_run],dtype=bool)
    while train_ind.sum()<2*n_sub:
        train_ind[:,np.random.randint(n_run)] = True
    test_ind = np.logical_not(train_ind)
#    print('train/test sets:',train_ind,test_ind)

    c_MLR.fit(vect_EC[train_ind,:], sub_labels[train_ind])
    perf[i_rep,0,0,2] = c_MLR.score(vect_EC[test_ind,:], sub_labels[test_ind])

    c_1NN.fit(vect_EC[train_ind,:], sub_labels[train_ind])
    perf[i_rep,0,1,2] = c_1NN.score(vect_EC[test_ind,:], sub_labels[test_ind])

    c_MLR.fit(vect_FC[train_ind,:], sub_labels[train_ind])
    perf[i_rep,1,0,2] = c_MLR.score(vect_FC[test_ind,:], sub_labels[test_ind])

    c_1NN.fit(vect_FC[train_ind,:], sub_labels[train_ind])
    perf[i_rep,1,1,2] = c_1NN.score(vect_FC[test_ind,:], sub_labels[test_ind])



#%% plot performance

pp.figure()
pp.violinplot(perf[:,0,0,:],positions=np.arange(3)-0.3,widths=[0.15]*3)
pp.violinplot(perf[:,0,1,:],positions=np.arange(3)-0.1,widths=[0.15]*3)
pp.violinplot(perf[:,1,0,:],positions=np.arange(3)+0.1,widths=[0.15]*3)
pp.violinplot(perf[:,1,1,:],positions=np.arange(3)+0.3,widths=[0.15]*3)
pp.plot([-0.5,0.5],[0.6,0.6],'--k')
pp.plot([0.5,1.5],[0.4,0.4],'--k')
pp.plot([1.5,2.5],[1/n_sub,1/n_sub],'--k')
pp.axis(xmin=-0.5,xmax=2.5,ymin=0.0,ymax=1)
pp.xticks(range(3),['rest vs movie','4 tasks','subject identification'],fontsize=8)
pp.ylabel('accuracy',fontsize=8)
pp.title('blue: EC+MLR; orange: EC+1NN; green: FC+MLR; red: FC+1NN\nchance level in dashed',fontsize=8)

pp.show()

