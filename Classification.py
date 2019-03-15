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


# load EC matrices, session labels and general parameters
n_sub = 22
n_run = 5
N = 66

EC = np.load('model_param/J_mod.npy')
mask_EC = np.load('model_param/mask_EC.npy')

vect_EC = EC[:,:,mask_EC] # vectorized EC matrices (only retaining existing connections)
dim_feature = vect_EC.shape[2] # dimension of vectorized EC


# labels of sessions for classification (train+test)
RM_labels = np.repeat(np.array([0,0,1,1,1],dtype=np.int).reshape([1,-1]), n_sub, axis=0)
sub_labels = np.repeat(np.arange(n_sub).reshape([-1,1]), n_run, axis=1)


# classifier and learning parameters
c_MLR = skppl.make_pipeline(skppr.StandardScaler(),skllm.LogisticRegression(C=1, penalty='l2', multi_class='multinomial', solver='lbfgs'))
c_1NN = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

n_rep = 20  # number of repetition of classification procedure
perf = np.zeros([n_rep,2,2]) # record classification performance; last index: 1) rest vs movie; 2) subject identification


# perform classification
for i_rep in range(n_rep):
    # split samples in train and test sets (80% and 20% of subjects, respectively)
    train_labels = np.ones([n_sub,n_run],dtype=bool)
    while train_labels.sum()>0.8*n_sub*n_run:
        train_labels[np.random.randint(n_sub),:] = False
    test_labels = np.logical_not(train_labels)
#    print('train/test sets:',train_labels,test_labels)
    
    # train and test classifiers with run labels
    c_MLR.fit(vect_EC[train_labels,:].reshape([-1,dim_feature]), RM_labels[train_labels].reshape([-1]))
    perf[i_rep,0,0] = c_MLR.score(vect_EC[test_labels,:].reshape([-1,dim_feature]), RM_labels[test_labels].reshape([-1]))

    c_1NN.fit(vect_EC[train_labels,:].reshape([-1,dim_feature]), RM_labels[train_labels].reshape([-1]))
    perf[i_rep,1,0] = c_1NN.score(vect_EC[test_labels,:].reshape([-1,dim_feature]), RM_labels[test_labels].reshape([-1]))


    # split samples in train and test sets (2 runs and 3 runs per subject, respectively)
    train_labels = np.zeros([n_sub,n_run],dtype=bool)
    while train_labels.sum()<2*n_sub:
        train_labels[:,np.random.randint(n_run)] = True
    test_labels = np.logical_not(train_labels)
#    print('train/test sets:',train_labels,test_labels)


    c_MLR.fit(vect_EC[train_labels,:].reshape([-1,dim_feature]), sub_labels[train_labels].reshape([-1]))
    perf[i_rep,0,1] = c_MLR.score(vect_EC[test_labels,:].reshape([-1,dim_feature]), sub_labels[test_labels].reshape([-1]))

    c_1NN.fit(vect_EC[train_labels,:].reshape([-1,dim_feature]), sub_labels[train_labels].reshape([-1]))
    perf[i_rep,1,1] = c_1NN.score(vect_EC[test_labels,:].reshape([-1,dim_feature]), sub_labels[test_labels].reshape([-1]))



# plot perf

pp.figure()
pp.violinplot(perf[:,:,0],positions=np.arange(2)-0.2,widths=[0.3]*2)
pp.violinplot(perf[:,:,1],positions=np.arange(2)+0.2,widths=[0.3]*2)
pp.plot([-1,2],[0.4,0.4],'--k')
pp.axis(xmin=-0.5,xmax=1.5,ymin=0.0,ymax=1)
pp.xticks([0,1],['MLR','1NN'],fontsize=8)
pp.ylabel('accuracy',fontsize=8)
pp.title('left: rest vs movie; right: subject identification',fontsize=8)

pp.show()

