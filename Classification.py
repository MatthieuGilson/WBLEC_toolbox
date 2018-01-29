#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:46:02 2017

@author: matgilson, ainsabato, vpallares
"""

import numpy as np
import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn


# load EC matrices, session labels and general parameters
n_sub = 22
n_run = 5
N = 66

EC = np.load('model_param/J_mod.npy')
mask_EC = np.load('model_param/mask_EC.npy')

vect_EC = EC[:,:,mask_EC] # vectorized EC matrices (only retaining existing connections)
dim_feature = vect_EC.shape[2] # dimension of vectorized EC


# labels of sessions for classification (train+test)
sub_labels = np.repeat(np.arange(n_sub).reshape([-1,1]), n_run, axis=1)
run_labels = np.repeat(np.arange(n_run).reshape([1,-1]), n_sub, axis=0)


# classifier and learning parameters
c_MLR = skllm.LogisticRegression(C=10000, penalty='l2', multi_class='multinomial', solver='lbfgs')
c_1NN = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

n_rep = 6  # number of repetition of classification procedure
perf = np.zeros([n_rep,2]) # record classification performance


# perform classification
for i_rep in range(n_rep):
    # split run indices in train and test sets
    run_train_labels = np.zeros([n_run],dtype=bool)
    run_test_labels = np.ones([n_run],dtype=bool)
    while run_train_labels.sum()<2:
        rnd_int = np.random.randint(n_run)
        if not run_train_labels[rnd_int]:
            run_train_labels[rnd_int] = True
            run_test_labels[rnd_int] = False
    print('train/test sets:',run_train_labels,run_test_labels)
    
    # train and test classifiers with subject labels
    c_MLR.fit(vect_EC[:,run_train_labels,:].reshape([-1,dim_feature]), sub_labels[:,run_train_labels].reshape([-1]))
    perf[i_rep,0] = c_MLR.score(vect_EC[:,run_test_labels,:].reshape([-1,dim_feature]), sub_labels[:,run_test_labels].reshape([-1]))

    c_1NN.fit(vect_EC[:,run_train_labels,:].reshape([-1,dim_feature]), sub_labels[:,run_train_labels].reshape([-1]))
    perf[i_rep,1] = c_1NN.score(vect_EC[:,run_test_labels,:].reshape([-1,dim_feature]), sub_labels[:,run_test_labels].reshape([-1]))



# plot perf
print('average/std performance MLR',perf[:,0].mean(),perf[:,0].std())
print('average/std performance 1NN',perf[:,1].mean(),perf[:,1].std())

import matplotlib.pyplot as pp
pp.violinplot(perf,positions=[0,1])
pp.axis(xmin=-0.4,xmax=1.4,ymin=0.5,ymax=1)
pp.xticks([0,1],['MLR','1NN'],fontsize=8)
pp.ylabel('accuracy',fontsize=8)
pp.show()

