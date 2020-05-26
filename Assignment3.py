# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:25:47 2020

@author: ak2524
"""

import numpy as np
import sys

f = open(sys.argv[1])
data = np.loadtxt(f)
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)

print("train=",train)
print("train shape=",train.shape)

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:,1:]
testlabels = data[:,0]

onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)

rows = train.shape[0]
cols = train.shape[1]

hnodes = int(sys.argv[3])

#hidden_nodes = 3

''' Initialize Weights '''
w = np.random.rand(hnodes)
print("w = ",w)


W = np.random.rand(hnodes, cols)
print("W = ",W)

epochs = 10000
eta = .01
prevobj = np.inf
i=0

''' Calculating the Objective '''
hlayer = np.matmul(train, np.transpose(W))
print("hidden_layer = ", hlayer)
print("hidden_layer shape = ", hlayer.shape)

sigmoid = lambda x: 1/(1+np.exp(-x))
hlayer = np.array([sigmoid(xi) for xi in hlayer])
print("hidden_layer = ",hlayer)
print("hidden_layer shape = ", hlayer.shape)

output_layer = np.matmul(hlayer, np.transpose(w))
print("output_layer=",output_layer)

obj = np.sum(np.square(output_layer - trainlabels))
print("obj=",obj)

''' Begin Gradient Descent '''

stop = 0.0000001
while(prevobj - obj > stop and i < epochs):
    prevobj = obj
    
    dellw = (np.dot(hlayer[0,:],w) - trainlabels[0]) * hlayer[0,:]
    for j in range(1, rows): 
        dellw += (np.dot(hlayer[j,:], np.transpose(w)) - trainlabels[j]) * hlayer[j,:]
        
    ''' Update w '''
    w = w - eta*dellw
    
    '''Dells for Hidden Nodes'''
    dells = np.sum(np.dot(hlayer[0,:],w)-trainlabels[0])*w[0] * (hlayer[0,0])*(1-hlayer[0,0])*train[0]
    dellu = np.sum(np.dot(hlayer[1,:],w)-trainlabels[0])*w[1] * (hlayer[0,1])*(1-hlayer[0,1])*train[0]
    dellv = np.sum(np.dot(hlayer[2,:],w)-trainlabels[0])*w[2] * (hlayer[0,2])*(1-hlayer[0,2])*train[0]
    
    for j in range(1, rows):
    		dells += np.sum(np.dot(hlayer[j,:],w)-trainlabels[j])*w[0] * (hlayer[j,0])*(1-hlayer[j,0])*train[j]
    		dellu += np.sum(np.dot(hlayer[j,:],w)-trainlabels[j])*w[1] * (hlayer[j,1])*(1-hlayer[j,1])*train[j]
    		dellv += np.sum(np.dot(hlayer[j,:],w)-trainlabels[j])*w[2] * (hlayer[j,2])*(1-hlayer[j,2])*train[j]
    
    ''''Put dells into rows'''
    dellW = np.vstack((dells,dellu,dellv))
    
    '''Update W'''
    W = W - eta*dellW
    
    '''Recalculate objective'''
    hlayer = np.matmul(train, np.transpose(W))
    print("hidden_layer=", hlayer)

    hlayer = np.array([sigmoid(xi) for xi in hlayer])
    print("hidden_layer=", hlayer)

    output_layer = np.matmul(hlayer, np.transpose(w))
    print("output_layer=", output_layer)

    obj = np.sum(np.square(output_layer - trainlabels))
    print("obj=",obj)

    i = i + 1
    print("Objective=",obj)

''' Final Prediction '''
def predict(x):
    hlayer = np.matmul(x, np.transpose(W))
    hlayer = np.array([sigmoid(xi) for xi in hlayer])
    output_layer = np.matmul(hlayer, np.transpose(w))
    predictions = np.sign(output_layer)
    return predictions

trainpred = predict(train)
trainerr = (1 - (trainpred == trainlabels).mean()) * 100

testpred = predict(test)
testerr = (1 - (testpred == testlabels).mean()) * 100

print("Train predictions:\t", trainpred)
print("Train error:\t", trainerr,'%')

print("Test predictions\t", testpred)
print("Test error\t", testerr,'%')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    