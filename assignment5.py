import numpy as np
import sys
import os
import pandas as pd
from scipy import signal
from sklearn.metrics import accuracy_score

sigmoid = lambda x: 1/(1+ np.exp(-x))

train_dir = sys.argv[1]
test_dir = sys.argv[2]

df = pd.read_csv(train_dir+'/data.csv')
names = df['Name'].values
labels = df['Label'].values 

train_data = np.empty((len(labels),3,3), dtype=np.float )

for i in range(0,len(labels)):
    image_matrix = np.loadtxt(train_dir+'/'+names[i])
    train_data[i] = image_matrix



df_test = pd.read_csv(test_dir+'/data.csv')
test_name = df_test['Name'].values
test_label = df_test['Label'].values 

test_data = np.empty((len(test_label),3,3),dtype=np.float)

for i in range(0,len(test_label)):
    test_image_matrix = np.loadtxt(test_dir+'/'+test_name[i])
    test_data[i] = test_image_matrix



c = np.ones((2,2),dtype=np.float)

epochs = 1000
eta = 0.1
stop = 0.01
prevobj = np.inf
i=0

obj = 0 
for i in range(0,len(labels)):
    hidden_layer = signal.convolve2d(train_data[i],c, mode='valid')
    for j in range(0,2,1):
        for k in range(0,2,1):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
    output_layer = (hidden_layer[0][0] + hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4
    obj += (output_layer - labels[i])**2



while(prevobj - obj >stop and i<epochs):

    prevobj = obj

    dellc1 = 0
    dellc2 = 0
    dellc3 = 0
    dellc4 = 0

    f = (output_layer)**0.5

    for i in range(0,len(labels)):
        hidden_layer = signal.convolve2d(train_data[i],c,mode="valid")
        for j in range(0,2,1):
            for k in range(0,2,1):
                hidden_layer[j][k]= sigmoid(hidden_layer[j][k])

        #gd c1
        sqrtf = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1])/4 - labels[i]
        dz1dc1 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*train_data[i][0][0]
        dz2dc1 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*train_data[i][0][1]
        dz3dc1 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*train_data[i][1][0]
        dz4dc1 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*train_data[i][1][1]
        dellc1 += (sqrtf * (dz1dc1 + dz2dc1 + dz3dc1 +dz4dc1))/2
        #gd c2
        dz1dc2 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*train_data[i][0][1]
        dz2dc2 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*train_data[i][0][2]
        dz3dc2 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*train_data[i][1][1]
        dz4dc2 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*train_data[i][1][2]
        dellc2 += (sqrtf * (dz1dc2 + dz2dc2 + dz3dc2 +dz4dc2))/2
        #gd c3
        dz1dc3 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*train_data[i][1][0]
        dz2dc3 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*train_data[i][1][1]
        dz3dc3 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*train_data[i][2][0]
        dz4dc3 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*train_data[i][2][1]
        dellc3 += (sqrtf * (dz1dc3 + dz2dc3 + dz3dc3 +dz4dc3))/2
        #gd c4
        dz1dc4 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*train_data[i][1][1]
        dz2dc4 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*train_data[i][1][2]
        dz3dc4 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*train_data[i][2][1]
        dz4dc4 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*train_data[i][2][2]
        dellc4 += (sqrtf * (dz1dc4 + dz2dc4 + dz3dc4 +dz4dc4))/2

        
    c[0][0] -= eta*dellc1
    c[0][1] -= eta*dellc2
    c[1][0] -= eta*dellc3
    c[1][1] -= eta*dellc4

    obj = 0 
    for i in range(0,len(labels)):
        hidden_layer = signal.convolve2d(train_data[i],c, mode='valid')
        for j in range(0,2,1):
            for k in range(0,2,1):
                hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
        output_layer = (hidden_layer[0][0] + hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4
        obj += (output_layer - labels[i])**2


print("C=",c)
print('\n'+"output=")
pred_label = []
for i in range(0,len(test_label)):
    hidden_layer = signal.convolve2d(test_data[i],c, mode='valid')
    for j in range(0,2,1):
        for k in range(0,2,1):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
    output_layer = (hidden_layer[0][0] + hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4
    if (output_layer < 0.5):
        pred_label.append(-1)
        print(-1)
    else:
        pred_label.append(1)
        print(1)
print('accuracy',accuracy_score(test_label, pred_label))
