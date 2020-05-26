import numpy as np
import sys

def shuffle(data, mini_batch_size):
        np.random.shuffle(data)
        train_s = data[0:mini_batch_size,1:]
        onearray = np.ones((train_s.shape[0],1))
        train_s = np.append(train_s,onearray,axis=1)
        trainlabel_s = data[0:mini_batch_size,0]

        return train_s, trainlabel_s, data
	

#################
### Read data ###

f = open(sys.argv[1])
# f = open("test.0")
data = np.loadtxt(f)
data_s = data.copy()
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)

# print("train=",train)
# print("train shape=",train.shape)

f = open(sys.argv[2])
# f = open("test.0")
data = np.loadtxt(f)
test = data[:,1:]
testlabels = data[:,0]

onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)

rows = train.shape[0]
cols = train.shape[1]
# print(np.shape(test))
# print(np.shape(train))
# exit()
#Set a size for minibatch and create a random row number array
mini_batch_size = int(sys.argv[3])
# mini_batch_size = 4
mini_batch_array = np.random.randint(0,len(train),mini_batch_size)

# print(mini_batch_array)
# exit()

# np.random.shuffle(data_s)
train_s = data_s[:mini_batch_size,1:]
onearray = np.ones((train_s.shape[0],1))
train_s = np.append(train_s,onearray,axis=1)
trainlabel_s = data_s[:mini_batch_size,0]
# print(np.shape(trainlabel_s))
# exit()

# print(mini_batch_array)
# exit()

#hidden_nodes = int(sys.argv[3])

hidden_nodes = 3


##############################
### Initialize all weights ###

#w = np.random.rand(1,hidden_nodes)
w = np.random.rand(hidden_nodes)
# print("w=",w)
# print(np.shape(w))
# exit()

#check this command
#W = np.zeros((hidden_nodes, cols), dtype=float)
# W = np.ones((hidden_nodes, cols), dtype=float)
W = np.random.rand(hidden_nodes, cols)
# print("W=",W)
# print(np.shape(W))
# exit()


epochs = 30000
#please read through my entire code before changing the epoch values, I am using a slightly different apporach
eta = 0.001
stop = 0.000000001
prevobj = np.inf
i=0

#print("epochs=", epochs)
#print("eta=", eta)


###########################
### Calculate objective ###

hidden_layer = np.matmul(train, np.transpose(W))
hidden_layer_mb = np.matmul(train_s, np.transpose(W))

# print("hidden_layer_dim=",np.shape(hidden_layer))
# print("hidden_layer shape=",hidden_layer.shape)
# def relu(arr):
# 	for i in range(len(arr)):
# 		if(arr[i] > 0):
# 			arr[i] = arr[i]
# 		else:
# 			arr[i] = 0
# 	return arr
	

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
hidden_layer_mb = np.array([sigmoid(xi) for xi in hidden_layer_mb])

output_layer = np.matmul(hidden_layer, np.transpose(w))
# print("output_layer=",output_layer)

obj = np.sum(np.square(output_layer - trainlabels))

###############################
### Begin gradient descent ####
# print(train)
# print((train_s))
# exit()

# b_w = 10

b_w = np.random.rand(hidden_nodes)
b_W = np.random.rand(hidden_nodes, cols)

bestobj = 1000
while(i < epochs ):
#while(prevobj - obj > 0):
        # train, trainlabels, data =  shuffle(data, mini_batch_size)
        # print(train[0])
	mini_batch_array = np.random.randint(0,train.shape[0],mini_batch_size)
	# mini_batch_array = np.random.choice(train.shape[0],mini_batch_size, replace=False)
	prevobj = obj
	w = b_w
	W = b_W
	
	dellw = (np.dot(hidden_layer_mb[0,:],w)-trainlabels[mini_batch_array[0]])*hidden_layer_mb[0,:]
	for j in range(1, mini_batch_size):
		dellw += (np.dot(hidden_layer_mb[j,:],np.transpose(w))-trainlabels[mini_batch_array[j]])*hidden_layer_mb[j,:]

	#Update w
	w = w - eta*dellw

	#print("dellf=",dellf)
	
	#Calculate gradient update for hidden layer weights (W)
	#dellW has to be of same dimension as W

	#Let's first calculate dells. After that we do dellu and dellv.
	#Here s, u, and v are the three hidden nodes
	#dells = df/dz1 * (dz1/ds1, dz1,ds2)
	dells = np.sum(np.dot(hidden_layer_mb[0,:],w)-trainlabels[mini_batch_array[0]])*w[0] * (hidden_layer_mb[0,0])*(1-hidden_layer_mb[0,0])*train[mini_batch_array[0]]
	dellu = np.sum(np.dot(hidden_layer_mb[0,:],w)-trainlabels[mini_batch_array[0]])*w[1] * (hidden_layer_mb[0,1])*(1-hidden_layer_mb[0,1])*train[mini_batch_array[0]]
	dellv = np.sum(np.dot(hidden_layer_mb[0,:],w)-trainlabels[mini_batch_array[0]])*w[2] * (hidden_layer_mb[0,2])*(1-hidden_layer_mb[0,2])*train[mini_batch_array[0]]
	for j in range(1, mini_batch_size):
		dells += np.sum(np.dot(hidden_layer_mb[j,:],w)-trainlabels[mini_batch_array[j]])*w[0] * (hidden_layer_mb[j,0])*(1-hidden_layer_mb[j,0])*train[mini_batch_array[j]]
		dellu += np.sum(np.dot(hidden_layer_mb[j,:],w)-trainlabels[mini_batch_array[j]])*w[1] * (hidden_layer_mb[j,1])*(1-hidden_layer_mb[j,1])*train[mini_batch_array[j]]
		dellv += np.sum(np.dot(hidden_layer_mb[j,:],w)-trainlabels[mini_batch_array[j]])*w[2] * (hidden_layer_mb[j,2])*(1-hidden_layer_mb[j,2])*train[mini_batch_array[j]]


	#TODO: Put dells, dellu, and dellv as rows of dellW
	dellW = np.array([dells, dellu, dellv])

	#Update W
	for k in range(3):
		W[k] = W[k] - eta*dellW[k]


	#Recalculate objective
	hidden_layer_mb = np.matmul(train, np.transpose(W))
	# print("hidden_layer=",hidden_layer)

	hidden_layer_mb = np.array([sigmoid(xi) for xi in hidden_layer_mb])
	# print("hidden_layer=",hidden_layer)

	output_layer = (np.matmul(hidden_layer_mb, np.transpose(w)))
	
	obj = np.sum(np.square(output_layer - trainlabels))
	#print("obj=",obj)

	if(obj < bestobj):
		bestobj = obj
		b_w = w
		b_W = W
	else:
		b_w = b_w
		b_W = b_W

	i = i+1
	

predict_hidden = sigmoid(np.matmul(test, np.transpose(b_W)))
predictions = np.sign(np.matmul(predict_hidden,np.transpose(b_w)))

print(predictions)

