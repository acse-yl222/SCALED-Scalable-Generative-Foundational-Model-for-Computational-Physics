#%precision 3
#%matplotlib inline
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

import time
import platform
print('python version', platform.python_version())
print('torch version', torch.__version__)
print('numpy version', np.version.version)

#cat /usr/local/cuda/version.txt

########################################################################################
t_read_0 = time.time()

# read in the data (1000 csv files)
nTrain = 800
nValid = 100
nTest = 100
nNodes = 20550 # should really work this out

# [:, :, 2] is speed, [:, :, 3] is u, [:, :, 4] is v
# (speed not really needed)
# [:, :, 0] and [:, :, 1] are the SFC orderings

training_data = np.zeros((nTrain,nNodes,4))
for i in range(nTrain):
    data = np.loadtxt('csv_data/data_' +str(i)+ '.csv', delimiter=',')
    training_data[i,:,:] = data
training_data = np.array(training_data)
print('size training data', training_data.shape)

valid_data = np.zeros((nValid,nNodes,4))
for i in range(nTrain,nTrain+nValid):
    data = np.loadtxt('csv_data/data_' +str(i)+ '.csv', delimiter=',')
    valid_data[i-nTrain,:,:] = data
valid_data = np.array(valid_data)
print('size validation data', valid_data.shape)

test_data = np.zeros((nTest,nNodes,4))
for i in range(nTrain+nValid,nTrain+nValid+nTest):
    data = np.loadtxt('csv_data/data_' +str(i)+ '.csv', delimiter=',')
    test_data[i-nTrain-nValid,:,:] = data
test_data = np.array(test_data)
print('size test data', test_data.shape)

t_read_1 = time.time()

########################################################################################
# rescale the data so that u and v data lies in the range [-1,1] (and speed in [0,1])
# ma = np.max(training_data[:, :, 2])
# mi = np.min(training_data[:, :, 2])
# k = 1./(ma - mi)
# b = 1 - k*ma #k*mi
# training_data[:, :, 2] = k * training_data[:, :, 2] + b #- b
# this won't be used

ma = np.max(training_data[:, :, 2])
mi = np.min(training_data[:, :, 2])
ku = 2./(ma - mi)
bu = 1 - ku*ma
training_data[:, :, 2] = ku * training_data[:, :, 2] + bu
valid_data[:, :, 2] = ku * valid_data[:, :, 2] + bu
test_data[:, :, 2] = ku * test_data[:, :, 2] + bu

ma = np.max(training_data[:, :, 3])
mi = np.min(training_data[:, :, 3])
kv = 2./(ma - mi)
bv = 1 - kv*ma
training_data[:, :, 3] = kv * training_data[:, :, 3] + bv
valid_data[:, :, 3] = kv * valid_data[:, :, 3] + bv
test_data[:, :, 3] = kv * test_data[:, :, 3] + bv

########################################################################################
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

########################################################################################
# SFC-CAE: one curve with nearest neighbour smoothing and compressing to 128 latent variables
print("compress to 128")
torch.manual_seed(42)
# Hyper-parameters
EPOCH = 100
BATCH_SIZE = 800
LR = 0.001
k = nNodes # number of nodes - this has to match training_data.shape[0]
print(training_data.shape) # nTrain by number of nodes by 5

# Data Loader for easy mini-batch return in training
train_loader = Data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder_h1 = nn.Sequential(
            # input shape (b, 1, 20, 550)
            nn.Tanh(),
            nn.Conv1d(4, 16, 32, 4, 16),
            # output shape (b, 4, 5138)
            nn.Tanh(),
            # activation
            nn.Conv1d(16, 16, 32, 4, 16),
            # output shape (b, 8,1285)
            nn.Tanh(),
            # activation
            nn.Conv1d(16, 16, 32, 4, 16),
            # output shape (b, 1, 6, 322)
            nn.Tanh(),
            # activation
            nn.Conv1d(16, 16, 32, 4, 16),
            # output shape (b, 3, 2, 81)
            nn.Tanh(),
            # activation
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1296, 128),
            nn.Tanh(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 1296),
            nn.Tanh(),
        )
        self.decoder_h1 = nn.Sequential(
            # (b, 16, 81)
            nn.Tanh(),
            nn.ConvTranspose1d(16, 16, 32, 4, 15), # (b, 8, 322)
            nn.Tanh(),
            nn.ConvTranspose1d(16, 16, 32, 4, 15), # (b, 4, 1286)
            nn.Tanh(),
            nn.ConvTranspose1d(16, 16, 32, 4, 16), # (b, 2, 5140)
            nn.Tanh(),
            nn.ConvTranspose1d(16, 4, 32, 4, 19), # (b, 1, 20550)
            nn.Tanh(),
        )

        # input sparse layers, initialize weight as 0.33, bias as 0
        self.weight1 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight1_0 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight1_1 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.bias1 = torch.nn.Parameter(torch.FloatTensor(torch.zeros(k)),requires_grad = True)
        self.weight11 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight11_0 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight11_1 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.bias11 = torch.nn.Parameter(torch.FloatTensor(torch.zeros(k)),requires_grad = True)
        self.weight2 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight2_0 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight2_1 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.bias2 = torch.nn.Parameter(torch.FloatTensor(torch.zeros(k)),requires_grad = True)
        self.weight22 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight22_0 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight22_1 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.bias22 = torch.nn.Parameter(torch.FloatTensor(torch.zeros(k)),requires_grad = True)
        self.weight3 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight3_0 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight3_1 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.bias3 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.zeros(k)),requires_grad = True)
        self.weight33 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight33_0 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight33_1 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.bias33 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.zeros(k)),requires_grad = True)
        self.weight4 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight4_0 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight4_1 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.bias4 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.zeros(k)),requires_grad = True)
        self.weight44 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight44_0 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.weight44_1 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.ones(k)),requires_grad = True)
        self.bias44 = torch.nn.Parameter(torch.FloatTensor(0.33 * torch.zeros(k)),requires_grad = True)
        
        # output sparse layers, initialize weight as 0.5, bias as 0
        self.weight_out1 = torch.nn.Parameter(torch.FloatTensor(0.083 *torch.ones(k)),requires_grad = True) 
        self.weight_out1_0 = torch.nn.Parameter(torch.FloatTensor(0.083* torch.ones(k)),requires_grad = True) 
        self.weight_out1_1 = torch.nn.Parameter(torch.FloatTensor(0.083* torch.ones(k)),requires_grad = True)
        
        self.weight_out11 = torch.nn.Parameter(torch.FloatTensor(0.083 *torch.ones(k)),requires_grad = True) 
        self.weight_out11_0 = torch.nn.Parameter(torch.FloatTensor(0.083* torch.ones(k)),requires_grad = True) 
        self.weight_out11_1 = torch.nn.Parameter(torch.FloatTensor(0.083* torch.ones(k)),requires_grad = True)
        
        self.weight_out2 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True)
        self.weight_out2_0 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True)
        self.weight_out2_1 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True)
        
        self.weight_out22 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True)
        self.weight_out22_0 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True)
        self.weight_out22_1 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True)
        
        self.weight_out3 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True) 
        self.weight_out3_0 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True) 
        self.weight_out3_1 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True) 
        
        self.weight_out33 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True) 
        self.weight_out33_0 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True) 
        self.weight_out33_1 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True) 
        
        self.weight_out4 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True) 
        self.weight_out4_0= torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True) 
        self.weight_out4_1 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True) 
        
        self.weight_out44 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True) 
        self.weight_out44_0= torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True) 
        self.weight_out44_1 = torch.nn.Parameter(torch.FloatTensor(0.083 * torch.ones(k)),requires_grad = True)
        
        self.bias_out1 = torch.nn.Parameter(torch.FloatTensor(torch.zeros(k)),requires_grad = True)
        self.bias_out2 = torch.nn.Parameter(torch.FloatTensor(torch.zeros(k)),requires_grad = True)


    def forward(self, x):

        # first curve
        ToSFC1 = x[:, :, 0] # sfc indices
        ToSFC1Up = torch.zeros_like(ToSFC1)
        ToSFC1Down = torch.zeros_like(ToSFC1)
        ToSFC1Up[:-1] = ToSFC1[1:]
        ToSFC1Up[-1] = ToSFC1[-1]
        ToSFC1Down[1:]=ToSFC1[:-1]
        ToSFC1Down[0]=ToSFC1[0]

        batch_num = ToSFC1.shape[0]
        #print("ToSFC1",ToSFC1.shape) # (16, 20550)
        x1 = x[:, :, 2:4] # u and v
        #print("x1", x1.shape) #        # (16, 20550, 2)
        x1_1d = torch.zeros((batch_num, 4, k)).to(device)
        # first input sparse layer, then transform to sfc order1
        for j in range(batch_num):
            x1_1d[j, 0, :] = x1[j, :, 0][ToSFC1[j].long()] * self.weight1 + \
                             x1[j, :, 0][ToSFC1Up[j].long()] * self.weight1_0 + \
                             x1[j, :, 0][ToSFC1Down[j].long()] * self.weight1_1 + self.bias1
        
            x1_1d[j, 1, :] = x1[j, :, 0][ToSFC1[j].long()] * self.weight11 + \
                             x1[j, :, 0][ToSFC1Up[j].long()] * self.weight11_0 + \
                             x1[j, :, 0][ToSFC1Down[j].long()] * self.weight11_1 + self.bias11

            x1_1d[j, 2, :] = x1[j, :, 1][ToSFC1[j].long()] * self.weight2 + \
                             x1[j, :, 1][ToSFC1Up[j].long()] * self.weight2_0 + \
                             x1[j, :, 1][ToSFC1Down[j].long()] * self.weight2_1 + self.bias2

            x1_1d[j, 3, :] = x1[j, :, 1][ToSFC1[j].long()] * self.weight22 + \
                             x1[j, :, 1][ToSFC1Up[j].long()] * self.weight22_0 + \
                             x1[j, :, 1][ToSFC1Down[j].long()] * self.weight22_1 + self.bias22

        # first cnn encoder
        encoded_1 = self.encoder_h1(x1_1d.view(-1, 4, k)) #(b, 32, 81)
        #print("encoded_1", encoded_1.shape
        # flatten and concatenate
        encoded_3 = encoded_1.view(-1,1296)
        #print("encoded_3", encoded_3.shape)
        # fully connection
        encoded = self.fc1(encoded_3) # (b,64)
        decoded_3 = self.decoder_h1(self.fc2(encoded).view(-1, 16, 81))
        #print("decoded_3", decoded_3.shape) # (16, 2, 20550)
        BackSFC1 = torch.argsort(ToSFC1)
        BackSFC1Up = torch.argsort(ToSFC1Up)
        BackSFC1Down = torch.argsort(ToSFC1Down)

        decoded_sp = torch.zeros((batch_num, k, 2)).to(device)
        # output sparse layer, resort according to sfc transform
        for j in range(batch_num):
            decoded_sp[j, :, 0] = decoded_3[j, 0, :][BackSFC1[j].long()]* self.weight_out1 + \
                                  decoded_3[j, 0, :][BackSFC1Up[j].long()] * self.weight_out1_0 + \
                                  decoded_3[j, 0, :][BackSFC1Down[j].long()] * self.weight_out1_1 + \
                                  decoded_3[j, 1, :][BackSFC1[j].long()]* self.weight_out11 + \
                                  decoded_3[j, 1, :][BackSFC1Up[j].long()] * self.weight_out11_0 + \
                                  decoded_3[j, 1, :][BackSFC1Down[j].long()] * self.weight_out11_1 + self.bias_out1

            decoded_sp[j, :, 1] = decoded_3[j, 2, :][BackSFC1[j].long()] * self.weight_out3 + \
                                  decoded_3[j, 2, :][BackSFC1Up[j].long()] * self.weight_out3_0 + \
                                  decoded_3[j, 2, :][BackSFC1Down[j].long()] * self.weight_out3_1 + \
                                  decoded_3[j, 3, :][BackSFC1[j].long()] * self.weight_out33 + \
                                  decoded_3[j, 3, :][BackSFC1Up[j].long()] * self.weight_out33_0 + \
                                  decoded_3[j, 3, :][BackSFC1Down[j].long()] * self.weight_out33_1 + self.bias_out2 
        
        # resort 1D to 2D
        decoded = F.tanh(decoded_sp) # both are BATCH_SIZE by nNodes by 2
        return encoded, decoded

########################################################################################
# train the autoencoder

t_train_0 = time.time()

autoencoder = CNN().to(device)
#autoencoder = torch.load("./UnstructuredNeigh128_200.pkl")
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

loss_list = []
loss_valid = []
for epoch in range(EPOCH):
    for step, x in enumerate(train_loader):
        #print("x", x.shape)
        b_y = x[:, :, 2:].to(device)
        b_x = x.to(device)
        #print("b_y",b_y.shape)

        encoded, decoded = autoencoder(b_x.float())

        loss = loss_func(decoded, b_y.float())  # mean square error
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients

    loss_list.append(loss)

    encoded, decoded = autoencoder(torch.tensor(valid_data).to(device))
    error_autoencoder = (decoded.detach() - torch.tensor(valid_data[:,:, 2:]).to(device))
    MSE_valid = (error_autoencoder**2).mean()
    loss_valid.append(MSE_valid)

    print('Epoch: ', epoch, '| train loss: %.6f' % loss.cpu().data.numpy(), '| valid loss: %.6f' % MSE_valid)

    # save the weights every 100 epochs 
    if (epoch%100 == 0):
        torch.save(autoencoder, 'OneNeigh128_'+str(epoch)+'.pkl')

t_train_1 = time.time()

########################################################################################
# save training and validation loss
losses_combined = np.zeros((EPOCH,2))
losses_combined[:,0] = np.asarray(loss_list)
losses_combined[:,1] = np.asarray(loss_valid)
np.savetxt('losses.csv', losses_combined , delimiter=',')

########################################################################################
# pass training, validation and test data through the autoencoder
t_predict_0 = time.time()

encoded, training_decoded = autoencoder.to(device)(torch.tensor(training_data).to(device))
#error_autoencoder = (training_decoded.cpu().detach().numpy() - training_data[:,:,3:5])
#print("MSE_err of training data", (error_autoencoder**2).mean())

encoded, valid_decoded = autoencoder.to(device)(torch.tensor(valid_data).to(device))
#error_autoencoder = (valid_decoded.cpu().detach().numpy() - valid_data[:, :, 3:5])
#print("Mse_err of validation data", (error_autoencoder**2).mean())

encoded, test_decoded = autoencoder.to(device)(torch.tensor(test_data).to(device))
#error_autoencoder = (test_decoded.cpu().detach().numpy() - test_data[:, :, 3:5])
#print("Mse_err of test data", (error_autoencoder**2).mean())

#print("Shape of training, valid, test after decoding", training_decoded.shape, valid_decoded.shape, test_decoded.shape)

########################################################################################
# rescale results
training_decoded[:, :, 0] = (training_decoded[:, :, 0] - bu)/ku
valid_decoded[:, :, 0] = (valid_decoded[:, :, 0] - bu)/ku
test_decoded[:, :, 0] = (test_decoded[:, :, 0] - bu)/ku

training_decoded[:, :, 1] = (training_decoded[:, :, 1] - bv)/kv
valid_decoded[:, :, 1] = (valid_decoded[:, :, 1] - bv)/kv
test_decoded[:, :, 1] = (test_decoded[:, :, 1] - bv)/kv

results = np.concatenate((training_decoded.cpu().data.numpy(), valid_decoded.cpu().data.numpy(), test_decoded.cpu().data.numpy()))
##results = np.concat(training_data, valid_data)
print('results shape', results.shape)
N = results.shape[1] * results.shape[2]
results = results.reshape((results.shape[0],N), order='F')
print('results shape', results.shape, type(results))

## write results to file
np.savetxt('results.csv', results , delimiter=',')

t_predict_1 = time.time()

print("Total time taken          :", t_predict_1 - t_read_0, "seconds")
print("time reading in           :", t_read_1    - t_read_0, "seconds")
print("time training autoencoder :", t_train_1   - t_train_0, "seconds")
print("time predicting           :", t_predict_1 - t_predict_0, "seconds")
print("Ending...")

#######################################################################################