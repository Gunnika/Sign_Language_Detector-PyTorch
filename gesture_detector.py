#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import string
import os
import cv2
from glob import glob

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.autograd import Variable
import torchvision.transforms as transforms

from barbar import Bar


# In[2]:


train_data_raw = np.array(glob('asl-alphabet/asl_alphabet_train/asl_alphabet_train/*/*'))
test_data_raw = np.array(glob('asl-alphabet/asl_alphabet_test/asl_alphabet_test/*'))


# In[3]:


print('There are %d total train images.' % len(train_data_raw))
print('There are %d total test images.' % len(test_data_raw)) #No test image for delete


# In[4]:


from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler


# In[5]:


# percentage of training set to use as validation
valid_size = 0.2

train_transform = transforms.Compose([ transforms.Grayscale(num_output_channels=1),
                                transforms.Resize(size=(50,50)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
valid_transform = transforms.Compose([ transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize(50),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
test_transform = transforms.Compose([ transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize(size=(50,50)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])

train_data = datasets.ImageFolder(root = 'asl-alphabet/asl_alphabet_train/asl_alphabet_train', transform=train_transform)
test_data = datasets.ImageFolder(root='asl-alphabet/asl_alphabet_test', transform=test_transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# print number of images in each dataset
print('There are %d total train images.' % len(indices[split:]))
print('There are %d total dog validation images.' % len(indices[:split]))
print('There are %d total test images.' % len(test_data))


trainloader = torch.utils.data.DataLoader(train_data, batch_size=20,sampler=train_sampler)
validloader = torch.utils.data.DataLoader(train_data, batch_size=20,sampler=valid_sampler)
testloader = torch.utils.data.DataLoader(test_data,batch_size=20, shuffle=False)


loaders = dict(train=trainloader,
                       valid = validloader,
                       test=testloader)


# In[6]:


dim=50


# In[7]:


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,20,3)
        self.conv3 = nn.Conv2d(20,30,3)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.2)
        
        self.fc1 = nn.Linear(2430, 270)
        self.fc2 = nn.Linear(270,29)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.softmax(F.relu(self.fc2(x)))
        return(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[8]:


use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    model = Network().cuda()
else:
    model = Network()


# In[9]:


summary(model, (1,dim,dim)) #takes the model and the input tensor shape, displays the output shape


# In[10]:


epochs = 50
learning_rate = 0.001


# In[11]:


optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.007)
criterion = nn.CrossEntropyLoss()


# In[12]:


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        print('Epoch: {} '.format(
        epoch
        ))
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(Bar(loaders['train'])):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + ((1/(batch_idx +1))*(loss.data - train_loss))
            
#             if batch_idx % 1000 == 0:
#                 print('Epoch %d, Batch %d loss: %.6f' %(epoch, batch_idx + 1, train_loss))
            
#         print('Training Loss: {:.6f} '.format(
#         train_loss
#         ))
                
    # return trained model
#     return model
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output,target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))


#       print training/validation statistics 
        print('  Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format( 
            train_loss,
            valid_loss
            ))
        
        ## save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model


# In[13]:


# train the model
model_scratch = train(epochs, loaders, model, optimizer, criterion, use_cuda, 'saved_model.pt')


# In[14]:


# load the model that got the best validation accuracy 
model_scratch.load_state_dict(torch.load('saved_model.pt'))


# In[15]:


def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * float(correct / total), correct, total))

# call test function    
test(loaders, model_scratch, criterion, use_cuda)


# In[16]:


dict_labels = {
    0:'A',
    1:'B',
    2:'C',
    3:'D',
    4:'E',
    5:'F',
    6:'G',
    7:'H',
    8:'I',
    9:'J',
    10:'K',
    11:'L',
    12:'M',
    13:'N',
    14:'O',
    15:'P',
    16:'Q',
    17:'R',
    18:'S',
    19:'T',
    20:'U',
    21:'V',
    22:'W',
    23:'X',
    24:'Y',
    25:'Z',
    26:'del',
    27:'nothing',
    28:'space'
    
}


# In[17]:


def predict(img_path):
    # load the image and return the predicted breed
    img = Image.open(img_path)
#     img = Image.open(img_path).convert('L')
    transformations = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            transforms.Resize(size=50),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5],[0.5])])
    image_tensor = transformations(img)[:3,:,:].unsqueeze(0)

    # move model inputs to cuda, if GPU available
    if use_cuda:
        image_tensor = image_tensor.cuda()

    # get sample outputs
    output = model_scratch(image_tensor)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)

    pred = np.squeeze(preds_tensor.numpy()[0]) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy()[0])

    return dict_labels[pred]


# In[18]:


# plt.figure(figsize=(10,8))
# plt.plot(loss_log[2:])
# plt.plot(acc_log)
# plt.plot(np.ones(len(acc_log)), linestyle='dashed')
# plt.show()


# In[25]:


prediction = predict('Inference_Images/c.jpg')
lab = 'c'


# In[26]:


print("Prediction: {}".format(prediction))
print("Actual Label: {}".format(lab))


# In[21]:


# pixels = cv2.imread('./c.jpg').reshape(28, 28)
# plt.subplot(223)
# sns.heatmap(data=pixels)
# lab = 'c'
# test_sample = torch.FloatTensor([pixels.reshape(1, 28, 28).tolist()])
# pred = model(Variable(input_img))
# print("Prediction: {}".format(alph[torch.max(net_out_sample.data, 1)[1].numpy()[0]]))
# print("Actual Label: {}".format(lab))

