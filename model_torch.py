# -*- coding:utf-8 -*-
# author : Han
# date : 2021/1/7 11:14
# IDE : PyCharm
# FILE : model_torch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#self.net
    # convnet.add(Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same',input_shape=input_shape))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    # convnet.add(MaxPooling1D(strides=2))
# encode
    # convnet.add(Flatten())
    # convnet.add(Dense(100,activation='sigmoid'))

# 自定义Flatten层 reshape Tensor
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, output):
        return output.view(output.size()[0], -1)

# L1Layer

class L1Layer(nn.Module):   #相似度，度量学习？
    def __init__(self, *args):
        super(L1Layer, self).__init__()
        
    def forward(self, output1, output2):
        return torch.abs(output1 - output2)


class SiameseNet(nn.Module):   #C_in 通道数
    def __init__(self, C_in):
        super(SiameseNet, self).__init__()
        
        # Dense layer
        self.fc1 = self.dense(64*4, 100)
        self.fc2 = self.dense(100, 1)
        self.L1_layer = L1Layer()
        self.dp = nn.Dropout(0.5)
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=64, stride=16,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            Reshape(),
            self.fc1
            
        )
        


#     def encode(self, x):
#         output = self.net(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc1(output)
#         return output

    def dense(self, in_features, out_features):
        # Dense layer
        fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
        )
        return fc

    def forward(self, input1, input2):     #孪生模型的输入
        output1 = self.net(input1)        #两个同样的模型
        output2 = self.net(input2)
        L1_distance = self.L1_layer(output1,output2)
        D1_layer = self.dp(L1_distance)
        prediction = self.fc2(D1_layer)

        return prediction

class WDCNN(nn.Module):

    def __init__(self, C_in = 2, nClass = 10):
        super(WDCNN, self).__init__()
        self.n_class = nClass
        self.input_channel = C_in

        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_channel, out_channels=16, kernel_size=64, stride=16, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Dense layer
        self.fc1 = nn.Sequential(
            nn.Linear(64*4, 100),
            nn.Sigmoid(),
        )

        self.layer =  nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(100, nClass),
            nn.Softmax(),
        )

    def encode(self, x):
        output = self.net(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output


    def forward(self, x):
        output = self.encode(x)
        output = self.layer(output)

        return output


# from torchsummary import summary
if __name__ == '__main__':
    net = WDCNN(2, 10)
    data_input = Variable(torch.randn([2, 2, 2048]))
    net(data_input)
    summary(net, input_size=(2,2048), device='cpu')

