import torch
import torch.nn as nn
import pandas as pd 
import numpy as np

class Encoder(nn.Module):
    #encoder，输入自变量和协变量的历史数据
    #input_size = covariate_size+1
    def __init__(self,
                 horizon_size:int,
                 covariate_size:int,
                 hidden_size:int,
                 dropout:int,
                 layer_size:int,
                 b_direction:bool,
                 device):
        super(Encoder,self).__init__()
        self.horizon_size =horizon_size
        self.covariate_size = covariate_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.b_direction = b_direction
        self.dropout = dropout
        self.LSTM = nn.LSTM(input_size= covariate_size+1,
                            hidden_size= hidden_size,
                            num_layers= layer_size,
                            dropout= dropout,
                            bidirectional= b_direction)
        #二维及以上的参数，进行正交矩阵初始化
        #一维参数使用正态分布初始化
        for param in self.LSTM.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    
    def forward(self,input):
        
        seq_len = input.shape[0]
        batch_size = input.shape[1]
        
        output,_ = self.LSTM(input)
        output = output.view(seq_len,batch_size,self.hidden_size)
        return output




                            