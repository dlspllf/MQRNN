import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import GlobalDecoder,LocalDecoder
from data import MQRNN_dataset

class MQRNN(nn.Module):
    def __init__(self,
                horizon_size:int,
                hidden_size:int,
                quantiles:list,
                dropout:float,
                layer_size:int,
                b_direction:bool,
                lr:float,
                batch_size:int,
                num_epochs:int,
                context_size:int,
                covariate_size:int,
                device):
        super(MQRNN,self).__init__()
        self.device = device
        self.horizon_size = horizon_size
        self.hidden_size = hidden_size
        quantiles_size = len(quantiles)
        self.quantiles_size = quantiles_size
        self.quantiles = quantiles
        self.dropout = dropout
        self.layer_size = layer_size
        self.b_direction = b_direction
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.context_size = context_size
        self.covariate_size = covariate_size
        
        #encoder，输入自变量和协变量的历史数据
        #input [seq_len,batch_size,1+covariate_size]
        #output [seq_len,batch_size,hidden_size]
        self.encoder = Encoder(horizon_size = horizon_size,
                               covariate_size = covariate_size,
                               hidden_size = hidden_size,
                               dropout = dropout,
                               layer_size = layer_size,
                               b_direction = b_direction,
                               device = device)
        
        #globaldecoder，输入encoder的隐状态和未来协变量数据
        #生成horizon个horizon_specific_context和一个horizon_agnostic_context
        #input [seq_len,batch_size,hidden_size]
        #output [seq_len,batch_size,context_size*(horizon_size+1)]
        self.gdecoder = GlobalDecoder(hidden_size = hidden_size,
                                      covariate_size = covariate_size,
                                      horizon_size = horizon_size,
                                      context_size = context_size)

        #localdecoder，输入单个horizon的未来协变量数据、horizon_specific_context以及horizon_agnostic_context
        #输出该horizon每个quantile上的预测值
        #input [seq_len,batch_size,horizon_size,context_size*2+covariate_size]
        #output [seq_len,batch_size,horizon_size,quantiles_size]
        self.ldecoder = LocalDecoder(context_size = context_size,
                                     covariate_size = covariate_size,
                                     quantiles_size = quantiles_size)
        self.encoder.double()
        self.gdecoder.double()
        self.ldecoder.double()
    
    def forward(self, cur_series_covariate_tensor : torch.Tensor, 
                      next_covariate_tensor: torch.Tensor,):
        LSTM_output = self.encoder(cur_series_covariate_tensor)
        hidden_and_covariate = torch.cat([LSTM_output, next_covariate_tensor], dim=2)
        Gdecoder_output = self.gdecoder(hidden_and_covariate)#[seq_len, batch_size, (horizon_size+1)*context_size]
        seq_len = Gdecoder_output.shape[1]
        # print(f"Gdecoder_output.shape: {Gdecoder_output.shape}")
        Gdecoder_output = Gdecoder_output.view(seq_len,self.batch_size,self.horizon_size+1,self.context_size)
        horizon_agnostic_context = Gdecoder_output[:,:,-1,:]
        horizon_specific_context = Gdecoder_output[:,:,:-1,:]
        horizon_agnostic_context = horizon_agnostic_context.repeat(1,1,self.horizon_size,1)
        next_covariate_tensor = next_covariate_tensor.view(seq_len,self.batch_size,self.horizon_size,self.covariate_size)
        Ldecoder_input = torch.cat([horizon_specific_context, next_covariate_tensor], dim=3)
        # print(f"horizon_agnostic_context.shape: {horizon_agnostic_context.shape}")
        # print(f"Ldecoder_input.shape: {Ldecoder_input.shape}")
        horizon_agnostic_context = horizon_agnostic_context.permute(1,0,2,3)
        Ldecoder_input = torch.cat([horizon_agnostic_context, Ldecoder_input],dim=3)#[seq_len, batch_size, horizon_size, 2*context_size+covariate_size]
        Ldecoder_output = self.ldecoder(Ldecoder_input)
        return Ldecoder_output

    def predict(self, train_target_df, train_covariate_df, test_covariate_df):
        #转成tensor
        input_target_tensor = torch.tensor(train_target_df)
        full_covariate = train_covariate_df
        full_covariate_tensor = torch.tensor(full_covariate)
        #未来协变量数据，从[horizon_size,covariate_size]变成[1,horizon_size * covariate_size]
        next_covariate = test_covariate_df
        next_covariate = next_covariate.reshape(-1, self.horizon_size * self.covariate_size)
        next_covariate_tensor = torch.tensor(next_covariate) #[1,horizon_size * covariate_size]

        input_target_tensor = input_target_tensor.to(self.device)#[seq_len,1]
        full_covariate_tensor = full_covariate_tensor.to(self.device)#[seq_len,covariate_size]
        next_covariate_tensor = next_covariate_tensor.to(self.device)#[1,horizon_size * covariate_size]

        with torch.no_grad():
            input_target_covariate_tensor = torch.cat([input_target_tensor, full_covariate_tensor], dim=1)
            input_target_covariate_tensor = torch.unsqueeze(input_target_covariate_tensor, dim= 0) #[1, seq_len, 1+covariate_size]
            input_target_covariate_tensor = input_target_covariate_tensor.permute(1,0,2) #[seq_len, 1, 1+covariate_size]

            outputs = self.encoder(input_target_covariate_tensor) #[seq_len,batch_size,hidden_size]

            hidden = torch.unsqueeze(outputs[-1],dim=0) #[1,1,hidden_size]
            next_covariate_tensor = torch.unsqueeze(next_covariate_tensor, dim=0) # [1,1, covariate_size * horizon_size]
            gdecoder_input = torch.cat([hidden, next_covariate_tensor], dim=2) #[1,1, hidden + covariate_size* horizon_size]

            gdecoder_output = self.gdecoder( gdecoder_input) #[1,1,(horizon_size+1)*context_size]
            
            seq_len = gdecoder_output.shape[0]
            gdecoder_output = gdecoder_output.view(seq_len,self.batch_size,self.horizon_size+1,self.context_size)
            horizon_agnostic_context = gdecoder_output[:,:,-1,:]
            horizon_specific_context = gdecoder_output[:,:,:-1,:]
            horizon_agnostic_context = horizon_agnostic_context.repeat(1,1,self.horizon_size,1)
            next_covariate_tensor = next_covariate_tensor.view(seq_len,self.batch_size,self.horizon_size,self.covariate_size)
            Ldecoder_input = torch.cat([horizon_specific_context, next_covariate_tensor], dim=3)
            Ldecoder_input = torch.cat([horizon_agnostic_context, Ldecoder_input],dim=3)#[seq_len, batch_size, horizon_size, 2*context_size+covariate_size]
            Ldecoder_output = self.ldecoder(Ldecoder_input)
            Ldecoder_output = Ldecoder_output.view(self.horizon_size,self.quantiles_size)
            output_array = Ldecoder_output.cpu().numpy()
            result_dict= {}
            for i in range(self.quantiles_size):
                result_dict[self.quantiles[i]] = output_array[:,i]
            print("prediction finished")
            return result_dict
