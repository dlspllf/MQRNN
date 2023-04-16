import torch
from Encoder import Encoder
from Decoder import GlobalDecoder, LocalDecoder
from data import MQRNN_dataset
from MQRNN import MQRNN
from torch.utils.data import DataLoader

def calc_loss(output_of_ldecoder:torch.Tensor,
              cur_real_vals_tensor:torch.Tensor,
              quantiles:list,
              device):
        #转换形状
        #[seq_len,batch_size,quantiles_size]
        quantiles_size = len(quantiles)
        total_loss = torch.tensor([0.0],device=device)
        cur_real_vals_tensor = cur_real_vals_tensor.permute(2,1,3,0)
        for i in range(quantiles_size):
            p = quantiles[i]
            errors = cur_real_vals_tensor - output_of_ldecoder[:,:,:,i]
            # cur_loss = torch.max((p-1)*errors,0)+torch.max(p*errors,0)#torch.max会把0认为是指定的维度，所以会返回一个元组
            cur_loss = torch.clamp((p-1)*errors, min=0) + torch.clamp(p*errors, min=0)#使用clamp函数限制tensor的取值范围，代替了torch.max
            total_loss += torch.sum(cur_loss)
        return total_loss

def train(net:MQRNN,
          dataset:MQRNN_dataset,
          lr:float,
          batch_size:int,
          num_epochs:int,
          quantiles:list,
          device):
    MQRNN_optimizer = torch.optim.Adam(net.parameters(),lr=lr)

    data_iter = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    
    for i in range(num_epochs):
        epoch_loss_sum = 0.0
        total_sample = 0
        for (cur_series_tensor, cur_covariate_tensor, cur_real_vals_tensor) in data_iter:
            batch_size = cur_series_tensor.shape[0]
            seq_len = cur_series_tensor.shape[1]
            horizon_size = cur_real_vals_tensor.shape[-1]
            total_sample += batch_size * seq_len * horizon_size
            MQRNN_optimizer.zero_grad()
            output = net(cur_series_tensor, cur_covariate_tensor)
            loss = calc_loss(output,cur_real_vals_tensor,quantiles,device)
            loss.backward()
            MQRNN_optimizer.step()
            epoch_loss_sum += loss.item()
        epoch_loss_mean = epoch_loss_sum/ total_sample
        if (i+1)%5 == 0:
            print(f"epoch_num {i+1}, current loss is: {epoch_loss_mean}")

    