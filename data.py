import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset

def read_df(config:dict):
    """
    读取数据
    """
    dateparse = lambda dates:pd.datetime.strptime(dates,'%Y%m%d')
    user_balance = pd.read_csv('C:/Users/Administrator/Desktop/科研/github/Data/Purchase Redemption Data/user_balance_table.csv',parse_dates=['report_date'],index_col=['report_date'],date_parser=dateparse)

    df = user_balance.groupby(['report_date'])['total_purchase_amt'].sum()
    #转成序列
    purchase_seq = pd.Series(df,name='value')

    #切分出训练集和测试集
    #horizon为30
    real_mean = np.mean(purchase_seq)
    real_std = np.std(purchase_seq)
    purchase_seq= (purchase_seq - np.mean(purchase_seq))/np.std(purchase_seq)
    purchase_seq_train = purchase_seq['2014-04-01':'2014-08-01']
    purchase_seq_test = purchase_seq['2014-08-02':'2014-08-31']

    purchase_seq_train.to_csv('./total_purchase_train.csv',header=True)
    purchase_seq_test.to_csv('./total_purchase_test.csv',header=True)

    horizon_size = config['horizon_size']
    time_range = pd.date_range('2014-04-01','2014-08-01',freq='D')
    covariate_df = pd.DataFrame(index=time_range,
                                data={'dayofweek':time_range.dayofweek,
                                      'month': time_range.month
                                })
    #日历数据（协变量）归一化
    for col in covariate_df.columns:
        covariate_df[col] = (covariate_df[col] - np.mean(covariate_df[col]))/np.std(covariate_df[col])
    
    train_target_df = purchase_seq_train.iloc[:-horizon_size]
    test_target_df = purchase_seq_train.iloc[-horizon_size:]
    train_covariate_df = covariate_df.iloc[:-horizon_size,:]
    test_covariate_df = covariate_df.iloc[-horizon_size:,:]

    return train_target_df, test_target_df, train_covariate_df, test_covariate_df, (real_mean,real_std)

class MQRNN_dataset(Dataset):
    #生成next_covariate，也就是读取当前序列信息之后，
    #在局部decoder上会用到的下一个序列的协变量信息，也就是未来协变量信息
    #所以未来协变量信息是从整个协变量信息序列里的第二个数据开始取的
    def __init__(self,
                 target_df:pd.DataFrame,#目标数据
                 covariate_df:pd.DataFrame, #协变量数据
                 horizon_size:int):#预测长度
        
        self.target_df = target_df
        self.covariate_df = covariate_df
        self.horizon_size = horizon_size

        #生成自变量的未来协变量数据next_covariate，也就是读取当前序列信息之后，
        #在局部decoder上会用到的下一个序列的协变量信息，也就是未来协变量信息
        #所以未来协变量信息是从整个协变量信息序列里的第二个数据开始取的
        full_covariate = []
        covariate_size = self.covariate_df.shape[1]
        print(f"self.covariate_df.shape[0] : {self.covariate_df.shape[0]}")
        for i in range(1, self.covariate_df.shape[0] - horizon_size+1):
            cur_covariate = []
            #for j in range(horizon_size):
            cur_covariate.append(self.covariate_df[i:i+horizon_size,:])
            full_covariate.append(cur_covariate)
        full_covariate = np.array(full_covariate)

        print(f"full_covariate shape: {full_covariate.shape}")
        full_covariate = full_covariate.reshape(-1, horizon_size * covariate_size)
        self.next_covariate = full_covariate

    def __len__(self):
        return self.target_df.shape[1]
    
    def __getitem__(self,idx):
        #当前序列和当前协变量
        cur_series = np.array(self.target_df[: -self.horizon_size])
        cur_covariate = np.array(self.covariate_df[:-self.horizon_size, :])

        #生成真实值序列，用于计算误差
        real_vals_list = []
        for i in range(1, self.horizon_size+1):
            real_vals_list.append(np.array(self.target_df[i: self.target_df.shape[0]-self.horizon_size+i]))
        
        real_vals_array = np.array(real_vals_list) #[horizon_size, seq_len]
        real_vals_array = real_vals_array.T #[seq_len, horizon_size]

        #将序列信息扩维
        #将序列信息和协变量在第1维上拼接,综合起来
        cur_series_tensor = torch.tensor(cur_series)
        cur_covariate_tensor = torch.tensor(cur_covariate) #[seq_len, covariate_size]
        cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor],dim=1)#[seq_len, 1+covariate_size]
        next_covariate_tensor = torch.tensor(self.next_covariate) #[seq_len, horizon_size * covariate_size]
        cur_real_vals_tensor = torch.tensor(real_vals_array)
        cur_series_covariate_tensor =torch.unsqueeze(cur_series_covariate_tensor,dim=0)
        next_covariate_tensor = torch.unsqueeze(next_covariate_tensor,dim=0)
        cur_real_vals_tensor = torch.unsqueeze(cur_real_vals_tensor,dim=0)
        return cur_series_covariate_tensor[idx], next_covariate_tensor[idx], cur_real_vals_tensor[idx]





                            


