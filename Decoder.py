import torch
import torch.nn as nn

class GlobalDecoder(nn.Module):
    #globaldecoder，输入encoder的隐状态和未来协变量数据
    #生成horizon个horizon_specific_context和一个horizon_agnostic_context
    #input_size = hidden_size + horizon_size*covariate_size
    #output_size = (horizon_size + 1)*context_size
    def __init__(self,
                 hidden_size:int,
                 horizon_size:int,
                 covariate_size:int,
                 context_size:int):
        super(GlobalDecoder,self).__init__()
        self.hidden_size = hidden_size
        self.horizon_size = horizon_size
        self.covariate_size = covariate_size
        self.context_size = context_size

        self.linear1 = nn.Linear(in_features= hidden_size + covariate_size*horizon_size,
                                 out_features= covariate_size*horizon_size*3)

        self.linear2 = nn.Linear(in_features= covariate_size*horizon_size*3,
                                 out_features= covariate_size*horizon_size*2) 

        self.linear3 = nn.Linear(in_features= covariate_size*horizon_size*2,
                                 out_features= (horizon_size + 1)*context_size)
        
        self.activation = nn.ReLU()
    
    def forward(self, input):
        layer1_output = self.linear1(input)
        layer1_output = self.activation(layer1_output)

        layer2_output = self.linear2(layer1_output)
        layer2_output = self.activation(layer2_output)

        layer3_output = self.linear3(layer2_output)
        layer3_output = self.activation(layer3_output)
        return layer3_output


class LocalDecoder(nn.Module):
    #localdecoder，输入单个horizon的未来协变量数据、horizon_specific_context以及horizon_agnostic_context
    #输出该horizon每个quantile上的预测值
    #input_size = covariate_size + context_size*2
    #output_size = quantiles_size 
    def __init__(self,covariate_size:int,
                 context_size:int,
                 quantiles_size:int):
        super(LocalDecoder,self).__init__()
        self.covariate_size = covariate_size
        self.context_size = context_size
        self.quantiles_size = quantiles_size

        self.linear1 = nn.Linear(in_features= covariate_size + context_size*2,
                                 out_features= covariate_size*3)

        self.linear2 = nn.Linear(in_features= covariate_size*3,
                                 out_features= quantiles_size) 
        
        self.activation = nn.ReLU()

    def forward(self,input):
        layer1_output = self.linear1(input)
        layer1_output = self.activation(layer1_output)

        layer2_output = self.linear2(layer1_output)
        layer2_output = self.activation(layer2_output)
        return layer2_output