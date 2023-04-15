# MQRNN
Model of Paper: A Multi-Horizon Quantile Recurrent Forecaster

本仓库是对上述文章的MQRNN模型的复现，encoder、decoder以及损失函数都是严格按照文章的公式实现的，但没有添加文章中提出的初始线性层。该线性层用来处理不随时间变化的背景协变量，然后与样本数据进行拼接，之后导入Encoder中。模型结构如下：

1. Encoder是一个LSTM：input.shape 为 [seq_len,batch_size,1+covariate_size], output.shape 为 [seq_len,batch_size,hidden_size]

2. Globaldecoder是一个MLP：input.shape 为 [seq_len,batch_size,hidden_size], output.shape 为 [seq_len,batch_size,(horizon_size+1)*context_size]

3. Localdecoder是一个MLP：input.shape 为 [seq_len,batch_size,horizon_size,context_size*2+covariate_size], output.shape 为 [seq_len,batch_size,horizon_size,quantiles_size]

4. 损失函数即为文章中的损失函数：![image](https://user-images.githubusercontent.com/87350210/232199195-7fef146a-f8ac-4094-8b4f-f022aa232fcd.png)
然后对每个FCT、每个batch、每个horizon的每个quantile的损失进行求和。

# 文章内容简介
![MQRNN](https://user-images.githubusercontent.com/87350210/232200757-8a61109b-8e22-4f15-819d-b146f11e5fc7.png)
