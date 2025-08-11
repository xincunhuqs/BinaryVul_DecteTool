import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import os
try:
    from DataPreprocessing import DefactcodeDataset,count_max_seq_len
    from LocalTokenizer import loadtokenizer, word2index,vocabularygenerate
except:
    from TransformerModel.DataPreprocessing import DefactcodeDataset, count_max_seq_len
    from TransformerModel.LocalTokenizer import loadtokenizer, word2index, vocabularygenerate

"""
# @Time    : 2025/07/17
# @Author  : huqinsong
# @Version: 本地分词器
# @Desc: 主要实现：
        利用Transformer深度学习模型，完成漏洞特征提取

参考：
"""

CUDA_FLAGE = torch.cuda.is_available()
# vocabularygenerate(r"TransformerModel\data\train.txt")
tokenizer_dict = loadtokenizer(r"./tokenize_dict.txt")
try:
    max_seq_length = count_max_seq_len(r"./data/train.txt")
    print(max_seq_length)
except:
    max_seq_length = count_max_seq_len(r"./TransformerModel/data/train.txt")
    # print("当前序列最大长度为:",max_seq_length)

src_len =tgt_len=max_seq_length
tgt_vocab_size=src_vocab_size = len(tokenizer_dict.keys())


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=max_seq_length): #max_len可以定位预测句子的最大长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  #随机失活，防止过拟合
        pe = torch.zeros(max_len, d_model)  #初始化一个矩阵保存计算的位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print("test",pe.shape,pe.unsqueeze(0).shape, pe.unsqueeze(0).transpose(0, 1).shape)
        pe = pe.unsqueeze(0).transpose(0, 1)  #交换矩阵的位置，并且在第0维添加一个维度，矩阵的维度变化：[max_len,d_model]-->[1,max_len,d_model]-->[max_len,1,d_model]
        self.register_buffer('pe', pe) #位置编码注册到模型的缓冲区：在加载模型的时候会被使用，但是反向传播的时候被忽略

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        # print("位置编码的计算",x.shape,self.pe[:x.size(0), :].shape,x.size(0))
        x = x + self.pe[:x.size(0), :]
        # print("位置编码计算完毕")
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):  #mask P
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    # print("huqinsong",pad_attn_mask,pad_attn_mask.shape)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # 扩展向量[batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    # print("进入decoder计算句子源码函数")
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]  #创建一个三维的矩阵
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix   使用np的truip函数创建一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  #将上三角矩阵转换为Tensor向量表示，并且非0即1
    # print("上三角计算结束")
    return subsequence_mask # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''

        #结合q和k的维度，发现q和k不能直接做点积，根据q和k的矩阵关系，将最后两个元素使用transpose函数进行位置交换即可解决
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        """
        实例：
        scores = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
        attn_mask = torch.tensor([[True, False, True],
                                 [False, True, False]])
        scores.masked_fill_(attn_mask, -1e9)
        结果
        tensor([[-1e9,  2.0, -1e9],
        [ 4.0, -1e9,  6.0]])  
        """

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new（d_k * n_heads）) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k] view函数中参数-1表示该位置的维度自动计算
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v]-->[batch_size, len_q, n_heads, d_v],
        # attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)   #根据q,k,v，mask计算自注意力
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        if CUDA_FLAGE:
            return nn.LayerNorm(d_model).cuda()(output + residual), attn   #返回的第一个参数实现残差连接
        else:
            return nn.LayerNorm(d_model)(output + residual), attn 


class PoswiseFeedForwardNet(nn.Module):
    """
    通过两个线性层变换实现特征提取
    """

    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False), #经过线性层变换将d_model转换为2048维度
            nn.ReLU(),                            #经过激活函数处理
            nn.Linear(d_ff, d_model, bias=False) #再经过线性层变换将2048维转换为d_model
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        if CUDA_FLAGE:
            return nn.LayerNorm(d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]   实现残差连接并且实现归一化
        else:
            return nn.LayerNorm(d_model)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V  这里之所以要传入三次enc_inpus，而不是写一次enc_inpus:是为了便于计算交叉多头注意力机制，传入三次参数后，计算多头注意力机制可以实现代码复用
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    """
    Transformer中Encoder模块的实现
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  #定义encoder中词向量层：src_vocab_sizeba-词表大小，d_model-每个词对应的向量大小
        self.pos_emb = PositionalEncoding(d_model)   #位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  #Transform中一个Encoder模块包括多个Encoder层，所以使用一个for循环将多个EncoderLayer实例保存到模型列表中

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # print("encoder的输入为：",enc_inputs,enc_inputs.shape)
        word_ebm = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        # print("输入序列被词向量后的结果",word_ebm)
        enc_outputs = self.pos_emb(word_ebm) # [batch_size, src_len, d_model]
        # print("huqinsong",enc_outputs.shape)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) #将输入sentence中pad进行标记，识别填充词pad  [batch_size, src_len, src_len]
        enc_self_attns = []  #创建一个列表用于保存attention的值
        for layer in self.layers: #这部分实际上就是Encoder多头注意力机制的实现
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask) #计算注意力的时候需要将Encoder端的输入向量结合位置编码后的向量，输入句子的mask表示
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)  #Decoder模块中的第一次多头注意力机制的计算
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask) #Decoder模块中第二次交叉多头注意力机制的计算
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model] 进行线性变化，提取特征
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)  #位置编码函数中就实现了将词向量和位置编码进行融合
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        if CUDA_FLAGE:
            dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda() # [batch_size, tgt_len, d_model]
            dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]  对pad填充词进行mask
            dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
            dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda() # [batch_size, tgt_len, tgt_len]  将pad掩码和句子掩码相加得到一个新的矩阵，并且使用gt函数大于0返回Ture，这里就实现屏蔽pad填充词，又可以将隐藏后文

        else:
            dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_model]
            dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]  对pad填充词进行mask
            dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs) # [batch_size, tgt_len, tgt_len]
            dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0) # [batch_size, tgt_len, tgt_len]  将pad掩码和句子掩码相加得到一个新的矩阵，并且使用gt函数大于0返回Ture，这里就实现屏蔽pad填充词，又可以将隐藏后文
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]  #在交叉注意力机制中会被使用

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        if CUDA_FLAGE:
            self.encoder = Encoder().cuda()  #输入一个句子，经过encoder处理后的维度为512
            self.decoder = Decoder().cuda()  #结合Encoder和Decoder的输出还是一个512维的向量
            self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()  #但是我们需要Decode的输出转换为目标词表的长度，这样才方便用计算目标词表中每个词出现的概率
        else:
            self.encoder = Encoder()  #输入一个句子，经过encoder处理后的维度为512
            self.decoder = Decoder()  #结合Encoder和Decoder的输出还是一个512维的向量
            self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)  
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len] ——表示encode的输入维度为：（句子数，句子向量维度）
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns






# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

if CUDA_FLAGE:
    model = Transformer().cuda()
else:
    model=Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0) 
optimizer = optim.SGD(model.parameters(), lr=0.008, momentum=0.75) #优先
# optimizer = optim.Adadelta(model.parameters(), lr=0.0085)
# optimizer = optim.Adam(model.parameters(),lr=0.008)
# optimizer=optim.Adagrad(model.parameters(), lr=0.008)
# optimizer=optim.ASGD(model.parameters(), lr=0.002)

def train():
    model.train()
    tokenizerdict_path = r"./tokenize_dict.txt"
    loader = Data.DataLoader(DefactcodeDataset(tokenizerdict_path,\
                r"data/train.txt", max_seq_length), 15, True)
    accuracy_list=[]
    loss_list=[]
    batch_index_list=[]
    for epoch in range(12):
        epoch_list=[]
        print(f"训练轮数：{epoch+1}")
        correct_time = total_time = 0
        for batch_index,(enc_inputs, dec_inputs, dec_outputs) in enumerate(loader):
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            if torch.cuda.is_available():
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            else:
                enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            # print(f"encoder的输入为：",enc_inputs)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            print(f"模型预测的结果为： {outputs},型号为：{outputs.shape}")
            print("原始输出的结果形状为：",outputs.shape)
            number_nonzreo = 1 #torch.nonzero(dec_outputs).size(0)
            preds=outputs.argmax(dim=-1)
            print("预测结果的维度为：",preds.shape)
            print(f"预测结果为：{preds}")
            correct_time += (preds[:number_nonzreo] == dec_outputs.view(-1)[:number_nonzreo]).sum().item()  # 统计预测正确的token数
            # print(f"模型预测的结果：{preds[:number_nonzreo]}，数据集中真实结果：{dec_outputs.view(-1)[:number_nonzreo].sum().item()}")
            total_time += 1
            print("预测结果的完整输出为：",preds)
            print("预测值的结果为：", preds[:number_nonzreo])
            print("decoder的输出结果为：",dec_outputs)
            print(f"正确数：{correct_time}，总数：{total_time}")
            print("\n\n")
            loss = criterion(outputs, dec_outputs.view(-1))
            optimizer.zero_grad()  #梯度清零
            loss.backward() #反向传播
            optimizer.step()  #参数更新
            if batch_index%1 ==0:
                print('Epoch:', '%d' % (epoch + 1), "batch index:",f"{batch_index+1}/{len(loader)}", 'loss =', '{:.6f}'.format(loss),"acc:",(correct_time/total_time)*100,"%")
                accuracy_list.append(correct_time/total_time)
                loss_list.append(loss.item())
                batch_index_list.append(batch_index)
    if not os.path.exists(r"./local_model"):
        os.mkdir(r"./local_model")
    time_flage=time.strftime("%Y%m%d%H%M%S")
    torch.save(model.state_dict(), rf"./local_model/transformer{time_flage}.pth")
    print("成功保存了训练的深度学习模型")



def greedy_decoder(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    finishes = False
    next_symbol = start_symbol
    eos_id = tokenizer_dict["E"]  # 假设E对应的id正确
    while not finishes:
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).cuda()], -1) if CUDA_FLAGE else \
        torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word

        # 修改条件判断：直接比较数值
        if next_symbol.item() == eos_id:
            finishes = True

        # 防止无限循环：添加最大长度限制
        if dec_input.size(1) > 3:  # 例如最大生成长度100
            finishes = True
    return dec_input


def test(model_path):
    # Test
    tokenizerdict_path = r"tokenize_dict.txt"
    loader = Data.DataLoader(DefactcodeDataset(tokenizerdict_path, "./data/test.txt", max_seq_length), 20, True)
    model.load_state_dict(torch.load(model_path,weights_only=False))
    model.eval()
    enc_inputs, _, _ = next(iter(loader))
    enc_inputs = enc_inputs.cuda()
    # print(f"正在构造贪婪输入序列..")
    for i in range(len(enc_inputs)):
        greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=torch.Tensor([[tokenizer_dict["S"]]]).to(torch.int64))
        predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]
        print(enc_inputs[i], '->', [list(tokenizer_dict)[n.item()-1] for n in predict.squeeze()])




def defactcode_detect(model_path,defactcode_setction,verbose):
    """
    对传入的defactcode_setction进行检测，输出该缺陷代码的漏洞类型和概率
    model_path：深度学习模型文件
    defactcode_setction：待检测的代码块
    verbose：是否输出检测过程中的详细信息
    """
    model=Transformer().to("cuda") if CUDA_FLAGE else  Transformer()
    if CUDA_FLAGE:
        model.load_state_dict(torch.load(model_path,weights_only=False))
    else:
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    defactcodeindex_section=torch.LongTensor(word2index(defactcode_setction,tokenizer_dict,max_seq_length)).to("cuda") if CUDA_FLAGE else \
                torch.LongTensor(word2index(defactcode_setction,tokenizer_dict,max_seq_length))
    greedy_dec_input=greedy_decoder(model,defactcodeindex_section.view(1,-1),start_symbol=torch.Tensor([[tokenizer_dict["S"]]]).to(torch.int64))
    predict, _, _, _=model(defactcodeindex_section.view(1,-1), greedy_dec_input)
    predict = predict.data.max(1,keepdim=True)[1]
    # print("kkkkk",predict.squeeze().shape,predict.squeeze().shape == (2,))
    if predict.squeeze().shape == (2,):
        vultype_results = " ".join([list(tokenizer_dict)[n.item()] for n in predict.squeeze()][:1])
        if verbose:
            print(defactcode_setction,"预测的结果为：",vultype_results)
        return defactcode_setction, vultype_results


def test_train():
    model.load_state_dict(torch.load(r"./local_model/transformer.pth"))
    model.train()
    tokenizerdict_path = r"./tokenize_dict.txt"
    loader = Data.DataLoader(DefactcodeDataset(tokenizerdict_path,\
                r"./data/train.txt", max_seq_length), 30, True)
    accuracy_list=[]
    loss_list=[]
    batch_index_list=[]
    for epoch in range(6): 
        epoch_list=[]
        print(f"训练轮数：{epoch+1}")
        correct_time = total_time = 0
        for batch_index,(enc_inputs, dec_inputs, dec_outputs) in enumerate(loader):
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            # print(f"encoder的输入为：",enc_inputs)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            print(f"模型预测的结果为： {outputs}")
            print("原始输出的结果形状为：",outputs.shape)
            number_nonzreo = 1 #torch.nonzero(dec_outputs).size(0)
            preds=outputs.argmax(dim=-1)
            print("预测结果的维度为：",preds.shape)
            print(f"预测结果为：{preds}")
            # print("decoder的输出结果为：",dec_outputs.view(-1)[number_nonzreo])
            # print("decoder的输出维度为：",dec_outputs.view(-1).shape)
            # print("非0元素的个数：",number_nonzreo,type(number_nonzreo))
            correct_time += (preds[:number_nonzreo] == dec_outputs.view(-1)[:number_nonzreo]).sum().item()  # 统计预测正确的token数
            # print(f"模型预测的结果：{preds[:number_nonzreo]}，数据集中真实结果：{dec_outputs.view(-1)[:number_nonzreo].sum().item()}")
            total_time += 1
            print("预测结果的完整输出为：",preds)
            print("预测值的结果为：", preds[:number_nonzreo])
            # print("encoder的输入内容为：",enc_inputs)
            # print("decoder的输入结果为：",dec_inputs)
            print("decoder的输出结果为：",dec_outputs)
            # print("预测的结果切片：",preds[:number_nonzreo])
            # print("decoder的输出结果切片",dec_outputs.view(-1)[:number_nonzreo])
            print(f"正确数：{correct_time}，总数：{total_time}")
            print("\n\n")
            loss = criterion(outputs, dec_outputs.view(-1))
            optimizer.zero_grad()  #梯度清零
            loss.backward() #反向传播
            optimizer.step()  #参数更新
            if batch_index%1 ==0:
                print('Epoch:', '%d' % (epoch + 1), "batch index:",f"{batch_index+1}/{len(loader)}", 'loss =', '{:.6f}'.format(loss),"acc:",(correct_time/total_time)*100,"%")
                accuracy_list.append(correct_time/total_time)
                loss_list.append(loss.item())
                batch_index_list.append(batch_index)
    with open(r"DetectMModel\TransformerModel\temp_test\accurity_file", "w", encoding="utf-8") as f:
        f.write(str(accuracy_list)+"\n"+str(batch_index_list))
    if not os.path.exists(r"DetectMModel\TransformerModel\local_model"):
        os.mkdir(r"DetectMModel\TransformerModel\local_model")
    torch.save(model.state_dict(), r"DetectMModel\TransformerModel\local_model\transformer1.pth")
    print("成功保存了训练的深度学习模型")


if __name__ == '__main__':
    # print("huqinsong")
    train()
    # test_train()
    # test("./local_model/transformer.pth")
    # test="ptr [0x41e008];test eax, eax;jne 0x401b1d;push 0x426194;call 0x401000;add esp, 4;push 1;call 0x40dca0"
    # # #mov eax, 1;test eax, eax;je 0x401941;mov dword ptr [ebp - 4], 0x423088 ! S CWE272_Least_Privilege_Violation ! CWE272_Least_Privilege_Violation E
    # #
    # defactcode_detect("local_model/transformer.pth", test)

    # defactcode_setction="mov ax, word ptr [0x423a70];mov word ptr [ebp - 0xfc], ax;push 0xc6;push 0;lea ecx, [ebp - 0xfa];push ecx;call 0x402e70;add esp, 0xc;lea edx, [ebp - 0xfc];mov dword ptr [ebp - 0x1d8], edx"    
    # defactcode_detect(r"TransformerModel\local_model\transformer.pth",defactcode_setction)
