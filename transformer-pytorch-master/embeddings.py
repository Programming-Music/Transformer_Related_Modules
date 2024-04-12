import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout_prob (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, num_embeddings, embedding_dim, dim, dropout_prob=0., padding_idx=0, max_len=5000):
        """
        
         :param num_embeddings: vocab_size词汇表大小
         :param embedding_dim: 词编码后的维度
         :param dim: 位置编码嵌入向量的维度
         :param padding_idx: 编码向量的非学习位置, defaults to 0
         :param max_len: pe的时间步数/最大序列长度, defaults to 5000
        """        
        super(PositionalEncoding, self).__init__()

            # (1)位置向量编码pe，为每个位置生成一个dim维的向量
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1) # max_len * 1
        div_term = torch.exp((torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim)).float()) # dim/2
        pe[:, 0::2] = torch.sin(position.float() * div_term)    # max_len * dim/2
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)    # 1 * max_len * dim
        """
            math.log()即自然对数
            广播规则, [5000,1]*[256] -> [5000,26] (广播[256]为[1, 256])
            [256]*[5000,1] -> [5000,26]. 根据展开规则从后向前遍历, 相等或有一方为1即可广播; 最后广播大小为每个维度的最大值
        """

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
            # num_embed/vocab_size市输入可能的种类数；embedding_dim是嵌入后的维度
        self.embbedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.weight = self.embbedding.weight
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim
        """
            self.embedding将词汇(词汇表大小num_embedding)映射成向量(inupt_size * em_dim), embedding[padding_idx] == 0
        """

    def forward(self, x, step=None):
        x = self.embbedding(x)  # x -> x[:, em_dim]
        x = x * math.sqrt(self.dim)
        if step is None:
            # 正常前向传播, 取前x.size(1)列进行处理
            x = x + self.pe[:, :x.size(1)]
        else:
            # 后向传播, 取pe中的step进行处理
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x