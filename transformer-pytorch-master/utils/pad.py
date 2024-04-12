import torch
import numpy as np

PAD_TOKEN_INDEX = 0


def pad_masking(x, target_len):
    # x: (batch_size, seq_len)
    batch_size, seq_len = x.size()
    padded_positions = x == PAD_TOKEN_INDEX  # (batch_size, seq_len)
        # seq_len是源域中字符/单词的个数；target_len是目标域中字符/单词的个数。
    pad_mask = padded_positions.unsqueeze(1).expand(batch_size, target_len, seq_len)
    return pad_mask


def subsequent_masking(x):
    # x: (batch_size, seq_len - 1)
    batch_size, seq_len = x.size()
        # np.triu 取矩阵的上半部分,k=1即向上偏移
    subsequent_mask = np.triu(np.ones(shape=(seq_len, seq_len)), k=1).astype('uint8')
    subsequent_mask = torch.tensor(subsequent_mask).to(x.device)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
    return subsequent_mask