import torch
from torch import nn


class TokenCrossEntropyLoss(nn.Module):

    def __init__(self, pad_index=0):
        super(TokenCrossEntropyLoss, self).__init__()

        self.pad_index = pad_index
        self.base_loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_index)

    def forward(self, outputs, targets):
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs_flat = outputs.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)

        batch_loss = self.base_loss_function(outputs_flat, targets_flat)

            # count记录有效标记（非padding）的个数, item()将tensor转为python数值
        count = (targets != self.pad_index).sum().item()

        return batch_loss, count


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, vocabulary_size, pad_index=0):
        assert 0.0 < label_smoothing <= 1.0

        super(LabelSmoothingLoss, self).__init__()

        self.pad_index = pad_index
            # 对最后一维向量进行操作
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='sum')

        smoothing_value = label_smoothing / (vocabulary_size - 2)  # exclude pad and true label
            # full/fill ope, by specific dimension and value
        smoothed_targets = torch.full((vocabulary_size,), smoothing_value)
        smoothed_targets[self.pad_index] = 0
            # 基于Module类提供方法注册一个buffer缓冲区，方便在后续loss计算中直接使用
        self.register_buffer('smoothed_targets', smoothed_targets.unsqueeze(0))  # (1, vocabulary_size)

        self.confidence = 1.0 - label_smoothing

    def forward(self, outputs, targets):
        """
        outputs (FloatTensor): (batch_size, seq_len, vocabulary_size)
        targets (LongTensor): (batch_size, seq_len)
        """
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs_log_softmax = self.log_softmax(outputs)
        outputs_flat = outputs_log_softmax.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)

            # repeat(a, b), 在第一个维度重复a次, 在第二个维度重复b次
        smoothed_targets = self.smoothed_targets.repeat(targets_flat.size(0), 1)
        # smoothed_targets: (vocab_size) -> (batch_size * seq_len, vocab_size)

            # scatter_(dim,index,source)按照索引,将source插入到向量的dim维度中。
            # eg. scatter_(1, torch.zeros((5, 6)), torch.arange(5)) 结果为对角矩阵
            # 即原始的target(batch_size * seq_len)其中部分为1，平滑后(b_s * seq, vocab)在每行(b_s*seq)寻找之前的target
        smoothed_targets.scatter_(1, targets_flat.unsqueeze(1), self.confidence)

            # masked_fill_(index,value)，对pad_index的地方赋零值
        smoothed_targets.masked_fill_((targets_flat == self.pad_index).unsqueeze(1), 0)

        loss = self.criterion(outputs_flat, smoothed_targets)
        count = (targets != self.pad_index).sum().item()

        return loss, count
