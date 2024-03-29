# 数据组织类

#-*-coding:utf-8-*-

import re
import unicodedata

import torch
from torch.autograd import Variable

import os

import sys
sys.path.append(os.path.abspath('..'))
from parameters import Parameters
param = Parameters()


# 字符统计类
class Lang:
    """
    Lang(uage)，语言——字符统计类
    """    
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        # self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.index2word = {0: '<blank>', 1: '<unk>', 2: '<s>', 3: '</s>'}
        self.n_words = 4  # 初始字典有4个字符，{空白，未知，句首，句尾}
        self.seq_max_len = 0

    def index_word(self, word):
        """
        在index2word、word2index和word2count等三个字典中新增word，或增加word键所在的值
        param word: 序列中的单词
        """        
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def index_words(self, sentence):
        """
        依次处理sentence中的单词，将其放入Lang的字典中
        param sentence: 包含单词的序列
        """        
        for word in sentence.split(' '):
            self.index_word(word)

    # Remove words below a certain count threshold
    def trim(self, min_count):
        """
        仅保存单词频率大于min_count的单词，并更新Lang(self)的三个字典
         :param min_count: 字典中保留单词的最低频率
        """         
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('    保留单词数 %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        # self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.index2word = {0: '<blank>', 1: '<unk>', 2: '<s>', 3: '</s>'}
        self.n_words = 4  # Count default tokens

        for word in keep_words:
            self.index_word(word)


# 数据准备类
class DataProcess(object):

    def __init__(self, train_src, train_tgt, val_src, val_tgt, test_src, test_tgt):
        self.train_src = train_src
        self.train_tgt = train_tgt

        self.val_src = val_src
        self.val_tgt = val_tgt

        self.test_src = test_src
        self.test_tgt = test_tgt

        # 源语言序列添加（start/end of sequence）后的固定最大长度。
        self.src_max_len = 2  
        self.tgt_max_len = 2  

    def unicode_to_ascii(self, s):
        """
        字符串编码预处理，消除编码差异
        """
        # 'NFD'，Normalization Form Decompose; 将组合字符分解为基本字符+修饰字符，然后再统一取出
        # 字符类别：如'Lu'(Letter, Uppercase), 'Nd'(Number,Decimal Digit, 数字), "Mn"（Mark,Non-Spacing, 标记字母）
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    
    def normalize_string(self, s):
        """
        格式化字符串，包括小写化、包括目标字符、消除多余空格
         :param s: eg. Two young, White males are outside near many bushes.
        return: eg. two young , white males are outside near many bushes .
        """        
        s = self.unicode_to_ascii(s.lower().strip())  # strip剔除首尾空格
        s = re.sub(r"([,.!?])", r" \1 ", s)  # 标签符号包裹在空格中
        s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)  # 将非目标字符处理为空格
        s = re.sub(r"\s+", r" ", s).strip()  # \s+表示至少一个空格，sub进行替换；
        return s

    def filter_pairs(self, pairs):
        """
        检查句子其有效长度是否在[param.min_len, param.max_len]中([0, 100])
         :param pairs(list): [seq1, seq2, ...]
        return: [seq1, seq2, ...]
        """        
        filtered_pairs = []
        for pair in pairs:
            sentence_num = 0
            for i in pair:
                if len(i.split(' ')) > param.min_len and len(i.split(' ')) <= param.max_len:
                    sentence_num += 1
            if sentence_num == len(pair):
                filtered_pairs.append(pair)
            else:
                temp = [len(i.split(' ')) for i in pair]
                print(temp, pair)

        return filtered_pairs

    def indexes_from_sentence(self, lang, sentence):
        """
        基于src/tgt的lang类，得到sentence中单词对应的索引列表
        param lang: src/tgt统计单词的类
        param sentence: 句子
        return: 句首索引 + 句内单词索引 + 句末索引
        """        
        # 前后加上sos和eos。注意句子的句号也要加上;
        # get(key, default)如果这个词没有出现在词典中（已经去除次数小于限定的词），以unk填充
        return [param.sos] + [lang.word2index.get(word, param.unk) for word in sentence.split(' ')] + [param.eos]

    def read_file(self, data):
        """
        输入文件路径，返回句子列表
         :param data: train/val/test_src, 
        return: 经过统一编码、过滤长度的句子列表 [seq1, seq2, ]
        """        
        # open返回文件obj, read()获取文件流
        content = open(data, encoding='utf-8').read().split('\n')  # 读取文件并处理
            # content: list of seq(str) 
        tmp = [self.normalize_string(s) for s in content]
        
        content = self.filter_pairs([self.normalize_string(s) for s in content])  # 规范化字符，并限制长度
        return content

    def get_src_tgt_data(self):
        """
        Input:
            train/val/test中的src, from _init_函数
        Exec:
            创建源语言Lang和目标语言Lang对象(包含word2index, index2word等属性). 
            用于建立词汇表(如 4: 'two', t: 'young')
        Return:
            src_tgt:源-目标句对列表
            src_long:源语言词汇表
            tgt_lang:目标语言词汇表
        """
        src_content = []
        for i in (self.train_src, self.val_src, self.test_src):
            src_content += self.read_file(i)
        src_lang = Lang('src')  # source字符类

        tgt_content = []
        for j in (self.train_tgt, self.val_tgt, self.test_tgt):
            tgt_content += self.read_file(j)
        tgt_lang = Lang('tgt')  # target字符类

        src_tgt = []  # 存储source和target序列
        for line in range(len(src_content)):  # 检索单词
            src_lang.index_words(src_content[line])
            tgt_lang.index_words(tgt_content[line])
            src_tgt.append([src_content[line], tgt_content[line]])
        
            # print(len(src_content), len(tgt_content), len(src_tgt))
        # 修剪单词表，少于限定次数将被删除
        print("修剪源域单词")
        src_lang.trim(param.min_word_count)
        print("修剪目标域单词")
        tgt_lang.trim(param.min_word_count)

        # 前后两个字符，加上最大序列的长度
        self.src_max_len += max([len(s[0].split(' ')) for s in src_tgt])  # 全部输入序列的最大长度
        self.tgt_max_len += max([len(s[1].split(' ')) for s in src_tgt])  # 全部目标序列的最大长度

            # print(f"N of seq is {len(tgt_content)}, total len of max {self.src_max_len}, total len of max {self.tgt_max_len}")
        return src_tgt, src_lang, tgt_lang

    def word_2_index(self, mode, src_lang, tgt_lang):
        """
        Input:
            mode:'t/v/t', lang of src or tgt
        Exec:
            get src/tgt seq by mode.
            get the index of src/tgt seq in src/tgt lang.
        Return:
            索引列表
            src_list, tgt_list, src_tgt_list(N*2*L_seq')
            
        """
        src = 0
        tgt = 0
        if mode == 'train':
            src = self.train_src
            tgt = self.train_tgt
        elif mode == 'val':
            src = self.val_src
            tgt = self.val_tgt
        elif mode == 'test':
            src = self.test_src
            tgt = self.test_tgt

        src_seq = self.read_file(src)
        tgt_seq = self.read_file(tgt)

        # 判断输入和目标序列的句子数量是否对应，这里或许可以使用assert断言
        if len(src_seq) != len(tgt_seq):
            print('输入与目标句子不对应！！！')
            exit()

        # 以list存储批序列
        src_list = []
        tgt_list = []
        src_tgt_list = []

        for i in range(len(src_seq)):  # 序列字符token转化为索引token
            # 即处理每个一句子，返回其在***_lang中的index列表
            src_list.append(self.indexes_from_sentence(src_lang, src_seq[i]))
            tgt_list.append(self.indexes_from_sentence(tgt_lang, tgt_seq[i]))
                # if i == 0:
                #     print(src_seq[i], src_list[-1], tgt_list[-1])
                # 将src_list和tgt_list以str的形式，存入src_tgt_list中
            src_tgt_list.append([str(self.indexes_from_sentence(src_lang, src_seq[i])),
                                 str(self.indexes_from_sentence(tgt_lang, tgt_seq[i]))])

        return src_list, tgt_list, src_tgt_list
