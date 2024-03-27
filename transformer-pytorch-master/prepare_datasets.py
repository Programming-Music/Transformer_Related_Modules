from datasets import TranslationDataset, TranslationDatasetOnTheFly
from datasets import TokenizedTranslationDataset, TokenizedTranslationDatasetOnTheFly
from datasets import InputTargetTranslationDataset, InputTargetTranslationDatasetOnTheFly
from datasets import IndexedInputTargetTranslationDataset, IndexedInputTargetTranslationDatasetOnTheFly
from dictionaries import IndexDictionary
from utils.pipe import shared_tokens_generator, source_tokens_generator, target_tokens_generator

from argparse import ArgumentParser

"""
   @staticmethod, 不依赖于类, 可以通过class.method方式进行访问
   @classmethod,  类似@staticmethod。其中的cls代表函数所在类, 可通过cls.**访问类中的元素, 接近self.则访问对象中的元素
"""


parser = ArgumentParser('Prepare datasets')
parser.add_argument('--train_source', type=str, default='data/example/raw/src-train.txt')
parser.add_argument('--train_target', type=str, default='data/example/raw/tgt-train.txt')
parser.add_argument('--val_source', type=str, default='data/example/raw/src-val.txt')
parser.add_argument('--val_target', type=str, default='data/example/raw/tgt-val.txt')
parser.add_argument('--save_data_dir', type=str, default='data/example/processed')
    # 语言相近的词汇对，可使用共享词汇表
parser.add_argument('--share_dictionary', type=bool, default=False)

args = parser.parse_args()

    # prepare为@static函数，独立于类，可直接调用
    # 1. 输入基于行的src/tgt文件，返回基于(src,tgt)字符串的dataset
        # 分别基于原始文件进行处理，和构建中间文件raw-***.txt进行处理
TranslationDataset.prepare(args.train_source, args.train_target, args.val_source, args.val_target, args.save_data_dir)
translation_dataset = TranslationDataset(args.save_data_dir, 'train')

translation_dataset_on_the_fly = TranslationDatasetOnTheFly('train')
assert translation_dataset[0] == translation_dataset_on_the_fly[0]

    # 2. 输入基于(src, tgt)字符串的dataset，返回基于(src, tgt)列表的dataset
tokenized_dataset = TokenizedTranslationDataset(args.save_data_dir, 'train')

    # 3. 基于(src, tgt)列表的dataset，基于源域和目标域是否共享词汇表，生成域字典vocabulary-{self.mode}.txt文件并保存
if args.share_dictionary:
        # 若共享，依次基于src, tgt构造索引字典
    source_generator = shared_tokens_generator(tokenized_dataset)
    source_dictionary = IndexDictionary(source_generator, mode='source')
    target_generator = shared_tokens_generator(tokenized_dataset)
    target_dictionary = IndexDictionary(target_generator, mode='target')

    source_dictionary.save(args.save_data_dir)
    target_dictionary.save(args.save_data_dir)
else:
        # 若不共享，分别基于src/tgt构造索引字典
    source_generator = source_tokens_generator(tokenized_dataset)
    source_dictionary = IndexDictionary(source_generator, mode='source')
    target_generator = target_tokens_generator(tokenized_dataset)
    target_dictionary = IndexDictionary(target_generator, mode='target')

        # 保存形式 {vocab_index, \t, vocab_token, \t, count, \n}
    source_dictionary.save(args.save_data_dir)
    target_dictionary.save(args.save_data_dir)

source_dictionary = IndexDictionary.load(args.save_data_dir, mode='source')
target_dictionary = IndexDictionary.load(args.save_data_dir, mode='target')

    # 4. 基于列表数据集生成source, inputs和targets(target)，得到在源域/目标域字典中的索引，保存在index-***中
    # 保存形式: '{indexed_sources}\t{indexed_inputs}\t{indexed_targets}\n'
    # 三元组 将翻译问题抽象为序列处理问题，是transformer标准的数据格式
IndexedInputTargetTranslationDataset.prepare(args.save_data_dir, source_dictionary, target_dictionary)

    # 将三元组中的字符串，处理为整形格式
indexed_translation_dataset = IndexedInputTargetTranslationDataset(args.save_data_dir, 'train')
indexed_translation_dataset_on_the_fly = IndexedInputTargetTranslationDatasetOnTheFly('train', source_dictionary, target_dictionary)
assert indexed_translation_dataset[0] == indexed_translation_dataset_on_the_fly[0]

print('Done datasets preparation.')