# 模型蒸馏分词器
import torch
from tqdm import tqdm
import time
import os
from datetime import timedelta
import pickle as pkl

UNK, PAD, CLS = '[UNK]', '[PAD]', '[CLS]'
max_vocab_size = 10000  # 词表长度限制


def build_vocab(file_path, tokenizer, max_size, min_freq):
    """
    # 构建字表，字的信息熵更低，用于模型蒸馏
    :param file_path: 文件路径
    :param tokenizer: 分词器
    :param max_size: 最大句子长度
    :param min_freq: 最小词频
    :return:词频映射字典vocab_dic
    """

    vocab_dic = {}  # 空字典，用于构建词频映射

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line_data = line.strip()
            if not line_data:
                continue
            content = line_data.split('\t')[0]  # 使用制表符分割文本
            # 使用给定的分词器进行分词，并更新词汇表
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1

    # 按照词频表对词进行排序，并选择词频较高的词
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] > min_freq],
                        key=lambda x: x[1], reverse=True)[:max_size]
    # 构建词频映射字典
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    # 添加特殊符号
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1, CLS: len(vocab_dic) + 2})

    return vocab_dic


def build_dataset_CNN(config):
    # 自定义一个字符级别的分词器
    tokenizer = lambda x: [y for y in x]
    # 检查是否存在词表，如果存在则加载，不存在则新建
    if os.path.exists(config.vocab_letter_path):
        vocab_letter = pkl.load(open(config.vocab_letter_path, 'rb'))
    else:
        min_freq = getattr(config, 'min_freq', 1)
        vocab_letter = build_vocab(config.train_path, tokenizer, max_vocab_size, min_freq)

        # 保存
        pkl.dump(vocab_letter, open(config.vocab_letter_path, 'wb'))

    print(f'分词表长度:', {len(vocab_letter)})

    def load_dataset(path, pad_size=32):
        content = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if not line.strip():
                    continue
                sentence, label = line.strip().split('\t')
                tokens = tokenizer(sentence)
                # 判断长度
                seq_len = len(tokens)
                if seq_len < pad_size:
                    tokens.extend([PAD] * (pad_size - seq_len))

                else:
                    tokens = tokens[:pad_size]
                    seq_len = len(tokens)

                # 从vocab_letter获取映射，建立一个list
                tokens_ids = []
                for token in tokens:
                    tokens_ids.append(vocab_letter.get(token, vocab_letter.get(UNK)))

                content.append((tokens_ids, int(label),seq_len, [0] * pad_size))

        return content

    train = load_dataset(config.train_path, pad_size=config.pad_size)
    dev = load_dataset(config.dev_path, pad_size=config.pad_size)
    test = load_dataset(config.test_path, pad_size=config.pad_size)

    return vocab_letter, train, dev, test


# 定义一个数据处理函数dataset
def build_dataset(config):
    # load dataset由于功能单一，可做成闭包，顺便限制函数作用域
    def load_dataset(data_path, pad_size):
        """
        一个数据处理方法
        :param data_path: 数据集路径
        :param pad_size: 句子长度
        :return: train，test
        """

        contents = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for i in tqdm(f):
                sentence, label = i.strip().split('\t')
                token = config.tokenizer.tokenize(sentence)
                sequence = ['[CLS]'] + token + ['[SEP]']
                seq_idx = config.tokenizer.convert_tokens_to_ids(sequence)

                # mask = []
                seq_len = len(seq_idx)
                if seq_len < pad_size:
                    seq_idx += [0] * (pad_size - seq_len)
                    mask = [1] * seq_len + [0] * (pad_size - seq_len)
                else:
                    seq_idx = seq_idx[:pad_size]
                    mask = [1] * pad_size
                    seq_len = pad_size

                contents.append((seq_idx, int(label), seq_len, mask))

            return contents

    train_dataset = load_dataset(config.train_path, config.pad_size)
    dev_dataset = load_dataset(config.dev_path, config.pad_size)
    test_dataset = load_dataset(config.test_path, config.pad_size)

    return train_dataset, dev_dataset, test_dataset


# 定义一个样本迭代类，底层是dataloader
class DatesetIterater():
    def __init__(self, batches, batch_size, device, model_name):
        self.batches = batches  # 样本列表
        self.batch_size = batch_size  # 每批次大小
        self.device = device  # 数据加载到的设备
        self.model_name = model_name  # 使用的模型名称
        self.n_batches = len(self.batches) // batch_size  # 批次数量，取整运算
        self.residue = False  # 记录batch是否为整数个
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 0  # 当前批次的index

    def __next__(self):
        if self.residue == True and self.index == self.n_batches:
            data = self.batches[self.index * self.batch_size:len(self.batches)]
            data = self.__to_tensor(data)
            self.index += 1
            return data
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            data = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            data = self.__to_tensor(data)
            self.index += 1
            return data


    def __to_tensor(self, data):
        x = torch.LongTensor([i[0] for i in data]).to(self.device)
        y = torch.LongTensor([i[1] for i in data]).to(self.device)

        if self.model_name == 'bert':
            seq_len = torch.LongTensor([i[2] for i in data]).to(self.device)
            mask = torch.LongTensor([i[3] for i in data]).to(self.device)
            return (x, seq_len, mask), y

        if self.model_name == 'textCNN':
            seq_len = torch.LongTensor([i[2] for i in data]).to(self.device)
            return (x, seq_len), y


    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

    def __iter__(self):
        return self


# 实例化dataloader
def build_iter(data, config, shuffle=True):
    iter = DatesetIterater(data, config.batch_size, config.device, config.model_name)
    return iter


# 计算时间差
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))  # 把时间差转换为整数秒


if __name__ == '__main__':
    tokenizer = lambda x: [y for y in x]
    file_path = '../01-fasttext/data/dev.txt'
    vocab_dic = build_vocab(file_path, tokenizer, max_vocab_size, min_freq=0)
    pkl.dump(vocab_dic, open('../03-distill/data/vocab_dic.pkl', 'wb'))

    print('模型蒸馏分词器 end')
