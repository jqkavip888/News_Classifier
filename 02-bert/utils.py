import torch
from tqdm import tqdm
import time
from datetime import timedelta

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
    test_dataset = load_dataset(config.test_path, config.pad_size)

    return train_dataset, test_dataset


# 定义一个样本迭代类，底层是dataloader
class DatesetIterater():
    def __init__(self, batches, batch_size, device, model_name):
        self.batches = batches  # 样本列表
        self.batch_size = batch_size  # 每批次大小
        self.device = device  # 数据加载到的设备
        self.model_name = model_name  # 使用的模型名称
        self.n_batches = len(self.batches) // batch_size  # 批次数量，取整运算
        self.residue = False  # 记录batch是否为整数个
        if len(batches) % self.n_batches != 0:
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
        seq_len = torch.LongTensor([i[2] for i in data]).to(self.device)
        mask = torch.LongTensor([i[3] for i in data]).to(self.device)

        if self.model_name == 'bert':
            return (x, seq_len, mask), y

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

    def __iter__(self):
        return self

# 实例化dataloader
def build_iter(data, config):
    iter = DatesetIterater(data, config.batch_size, config.device, config.model_name)
    return iter

# 计算时间差
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))     # 把时间差转换为整数秒

