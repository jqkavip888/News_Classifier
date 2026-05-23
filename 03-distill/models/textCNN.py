# 蒸馏配置文件
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置一个学生textcnn模型控制类
class TextCNNConfig(object):
    def __init__(self):
        self.model_name = 'textCNN'
        base_dir = os.path.dirname(os.path.abspath(__file__))  # models目录
        project_dir = os.path.dirname(base_dir)  # 03-distill目录

        self.data_path = os.path.join(project_dir, 'data')
        self.train_path = os.path.join(self.data_path, 'train.txt')
        self.dev_path = os.path.join(self.data_path, 'dev.txt')
        self.test_path = os.path.join(self.data_path, 'test.txt')
        self.vocab_letter_path = os.path.join(self.data_path, 'vocab.pkl')
        self.class_path = os.path.join(self.data_path, 'class.txt')
        self.class_list = [x.strip() for x in open(self.class_path, 'r', encoding='utf-8')]

        self.num_class = len(self.class_list)
        self.save_dir = '/Users/lianghao/Desktop/helloworld/news_classifier/03-distill/models/src/saved.dict'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir,exist_ok=True)
        # self.save_path += '/' + self.model_name + '.pt'
        self.save_path = f"{self.save_dir}/{self.model_name}.pt"
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.batch_size = 128
        self.pad_size = 32  # 最大句子长度
        self.require_improvement = 1000  # 超过1000次还没有提升则停止训练
        self.embedding_size = 300  # 字向量维度
        self.filter_sizes = (2, 3, 4)  # 卷积核的行数量分类，共3类
        self.num_filters = 256  # 每一类卷积核的数量
        self.dropout = 0.5
        self.learning_rate = 1e-3
        self.n_vocab = 0  # 词表大小，运行时赋值
        self.num_epochs = 3


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_class)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embedding_size)) for k in config.filter_sizes]
        )

    # 卷积与池化
    def conv_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        # 对每个卷积层进行卷积和池化操作，然后进行拼接
        out = torch.cat([self.conv_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
