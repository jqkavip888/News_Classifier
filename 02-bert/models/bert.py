import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# 配置一个bert控制类
class Config(object):
    def __init__(self):
        self.model_name = 'bert'
        self.data_path = '../../01-fasttext/data/'
        self.train_path = self.data_path + 'dev.txt'
        self.test_path = self.data_path + 'test.txt'
        self.class_list = [x.strip() for x in open(self.data_path + 'class.txt').readlines()]

        self.save_path1 = '../src/saved.dic1'
        if not os.path.exists(self.save_path1):
            os.makedirs(self.save_path1,exist_ok=True)
        self.save_path1 += '/' + self.model_name + '.pt'  # 模型训练结果保存路径

        self.save_path2 = '../src/saved.dic2'
        if not os.path.exists(self.save_path2):
            os.makedirs(self.save_path2,exist_ok=True)
        self.save_path2 += '/' + self.model_name + '_quantized.pt'  # 量化模型训练结果保存路径

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_classes = len(self.class_list)
        self.num_epochs = 2
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 5e-5
        self.bert_path = '../bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path + '/bert_config.json')  # bert模型配置器
        self.hidden_size = 768


# 配置一个sft类
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path,config=config.bert_config)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        _, pooled = self.bert(x[0], attention_mask=x[2], return_dict=False)  # 抽取[cls]
        out_data = self.fc(pooled)  # 得到logits，可以使用cross_entropy与y计算loss
        return out_data
