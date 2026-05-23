# 配置教师模型

import torch
import torch.nn as nn
import os
from transformers import BertTokenizer, BertModel, BertConfig

print(torch.backends.mps.is_available())  # 应该输出 True

# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 配置一个bert控制类
class Bert_Config(object):
    def __init__(self):
        self.model_name = 'bert'

        base_dir = os.path.dirname(os.path.abspath(__file__))  # models目录
        project_dir = os.path.dirname(base_dir)  # 03-distill目录

        self.data_path = os.path.join(project_dir, 'data')
        self.train_path = os.path.join(self.data_path, 'train.txt')
        self.dev_path = os.path.join(self.data_path, 'dev.txt')
        self.test_path = os.path.join(self.data_path, 'test.txt')
        self.vocab_letter_path = os.path.join(self.data_path, 'vocab.pkl')
        self.class_path = os.path.join(self.data_path, 'class.txt')
        self.class_list = [x.strip() for x in open(self.class_path, 'r', encoding='utf-8')]


        self.save_dir = '/Users/lianghao/Desktop/helloworld/news_classifier/03-distill/models/src/save_dict'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir,exist_ok=True)
        # self.save_path += '/' + self.model_name + '.pt'  # 模型训练结果保存路径
        self.save_path = f"{self.save_dir}/{self.model_name}.pt"


        # self.save_path2 = '../src/saved.dic2'
        # if not os.path.exists(self.save_path2):
        #     os.makedirs(self.save_path2,exist_ok=True)
        # self.save_path2 += '/' + self.model_name + '_quantized.pt'  # 量化模型训练结果保存路径

        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        self.num_classes = len(self.class_list)
        self.num_epochs = 2
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 5e-5
        self.bert_path = os.path.join(project_dir, 'models', 'bert_pretrain')
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path + '/bert_config.json')  # bert模型配置器
        self.hidden_size = 768


# 配置一个sft类
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        if isinstance(x,tuple):
            input_ids = x[0]
            mask = x[2] if len(x)>2 else None
        else:
            input_ids = x
            mask = None
        # _, pooled = self.bert(x[0], attention_mask=x[2], return_dict=False)  # 抽取[cls]
        _, pooled = self.bert(input_ids, attention_mask=mask, return_dict=False)
        out_data = self.fc(pooled)  # 得到logits，可以使用cross_entropy与y计算loss

        return out_data
