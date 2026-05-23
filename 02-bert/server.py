# 创建一个server功能

import numpy as np
import torch
from importlib import import_module
from flask import Flask, request
import jieba


# 建立一个分类映射表
CLS = '[CLS]'
SEP = '[SEP]'

id_to_name = {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science',
              5: 'society', 6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'}


# 全局参数与实例化
model_name = 'bert'
x = import_module('models.' + model_name)
config = x.Config()
model = x.Model(config)
weights = torch.load(config.save_path1, map_location=torch.device('cpu'))
model.load_state_dict(weights)


def inference(config, model, input_text, padding_size=32):
    """
    # 用于前台用户输入接口
    :param config: 配置工具
    :param model: 模型
    :param input_text:用户输入文本
    :param padding_size: 句子长度，决定是否填0或切断
    :return:
    """
    # text转idx
    token = config.tokenizer.tokenize(input_text)
    token = [CLS] + token + [SEP]
    seq_len = len(token)
    token_ids = config.tokenizer.convert_tokens_to_ids(token)

    # 取长补短
    if seq_len < padding_size:
        mask = [1] * seq_len + [0] * (padding_size - seq_len)
        token_ids += [0] * (padding_size - seq_len)
    else:
        mask = [1] * padding_size
        token_ids = token_ids[:padding_size]
        seq_len = padding_size

    # 张量化+增维
    token_ids = torch.LongTensor([token_ids])
    mask = torch.LongTensor([mask])
    seq_len = torch.LongTensor([seq_len])

    # 张量拼接，送入网络
    x = [token_ids,seq_len,mask]

    model.eval()  # 开启评估模式
    with torch.no_grad():  # 关闭梯度计算
        outputs = model(x)

    result_id = torch.max(outputs,1)[1]
    predict_name = id_to_name[result_id.item()]

    return predict_name


# 部署到flask
app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def index():
    uid = request.form['uid']
    text = request.form.get('text','')
    res = inference(config, model, text)
    return res

if __name__ == '__main__':
    # text = '北京大学近日正式倒闭'
    # print(inference(config, model, text))
    app.run(host='0.0.0.0', port=19999)




