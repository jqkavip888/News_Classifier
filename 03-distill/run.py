# 模型蒸馏启动器
import numpy as np
import torch
from importlib import import_module
from tqdm import tqdm
import argparse
from utils import build_dataset_CNN, build_iter, build_dataset
from train_eval import train, train_kd

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--task', type=str, default='train_kd', help='choose or task:trainbert,or train_kd')
args = parser.parse_args()

if __name__ == '__main__':
    # 执行教师bert
    if args.task == 'train_bert':
        model_name = 'bert'
        bert_model = import_module('models.' + model_name)
        bert_config = bert_model.Bert_Config()
        # 初始化
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

        # 构建数据集
        print('load data from bert model...')
        train_data, dev_data, test_data = build_dataset(bert_config)
        # train_data = train_data[:len(train_data) // 50]  # 先跑18000条
        train_iter = build_iter(train_data, bert_config)
        dev_iter = build_iter(dev_data, bert_config)
        test_iter = build_iter(test_data, bert_config)
        # 模型实例化，训练
        bert_model = bert_model.Model(bert_config).to(bert_config.device)
        train(bert_config, bert_model, train_iter, dev_iter, test_iter)

    if args.task == 'train_kd':
        model_name = 'bert'
        bert_model = import_module('models.' + model_name)
        bert_config = bert_model.Bert_Config()

        # 加载cnn学生模型
        model_name = 'textCNN'
        cnn_model = import_module('models.' + model_name)
        cnn_config = cnn_model.TextCNNConfig()

        # 初始化
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

        # 构建bert数据集，只需要bert训练结果作为soft ypre，不需要dev和test
        print('load data from bert model...')
        bert_train_data, _, _ = build_dataset(bert_config)
        bert_train_iter = build_iter(bert_train_data, bert_config)

        # 构建cnn数据集，需要全量数据集
        vocab, cnn_train_data, cnn_dev_data, cnn_test_data = build_dataset_CNN(cnn_config)
        cnn_train_data = build_iter(cnn_train_data, cnn_config)
        cnn_dev_data = build_iter(cnn_dev_data, cnn_config)
        cnn_test_data = build_iter(cnn_test_data, cnn_config)
        cnn_config.n_vocab = len(vocab)

        # 加载训练好的教师bert模型
        bert_model = bert_model.Model(bert_config).to(bert_config.device)
        print(f"模型设备: {next(bert_model.parameters()).device}")
        print(f"使用设备: {bert_config.device}")
        bert_model.load_state_dict(
            torch.load(bert_config.save_path, map_location=bert_config.device)
        )
        bert_model.eval()
        for p in bert_model.parameters():
            p.requires_grad = False
        # 加载学生cnn模型
        cnn_model = cnn_model.Model(cnn_config).to(cnn_config.device)
        print('teacher and student models loaded, start training...')
        # 训练时加载的双方配置，双方模型，双方迭代器，都要区分清楚
        train_kd(bert_config, cnn_config, bert_model, cnn_model,
                 bert_train_iter, cnn_train_data, cnn_dev_data, cnn_test_data)
