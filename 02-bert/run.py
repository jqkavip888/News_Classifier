import torch
import time
import numpy as np
from importlib import import_module
import argparse
from utils import build_dataset, build_iter
from train_eval import train, test

parser = argparse.ArgumentParser(description='Chinese text classification')
parser.add_argument('--model', type=str, default='bert',
                    help='choose a models:bert')
args = parser.parse_args()

if __name__ == '__main__':
    if args.model == 'bert':
        model_name = 'bert'
        x = import_module('models.' + model_name)
        config = x.Config()

        # 1. 设置随机种子，保证每次跑的结果一样
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

        # 2. 构建数据集，名字严格叫 train_data 和 test_data！
        print("正在加载数据集...")
        train_data, test_data = build_dataset(config)
        train_iter = build_iter(train_data, config)
        test_iter = build_iter(test_data, config)

        # 3. 创建模型并放入 GPU/CPU
        print("初始化模型...")
        models = x.Model(config).to(config.device)

        # 4. 正式启动训练！（这里才是 train 函数真正该出场的地方）
        print("启动训练引擎...")
        # train(config, models, train_iter, test_iter)

        # 5. 读取存档
        print("训练结束，直接读取已训练好的模型权重...")
        models.load_state_dict(torch.load(config.save_path1))

        # 6. 测试集最终评估（以及我们的坏例打印）
        print("开始全面测试与坏例抓取...")
        test(config, models, test_iter, test_data)