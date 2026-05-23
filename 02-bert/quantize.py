# 模型量化

import torch
import numpy as np
from train_eval import test
from importlib import import_module
import argparse
from utils import build_dataset, build_iter

# 命令行参数解析
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='bert', help='choose model')
args = parser.parse_args()

if __name__ == '__main__':
    if args.model == 'bert':
        # 指定模型bert
        model_name = 'bert'
        x = import_module('models.' + model_name)
        config = x.Config()

        # 设置random seed
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

        # 处理dataloader
        print("正在加载数据集...")
        train_data, test_data = build_dataset(config)
        train_iter = build_iter(train_data, config)
        test_iter = build_iter(test_data, config)

        # 实例化模型并加载参数，只能在cpu上量化模型
        model = x.Model(config)
        print(model)
        model.load_state_dict(torch.load(config.save_path1, map_location='cpu'))

        # 强制将模型也设为 CPU 模式并开启评估模式
        model.cpu()
        model.eval()

        # 🚨 专门为 Mac 电脑（尤其是 Apple Silicon）指定的量化引擎！
        torch.backends.quantized.engine = 'qnnpack'

        print("量化前的模型结构：")
        # 量化模型
        quantize_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, inplace=False)
        print(quantize_model)

        # 测试量化模型的表现
        test(config, quantize_model, test_iter, test_data)

        # 保存量化模型
        torch.save(quantize_model.state_dict(), config.save_path2)
