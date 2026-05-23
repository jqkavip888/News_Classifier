# 模型训练与评估
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import math
import time
from transformers import BertTokenizer
from utils import get_time_dif
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s')


def train(config, model, train_iter, test_iter):
    """
    配置训练函数
    :param config: 控制器
    :param model: 模型
    :param train_iter:训练集datalodaer
    :param test_iter: 测试集datalodaer
    :return:loss
    """
    # 1，opt相关配置
    param_optimizer = list(model.named_parameters())  # 配置一个优化器参数list
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    best_acc = 0.0

    # 2，外循环epoch，内循环batch，进行训练，并保存参数
    model.train()
    total_batch = 0
    for epoch in range(config.num_epochs):
        logging.info(f"Epoch [{epoch + 1}/{config.num_epochs}]")
        for idx, (trains, labels) in enumerate(tqdm(train_iter)):
            output = model(trains)
            loss = F.cross_entropy(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3. 打印日志与评估验证
            if total_batch % 50 == 0:  # 每 50 个 batch 打印一次情况
                # 刚刚我们讨论过的：获取预测的类别标签
                predicts = torch.max(output.data, 1)[1].cpu()
                true_labels = labels.data.cpu()

                # 计算训练集上的准确率
                train_acc = metrics.accuracy_score(true_labels, predicts)

                logging.info(f"Iter: {total_batch} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc * 100:.2f}%")

                # 如果你想在这里做验证集评估，可以调用 eval 函数
                # 4，模型最优保存
                test_acc, test_loss = eval(config, model, test_iter)
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(model.state_dict(), config.save_path1)
                    improve = "*"
                else:
                    logging.info(f"Best Test Acc: {best_acc:.2f}%")
                    improve = ""

                # 计算时间差
                time_dif = get_time_dif(start_time=time.time())
                # eval(config, model, test_iter)
                # 切记：eval 完之后，要把模型重新切回 model.train()!
                model.train()

            total_batch += 1


def eval(config, model, data_iter, test=False, raw_texts=None):
    """
    评估参数
    :param config: 模型控制器
    :param model: 模型
    :param test_iter:测试集test
    :return:acc,loss,report,f1_score
    """
    # 使用量化进行推理时需要关闭eval模式
    model.eval()
    total_loss = 0
    labels_all = np.array([], dtype=int)
    predicts_all = np.array([], dtype=int)
    # 用于捕获坏例的空list
    bad_cases_texts = []

    with torch.no_grad():
        for i, (texts, labels) in enumerate(tqdm(data_iter, desc="Evaluating")):
            output = model(texts)
            loss = F.cross_entropy(output, labels)
            total_loss += loss.item()

            labels = labels.cpu().numpy()
            labels_pred = torch.max(output.data, 1)[1].cpu().numpy()

            # --- 🚀 坏例抓取核心引擎 start ---
            # 只有在最终测试阶段，且提供了原始文本时才运行
            if test and raw_texts is not None:
                batch_size = labels.shape[0]
                # 计算这一个 batch 在整个数据集中的起始位置
                start_idx = i * batch_size

                # 遍历当前 batch 中的每一个样本
                for j in range(batch_size):
                    true_lab = labels[j]
                    pred_lab = labels_pred[j]

                    # 🎯 设定抓取目标：真实标签是 2 (游戏?)，但模型猜成 0 (财经?)
                    if true_lab == 2 and pred_lab == 0:
                        global_idx = start_idx + j
                        # 根据索引从原始文本列表中取出那条新闻
                        # 注意：具体取 texts[0] 还是 texts[1] 取决于 build_dataset 的返回格式，标准 BERT 是 [0]
                        raw_content = raw_texts[global_idx][0]
                        bad_cases_texts.append(raw_content)
            # --- 🚀 坏例抓取核心引擎 end ---

            predicts_all = np.append(predicts_all, labels_pred)
            labels_all = np.append(labels_all, labels)

    acc = metrics.accuracy_score(labels_all, predicts_all)

    if test:
        # 如果是测试集评估，就计算分类报告和f1混淆矩阵
        report = metrics.classification_report(labels_all, predicts_all)
        confusion_matrix = metrics.confusion_matrix(labels_all, predicts_all)
        return acc, total_loss / len(data_iter), report, confusion_matrix, bad_cases_texts
    else:
        # 如果是验证集评估，只计算准确率和平均loss
        return acc, total_loss / len(data_iter)


def test(config, model, test_iter, test_texts):
    """
    用于在测试集进行最终测试，调用eval函数
    :param config: 控制器，配置工具
    :param model: 模型
    :param test_iter:测试集
    :return: loss,acc,report,f1_score
    """
    # 使用量化进行推理时需要关闭eval模式
    # model.eval()

    start_time = time.time()

    print("\n正在进行最终全面评估并抓取坏例...")
    # 调用新的 eval 函数，传入 test_texts (即 run.py 里的 test_data)
    test_acc, test_loss, test_report, test_confusion, bad_cases = eval(
        config, model, test_iter, raw_texts=test_texts, test=True
    )
    # 打印结果信息
    msg = 'test loss {0:>5.2},test acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print('precision,recall,f1-score')
    print(test_report)
    print('confusion matrix')
    print(test_confusion)
    time_dif = get_time_dif(start_time=start_time)
    print('time dif', time_dif)

    # ------------------ 翻译与打印模块 ------------------
    print("\n" + "=" * 50)
    print(f"📡 【坏例分析报告】")
    print(f"目标：真实类别 [2] 被错误预测为 [0] 的样本")
    print(f"数量：共抓取到 {len(bad_cases)} 条")
    print("=" * 50)

    # 召唤官方中文翻译字典
    print("正在加载 BERT 字典进行翻译，请稍候...")
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        if len(bad_cases) > 0:
            for i, text_ids in enumerate(bad_cases):
                # 如果是 tensor，先转成 list
                if hasattr(text_ids, 'tolist'):
                    text_ids = text_ids.tolist()

                # 解码，跳过 101, 102, 0 这种特殊符号
                text = tokenizer.decode(text_ids, skip_special_tokens=True)
                # 去掉英文 decode 默认带的空格
                text = text.replace(" ", "")

                print(f"[{i + 1:>3}] 原文: {text}")
        else:
            print("太棒了，未抓取到符合条件的坏例！")

    except Exception as e:
        print(f"字典加载失败，报错信息: {e}")

    print("=" * 50 + "\n")
