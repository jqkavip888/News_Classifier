# 数据对齐

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from scipy.cluster.hierarchy import is_valid_im
from sklearn import metrics
import tqdm
import time
import math
from utils import get_time_dif
import logging


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# 教师模型进行y_pre输出
def fetch_teacher_output(teacher_model, train_iter):
    """
    使用教师模型进行预测
    :param teacher_model: 教师模型bert
    :param train_iter: 训练集迭代器
    :return: teacher_outputs教师模型预测值
    """
    teacher_model.eval()
    teacher_outputs = []
    with torch.no_grad():
        for i, (data_batch, labels_batch) in enumerate(tqdm.tqdm(train_iter)):
            output = teacher_model(data_batch)
            teacher_outputs.append(output.detach().to('cpu'))

        return teacher_outputs


# 由于使用真实y标签，使用CrossEntropyLoss求模型loss
def CE_loss_tool(outputs, labels):
    """
    模型loss
    :param outputs: 模型预测值
    :param labels: 真实y标签
    :return: CrossEntropyLoss
    """
    return nn.CrossEntropyLoss()(outputs, labels)


# 由于使用y_pre作为y标签，使用KL散度求loss
criterion = nn.KLDivLoss()


# 联合loss
def loss_teacher_student(student_outputs, labels, teacher_output):
    """
    使用两个标签求联合loss
    :param student_outputs:学生模型预测值
    :param labels:真实y标签
    :param teacher_output: 教师模型预测值
    :return:total_loss
    """
    alpha = 0.8
    T = 2  # 蒸馏温度，作为一个模型预测值除数，小=跟老师学/大=跟数据集学
    kl_loss = criterion(F.log_softmax(student_outputs / T, dim=1),
                        F.softmax(teacher_output / T, dim=1))  # 从老师学到的soft loss
    ce_loss = CE_loss_tool(student_outputs, labels)  # 学生模型hard loss
    total_loss = (kl_loss * alpha * T * T) + (1 - alpha) * ce_loss  # 软loss要* T^2

    return total_loss


# 模型训练
def train(config, model, train_iter, dev_iter, test_iter):
    """
    模型训练函数
    :param config: 教师模型训练配置
    :param model: bert模型
    :param train_iter: 训练集
    :param test_iter: 测试集
    :return:
    """
    # 记录开始时间
    start_time = time.time()

    # 将模型设置为训练模式
    model.train()

    # 获取模型参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # 参数分组，并设置优化器权重衰减
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay_rate": 0.01
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay_rate": 0.0
        }
    ]

    # 使用通用adamw优化器，并设置lr
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    # 记录最佳loss
    dev_best_loss = float('inf')
    # 嵌套循环开始训练
    for epoch in range(config.num_epochs):
        total_batches = 0
        print("epoch [{}/{}]".format(epoch + 1, config.num_epochs))  # print训练epoch
        # 遍历每个batch
        for i, (trains, labels) in enumerate(tqdm.tqdm(train_iter)):
            # grad清零，
            optimizer.zero_grad()
            # forward，计算loss
            # outputs = model(trains[0])
            outputs = model(trains)  # 把完整元组传进去
            loss = CE_loss_tool(outputs, labels)
            # backward，更新loss
            loss.backward()
            optimizer.step()
            total_batches += 1
            # 每400次batch打印一次信息
            if (total_batches % 400 == 0 > total_batches) > 0:
                true = labels.data.cpu()
                pred = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, pred)
                # 使用验证集进行评估
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # 检查当前模型是否是最佳模型
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # 当模型有提升时保存模型权重
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                else:
                    improve = ""

                time_dif = get_time_dif(start_time)
                msg = "iter:{0:>6},train loss:{1:>5.2},train acc:{2:>6.2%},val loss:{3:>5.2},val acc:{4:>6.2%},time:{5:>6.2}"
                print(msg.format(total_batches, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

                # 把模型重新修改为训练模式
                model.train()
    torch.save(model.state_dict(), config.save_path)
    # 使用测试集测试最终效果
    test(config, model, test_iter)


def train_kd(bert_config,cnn_config, bert_model, cnn_model, bert_train_iter, cnn_train_iter, cnn_dev_iter, cnn_test_iter):
    """
    使用知识蒸馏（knowledge distillation）方式训练学生模型
    :param cnn_config:学生模型配置
    :param bert_model:教师模型
    :param bert_train_iter:教师模型训练集
    :param cnn_train_iter:学生模型训练集
    :param cnn_test_iter:学生模型测试集
    :return:
    """
    # 记录训练开始时间
    start_time = time.time()
    # 获取cnn参数
    param_optimizer = list(cnn_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # 参数分组，并设置优化器权重衰减
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay_rate": 0.01
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay_rate": 0.0
        }
    ]
    # 配置优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=cnn_config.learning_rate)
    dev_best_loss = float('inf')  # 记录最佳验证损失
    # 学生cnn模型设置为训练模式
    cnn_model.train()
    # 教师bert模型设置为评估模式
    bert_model.eval()
    # 获取教师bert模型结果作为预测结果
    teacher_outputs = fetch_teacher_output(bert_model, bert_train_iter)
    # 嵌套循环进行训练
    for epoch in range(cnn_config.num_epochs):
        total_batches = 0
        print("epoch [{}/{}]".format(epoch + 1, cnn_config.num_epochs))
        for i, (trains, labels) in enumerate(tqdm.tqdm(cnn_train_iter)):
            # 梯度清零
            cnn_model.zero_grad()
            # 前向传播，计算蒸馏损失，也就是联合损失
            student_outputs = cnn_model(trains)
            # loss = loss_teacher_student(student_outputs, labels, teacher_outputs[i])
            teacher_output = teacher_outputs[i].to(cnn_config.device)
            loss = loss_teacher_student(student_outputs, labels, teacher_output)
            # 反向传播，更新参数
            loss.backward()
            optimizer.step()
            total_batches += 1
            # 每400个batch打印一次信息
            if (total_batches % 400 == 0 > total_batches) > 0:
                true = labels.data.cpu()
                pred = torch.max(student_outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, pred)
                # 在cnn验证集进行评估
                dev_acc, dev_loss = evaluate(cnn_config, cnn_model, cnn_dev_iter)
                # 检查当前模型是否是最优
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(cnn_model.state_dict(), cnn_config.save_path)
                    improve = "*"
                else:
                    improve = ""
                time_dif = get_time_dif(start_time)
                msg = "iter:{0:>6},train loss:{1:>5.2},train acc:{2:>6.2%},val loss:{3:>5.2},val acc:{4:>6.2%},time:{5:>6.2}"
                print(msg.format(total_batches, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 将cnn模型设置为训练模式
                cnn_model.train()
    torch.save(cnn_model.state_dict(), cnn_config.save_path)

    # 在cnn测试集测试最终模型
    test(cnn_config, cnn_model, cnn_test_iter)


def evaluate(config, model, data_iter, test=False):
    """
    模型评估函数
    :param config:配置信息
    :param model:模型
    :param test_iter:测试集迭代器
    :return:
    """
    model.eval()
    total_loss = 0
    # 预测结果
    predicted_all = np.array([], dtype=int)
    # label信息
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = CE_loss_tool(outputs, labels)
            total_loss += loss.item()
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predicted_all = np.append(predicted_all, predict)
    # 计算准确率
    acc = metrics.accuracy_score(labels_all, predicted_all)
    if test:
        # 如果是测试集评估，计算分类报告和混淆矩阵
        report = metrics.classification_report(labels_all, predicted_all, target_names=config.class_list)
        confusion_matrix = metrics.confusion_matrix(labels_all, predicted_all)
        return acc, report, confusion_matrix, total_loss / len(data_iter)
    else:
        # 如果是验证集评估，只返回准确率和平均loss
        return acc, total_loss / len(data_iter)


def test(config, model, test_iter):
    """
    模型测试函数，用来对模型进行最终测试
    :param config:配置信息
    :param model:模型
    :param test_iter:测试集迭代器
    :return:
    """
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    start_time = time.time()
    # 调用验证函数计算评估指标
    test_acc, test_report, test_confusion, test_loss = evaluate(config, model, test_iter, test=True)

    # 打印测试结果
    msg = "test loss: {0:>5.2}, test acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("precision, recall and F1_score")
    print(test_report)
    print("confusion matrix")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("time dif: {}".format(time_dif))
