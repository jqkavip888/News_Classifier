import os
import jieba
import fasttext
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time


os.chdir(os.path.dirname(__file__))
# print(os.getcwd())

# 建立一个labels映射关系
id_to_cls = {}
with open('data/class.txt', 'r', encoding='utf-8') as f:
    for idx, label in enumerate(f):
        label_clean = label.strip()
        id_to_cls[idx] = label_clean
    print(id_to_cls)


# 清洗数据，拼接成一个fasttext格式的训练集
def originate_data_clean():
    with (open('data/dev.txt', 'r', encoding='utf-8') as f_in,
          open('data/ft_train.txt', 'w', encoding='utf-8') as f_out):
        for i in f_in:
            clean_sentence, label = i.strip().split('\t')
            fast_sentence = ' '.join(jieba.cut(clean_sentence))
            features_fast = f"__label__{id_to_cls[int(label)]} {fast_sentence}"

            f_out.write(f"{features_fast}\n")

        print('数据保存txt完成')


# 使用fasttext学习，设置为随机自动搜索最优超参数
def fast_train():
    model = fasttext.train_supervised('data/ft_train.txt',
                                      autotuneValidationFile='data/ft_train.txt',
                                      autotuneDuration=30,
                                      wordNgrams=2,
                                      verbose=3)
    print('词表大小: ',len(model.words))
    print('标签列表:',model.labels)
    # result = model.test('data/ft_test.txt')
    # print(result)

    # 模型保存
    model.save_model('data/models.bin')


# 模型读取
model = fasttext.load_model('data/models.bin')


# 模型预测
def model_predict(sentence):
    y_pre = model.predict(' '.join(jieba.cut(sentence)))
    return y_pre



def model_evaluate(test_file='data/ft_test.txt'):
    y_true = []
    y_pred = []

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # fasttext 格式：__label__xxx 文本
            parts = line.split(' ', 1)
            true_label = parts[0].replace('__label__', '')
            text = parts[1] if len(parts) > 1 else ''

            pred_label = model.predict(' '.join(jieba.cut(text)))[0][0]
            pred_label = pred_label.replace('__label__', '')

            y_true.append(true_label)
            y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )

    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"macro F1:  {f1:.4f}")


if __name__ == '__main__':
    # originate_data_clean()
    # fast_train()
    # print(model_predict('阿根廷足协保证不再开除梅西'))
    # print(model_predict('北京大学今日起正式倒闭'))
    # print(model_predict('战神6加入新角色超级马里奥，玩家表示非常期待'))
    # print(model_predict('金价全面暴跌，1美元可买10吨黄金'))
    start_time = time.time()
    model_evaluate()
    end_time = time.time()
    print('cost time: ',end_time - start_time)
