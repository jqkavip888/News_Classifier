import jieba
import os


os.chdir(os.path.dirname(__file__))


# 建立一个labels映射关系
id_to_cls = {}
with open('data/class.txt', 'r', encoding='utf-8') as f:
    for idx, label in enumerate(f):
        label_clean = label.strip()
        id_to_cls[idx] = label_clean
    print(id_to_cls)


def make_ft_test():
    with (open('data/dev.txt', 'r', encoding='utf-8') as f_in,
        open('data/ft_dev.txt', 'w', encoding='utf-8') as f_out):
        for i in f_in:
            clean_sentence, label = i.strip().split('\t')
            fast_sentence = ' '.join(jieba.cut(clean_sentence))
            features_fast = f"__label__{id_to_cls[int(label)]} {fast_sentence}"

            f_out.write(f"{features_fast}\n")

        print('数据保存txt完成')



if __name__ == '__main__':
    make_ft_test()