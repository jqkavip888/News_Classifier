import random

# 1. 读取清洗好的全量数据
with open('data/ft_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 2. 随机打乱并按 8:2 切分
random.seed(42)
random.shuffle(lines)

split_idx = int(len(lines) * 0.8)
train_lines = lines[:split_idx]
test_lines = lines[split_idx:]

# 3. 保存为两个新文件
with open('data/train_80.txt', 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

with open('data/test_20.txt', 'w', encoding='utf-8') as f:
    f.writelines(test_lines)

print(f"切分完成！训练集: {len(train_lines)} 行，测试集: {len(test_lines)} 行")