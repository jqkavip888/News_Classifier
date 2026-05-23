# 模型剪枝
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.nn.utils.prune import l1_unstructured

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 输入通道1，输出通道6，卷积核尺寸3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)       # 5*5是卷积之后的图片尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))          # 池化后抻平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = LeNet().to(device)
    print(model)
    print(list(model.named_parameters()))   # 权重w
    print(list(model.named_buffers()))      # 剪枝矩阵，也就是生成一堆01乘数，把要剪掉的w * 0

    # 特定层剪枝方法
    prune.random_unstructured(model,'weight',amount=0.2)       # 剪枝操作，0.1比例，也可以设置整数=剪掉的数量
    prune.l1_unstructured(model,'weight',amount=3)  # 范数剪枝，L1=绝对值，L2=平方

    # 结构化剪枝，盯着一个神经元剪枝
    prune.ln_structured(model,'weight',amount=0.2,n=2,dim=1)  # 范数剪枝，L2=平方


    # 多层剪枝，遍历每一层，使用不同策略剪枝
    for name,module in model.named_modules():
        if isinstance(module,nn.Conv2d):
            prune.l1_unstructured(model, 'weight', amount=3)

        elif isinstance(module,nn.Linear):
            prune.ln_structured(model, 'weight', amount=0.2, n=2, dim=1)

    # 查看剪枝的权重w量与临时list存放的w
    print(list(model.named_parameters()))
    print(list(model.named_buffers()))

    # 全局剪枝
    # 定义一个全局剪枝参数
    prune_params = {
        (model.conv1, "weight"),
        (model.conv2, "weight"),
        (model.fc1, "weight"),
        (model.fc2, "weight"),
        (model.fc3, "weight")
    }
    # 调用全局api，设置剪枝方法与比例，执行操作
    prune.global_unstructured(prune_params,pruning_method=prune.L1Unstructured,amount=0.2)

    # 查看结果
    print(list(model.named_parameters()))
    print(list(model.named_buffers()))

    # 计算单一层剪枝效果
    print('sparisity in conv1.weight {:.2f}%'.format(
        100. * float(torch.sum(model.conv1.weight ==0))/float(model.conv1.weight.nelement())))


    # 自定义剪枝
    class MyPruner(prune.BasePruningMethod):
        def compute_mask(self, module, defalut_mask):
            mask = defalut_mask.clone()
            mask.view(-1)[::3] = 0          # 抻平，间隔2步设置为0
            return mask

    def myself_unstructured_pruning(module,name):   # 定义一个剪枝方法
        MyPruner.apply(module,name)     # 剪枝类实例化
        return module

    # 查看自定义剪枝效果
    myself_unstructured_pruning(module=model.fc2,name='weight')
    print(model.fc2.weight_mask)


