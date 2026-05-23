# 中文新闻分类实验 Chinese News Classification

**个人项目 · THUCNews 子集 · BERT / FastText / 知识蒸馏**
Personal Project · THUCNews Subset · BERT / FastText / Knowledge Distillation

THUCNews 10 类中文新闻数据集，覆盖 FastText → BERT → 量化 → 蒸馏完整实验链路。
THUCNews 10-class dataset: train 180,000 / dev 10,000 / test 10,000.
Confusion matrix revealed 2 categories below average — identified as stock vs finance label boundary ambiguity in the dataset annotation.

---

## ① 模型选型对比
Model Selection Comparison

数据集为 THUCNews 10 类子集：  

- 训练集：180,000 条  
- Dev / Test 集：各 10,000 条
- THUCNews 10-class dataset: train 180,000 / dev 10,000 / test 10,000.
  

观察混淆矩阵发现有 2 个分类显著低于均值，经过进一步分析定位为数据集标注本身的标签类别边界模糊问题（“股票”与“财经”）。
Confusion matrix revealed 2 categories below average — identified as stock vs finance label boundary ambiguity in the dataset annotation.

| model         | Accuracy | macro P | macro R | macro F1 | infer time cost |
| ------------ | -------- | ------- | ------- | -------- | -------- |
| FastText     | 77.89%   | 0.7842  | 0.7789  | 0.7801   | 1.48 s   |
| BERT FP32    | 90.21%   | 0.90    | 0.90    | 0.90     | 1 min 46 s |

---

## ② 量化quantize

对 BERT 实施 `quantize_dynamic` **INT8 动态量化**：  
Applied quantize_dynamic INT8: weights quantized offline, activations remain FP32 with per-forward conversion overhead.

- 权重离线量化为 INT8  
- 激活值推理时仍为 FP32，每次前向传播存在 FP32 ↔ INT8 转换开销  
- 量化效果：准确率 90.21% → 90.02%，推理耗时从 1 min 46 s 劣化至 6 min 41 s  

**原因分析：**  

- `quantize_dynamic` 走 QNNPACK 后端，只能命中 CPU  
- M1 的 Neural Engine（ANE）未开放编程接口，无法利用  
- INT8 计算单元无法命中 → 推理加速依赖可调度专用算力（如 NVIDIA Tensor Core 或 Core ML 命中 ANE）
- qnnpack only reaches CPU; M1 ANE has no open developer API, so INT8 units are unreachable. Confirmed quantization speedup requires platforms with dedicated compute (NVIDIA Tensor Core or Core ML ANE).

---

## ③ 知识蒸馏（TextCNN 学生 / BERT 教师）
Knowledge Distillation — TextCNN Student / BERT Teacher


构建 BERT-teacher → TextCNN-student 蒸馏管线，使用 KL 散度软标签损失替代硬标签交叉熵。  
Built BERT-teacher → TextCNN-student pipeline using KL-divergence soft-label loss.


| model                  | Accuracy | macro P | macro R | macro F1 | infer time cost（79 batch） | size |
| --------------------- | -------- | ------- | ------- | -------- | ------------------ | -------- |
| BERT FP32（teacher）      | 90.21%   | 0.90    | 0.90    | 0.90     | 1 min 46 s         | 409.2 MB |
| BERT INT8 quantize         | 90.02%   | 0.90    | 0.90    | 0.90     | 6 min 41 s         | 152.6 MB |
| TextCNN（student）        | 90.49%   | 0.91    | 0.90    | 0.90     | 6 s                | 6.3 MB   |
