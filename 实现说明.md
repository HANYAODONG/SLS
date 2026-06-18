# 实现说明

> 如果后续项目组决定弃用算法创新方案、暂不训练 hybrid 模型，优先阅读 `创新实验复原文档.md`。该文档说明了如何保留实验文件但回到原 SLS 稳定复现主线，以及如何删除或通过 Git 回退本轮新增的实验性文件。

本文档记录当前项目在原 SLS 主模型复现基础上的创新改进实现情况。当前版本参考了三个方案文档，并将模型结构创新以“并列实验线”的方式接入，避免影响原始复现主线。

## 一、参考文档

本轮创新改进参考以下三个文档：

```text
创新实现一(1).pdf
时间注意力改进实现二_可执行版.md
深度伪造语音检测作品实现过程.md
```

三个文档对应的方向分别是：

1. `创新实现一(1).pdf`

   提出“统计感知的层级、特征、时间三级门控网络”，核心包括：

   - Mean+Std 统计感知 SLS；
   - SwiGLU 特征门控；
   - 时间注意力池化；
   - 消融实验和权重可视化。

2. `时间注意力改进实现二_可执行版.md`

   提出在 SLS 后端加入：

   - 普通时间注意力；
   - 通道引导时间注意力；
   - 可切换 pooling 类型；
   - 独立实验模型文件；
   - 训练和评估脚本。

3. `深度伪造语音检测作品实现过程.md`

   提出横向应用扩展，包括：

   - 视频音画分离；
   - Whisper ASR；
   - WeSpeaker 声纹识别；
   - VLM 图像审计；
   - LLM 风险报告；
   - 证据固化和时间戳。

## 二、当前实现原则

由于本项目已经完成原 SLS 主模型复现，且原始权重 `MMpaper_model.pth` 对应的是旧模型结构，本轮改动采用隔离策略：

- 原 `model.py` 不修改；
- 原 `main.py` 不修改；
- 原评测脚本不修改；
- 新模型结构放入 `model_hybrid.py`；
- 新训练入口放入 `main_hybrid.py`；
- 训练脚本以 `scripts/train_hybrid_*.sh` 命名；
- 评测脚本以 `scripts/eval_hybrid_*.sh` 命名；
- 若后续不训练，可直接继续使用原主线。

这样既能把老师建议的模型创新先做上去，也不会破坏当前已经复现成功的稳定版本。

## 三、已完成的模型结构创新

### 3.1 Mean+Std 统计感知 SLS

新增文件：

```text
model_hybrid.py
```

新增类：

```text
StatisticalSLS
```

功能：

- 输入 XLS-R 多层隐藏特征 `[B, L, T, D]`；
- 对每一层计算时间维度上的均值；
- 可选计算标准差；
- 拼接为 `[mean, std]` 统计向量；
- 通过 MLP 预测每一层权重；
- 对 24 层特征加权求和，得到 `[B, T, D]`。

对应开关：

```bash
--use_stat_sls 1
--stat_sls_use_std 1
```

消融时可以关闭：

```bash
--use_stat_sls 0
```

关闭后使用原 SLS 的均值池化层权重逻辑。

### 3.2 SwiGLU 特征门控

新增类：

```text
SwiGLUGate
```

功能：

- 输入层融合后的 `[B, T, D]` 特征；
- 使用 `SiLU(Wg x) * Wv x` 做维度门控；
- 输出仍为 `[B, T, D]`；
- 使用残差连接保持训练稳定。

对应开关：

```bash
--use_swiglu 1
```

关闭方式：

```bash
--use_swiglu 0
```

### 3.3 普通时间注意力池化

新增类：

```text
TemporalAttentionPooling
```

功能：

- 输入 `[B, T, D]`；
- 通过 MLP 计算每一帧的注意力分数；
- softmax 得到时间权重；
- 加权求和得到 `[B, D]`；
- 接新的二分类头。

对应开关：

```bash
--pooling_type temporal
```

### 3.4 通道引导时间注意力

新增类：

```text
ChannelGuidedTemporalAttention
```

功能：

- 计算全局通道统计信息；
- 使用通道上下文引导时间注意力分数；
- 可选加入统计残差；
- 输出 `[B, D]`；
- 接新的二分类头。

对应开关：

```bash
--pooling_type cgta
--cgta_use_std 1
--cgta_stat_residual 1
```

当前完整 hybrid 模型默认使用该方式。

### 3.5 原 head 保留

为了方便消融，`model_hybrid.py` 中保留了原始 SLS 后端：

```text
original_sls_fusion()
original_head()
```

当使用：

```bash
--pooling_type maxpool
```

时，模型仍走原始 `BatchNorm2d + max_pool2d + fc1 + fc3` 路径。

这使得以下消融成为可能：

- 只加 Mean+Std SLS；
- 只加 SwiGLU；
- 不加时间注意力；
- 对比原 head 和 attention head。

## 四、已新增文件清单

### 4.1 模型与入口

```text
model_hybrid.py
main_hybrid.py
test_hybrid_modules.py
```

说明：

- `model_hybrid.py`：hybrid 创新模型；
- `main_hybrid.py`：hybrid 训练与评测入口；
- `test_hybrid_modules.py`：模块级 shape 测试。

### 4.2 测试脚本

```text
scripts/test_hybrid_modules.sh
```

运行：

```bash
bash scripts/test_hybrid_modules.sh
```

用于确认：

- `StatisticalSLS` 输出维度正确；
- `SwiGLUGate` 输出维度正确；
- `TemporalAttentionPooling` 权重归一化；
- `ChannelGuidedTemporalAttention` 权重归一化。

### 4.3 训练脚本

```text
scripts/train_hybrid_stat_sls.sh
scripts/train_hybrid_swiglu.sh
scripts/train_hybrid_temporal.sh
scripts/train_hybrid_full.sh
```

对应实验：

```text
train_hybrid_stat_sls.sh   只加 Mean+Std 统计感知 SLS
train_hybrid_swiglu.sh     只加 SwiGLU 特征门控
train_hybrid_temporal.sh   只加普通时间注意力
train_hybrid_full.sh       Mean+Std + SwiGLU + CGTA 完整模型
```

默认参数：

```text
EPOCHS=10
BATCH_SIZE=1
EARLY_STOP_PATIENCE=3
NUM_WORKERS=2
```

可通过环境变量覆盖，例如：

```bash
EPOCHS=3 BATCH_SIZE=1 bash scripts/train_hybrid_full.sh
```

### 4.4 评测脚本

```text
scripts/eval_hybrid_df_20000.sh
scripts/eval_hybrid_wild_20000.sh
```

这两个脚本需要指定训练后的 checkpoint：

```bash
MODEL_PATH=models/.../best.pth bash scripts/eval_hybrid_df_20000.sh
MODEL_PATH=models/.../best.pth bash scripts/eval_hybrid_wild_20000.sh
```

注意：

```text
必须使用由 main_hybrid.py 训练得到的权重。
不能直接把 MMpaper_model.pth 当作 hybrid 的最终有效权重。
```

## 五、训练必要性说明

本轮新增的结构包括：

- `stat_sls.weight_predictor.*`
- `swiglu.*`
- `temporal_attention.*`
- `cgta_pooling.*`
- `hybrid_classifier.*`

这些参数不存在于原始 `MMpaper_model.pth` 中。

因此：

- 新结构代码可以先接入；
- 模块 shape 可以先测试；
- 但有效实验结果必须重新训练；
- 如果不训练，新增层只能随机初始化，不能证明创新有效。

`main_hybrid.py` 支持：

```bash
--load_strict 0
```

该参数只适合调试或部分加载旧权重，不适合作为正式实验结论。

## 六、建议实验安排

最低可展示版本：

1. 保留当前原 SLS 结果作为 baseline；
2. 跑 `train_hybrid_stat_sls.sh`；
3. 跑 `train_hybrid_swiglu.sh`；
4. 跑 `train_hybrid_temporal.sh`；
5. 跑 `train_hybrid_full.sh`；
6. 每组先跑 3 epoch，确认 loss 下降；
7. 选择表现较好的组继续跑到 10 epoch 左右；
8. 在 DF 20000 和 In-the-Wild 20000 上评测。

建议命令：

```bash
bash scripts/test_hybrid_modules.sh
EPOCHS=3 BATCH_SIZE=1 bash scripts/train_hybrid_full.sh
```

训练完成后评测：

```bash
MODEL_PATH=models/<hybrid实验目录>/best.pth bash scripts/eval_hybrid_df_20000.sh
python evaluate_2021_DF.py scores/scores_Hybrid_DF_20000.txt ./keys eval

MODEL_PATH=models/<hybrid实验目录>/best.pth bash scripts/eval_hybrid_wild_20000.sh
python evaluate_in_the_wild.py scores/scores_Hybrid_Wild_20000.txt ./keys eval
```

## 七、横向扩展实现状态

`深度伪造语音检测作品实现过程.md` 中提出的应用层扩展，当前已放在独立目录：

```text
extensions/
extension_audit.py
```

已实现能力包括：

- FFmpeg 视频音画分离；
- Whisper ASR 适配；
- WeSpeaker 声纹注册和匹配适配；
- OpenAI-compatible VLM 审计接口；
- 本地模拟 TSA 时间戳存证；
- 多路风险报告汇总；
- 高风险样本清单导出。

这些扩展不改变主模型结构，可与原模型或 hybrid 模型评测结果并行使用。

## 八、复原方案

本次新增了专门的复原说明：

```text
创新实验复原文档.md
```

如果后续不进行训练，可以采用推荐方案：

```text
保留 hybrid 实验文件，但运行时只使用 main.py 和原评测脚本。
```

原主线仍然是：

```bash
bash scripts/eval_df_20000.sh
bash scripts/eval_wild_20000.sh
```

这保证了项目可以在“创新实验线”和“稳定复现线”之间快速切换。
