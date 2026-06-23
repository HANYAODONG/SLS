# 实现说明

> 如果后续项目组决定弃用算法创新方案、暂不训练 hybrid 模型，优先阅读 `创新实验复原文档.md`。该文档说明了如何保留实验文件但回到原 SLS 稳定复现主线，以及如何删除或通过 Git 回退本轮新增的实验性文件。

本文档记录当前项目在原 SLS 主模型复现基础上的创新改进实现情况。当前版本参考了多个方案文档，并将模型结构创新、横向扩展和零训练可解释性能力以隔离方式接入，避免影响原始复现主线。

## 一、参考文档

本轮创新改进参考以下文档：

```text
创新实现一(1).pdf
时间注意力改进实现二_可执行版.md
深度伪造语音检测作品实现过程.md
联合判别矩阵实现说明.md
XLS-R_SLS_大模型可解释性零训练可执行方案(1).md
```

这些文档对应的方向分别是：

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

4. `联合判别矩阵实现说明.md`

   提出将声纹相似度和伪造概率进行二维融合，输出联合风险象限和风险等级。

5. `XLS-R_SLS_大模型可解释性零训练可执行方案(1).md`

   提出在不重新训练、不改变 checkpoint 的前提下，通过层权重、遮挡分析、频带扰动、稳定性分析和模板/大模型报告生成结构化可解释证据。

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

### 7.1 联合判别矩阵

新增参考文档：

```text
联合判别矩阵实现说明.md
```

当前已将“声纹相似度 × 伪造概率”的二维联合判别矩阵作为后处理模块接入：

```text
analysis/joint_decision_matrix.py
extension_audit.py joint-risk
web_app.py /api/analyze-recording
web/index.html 单条语音分析
```

实现方式：

- 不修改 `model.py`；
- 不修改 `model_hybrid.py`；
- 不重新训练模型；
- 将 SLS 输出的 `fake_probability` 与可选的声纹相似度融合；
- 输出联合象限、联合风险等级和解释文本。

网页端当前支持两种方式：

1. 手动输入声纹相似度，用于演示联合判别矩阵；
2. 如果服务端配置了 `SPEAKER_ENROLLMENT`，则自动调用 WeSpeaker 原声库进行声纹匹配。

命令行示例：

```bash
python extension_audit.py joint-risk \
  --voice-similarity 0.86 \
  --fake-probability 0.92
```

示例输出为：

```text
Q2 / 高危：声纹高度匹配目标人物，且检测为 AI 合成语音。
```

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

## 九、零训练可解释性模块

新增参考文档：

```text
XLS-R_SLS_大模型可解释性零训练可执行方案(1).md
```

当前已完成第一阶段零训练可解释性功能，新增目录和脚本：

```text
configs/explainability.json
configs/explainability_cpu_debug.json
explainability/
scripts/inspect_model_output.py
scripts/compare_original_output.py
scripts/explain_audio.py
```

实现原则：

- 不修改 `model.py`；
- 不修改 `main.py`；
- 不新增可训练参数；
- 不重新训练；
- 不改变 `MMpaper_model.pth` 的 state_dict；
- 使用原模型输出、SLS 层权重和扰动前后分数变化生成解释证据。

已实现能力：

- 原始整段预测；
- SLS 层权重导出；
- 时间遮挡分析；
- 平滑频带遮挡；
- 扰动稳定性分析；
- 结构化 `evidence.json`；
- 固定模板中文报告；
- 报告事实校验与限制性表述。

暂缓能力：

- 修改 `model.py` 内部 forward 以支持真正的归一化层遮挡；
- 原型库与 OOD 检测；
- Ollama 本地大模型改写。

暂缓原因：

```text
当前项目已经有稳定复现主线，为避免破坏原 checkpoint 兼容性，本轮采用只读适配器 SLSModelAdapter 获取中间证据，不直接修改 model.py。
```

验证结果：

```text
hidden_states_shape = [1, 24, 201, 1024]
fused_sequence_shape = [1, 201, 1024]
layer_weights_shape = [24]
fake_class_index = 0
compare_original_output max_abs_difference = 0.0
```

示例命令：

```bash
python scripts/inspect_model_output.py \
  --audio release_in_the_wild/0.wav \
  --checkpoint MMpaper_model.pth \
  --xlsr-checkpoint xlsr2_300m.pt \
  --device cpu

python scripts/compare_original_output.py \
  --audio release_in_the_wild/0.wav \
  --checkpoint MMpaper_model.pth \
  --xlsr-checkpoint xlsr2_300m.pt \
  --device cpu

python scripts/explain_audio.py \
  --audio release_in_the_wild/0.wav \
  --checkpoint MMpaper_model.pth \
  --xlsr-checkpoint xlsr2_300m.pt \
  --config configs/explainability_cpu_debug.json \
  --output-dir artifacts/reports \
  --device cpu
```

示例输出：

```text
artifacts/reports/0_evidence.json
artifacts/reports/0_report.txt
```

注意：

```text
报告中的时间片段、频带和层权重表示“模型输出对干预的敏感性”，不能写成真实伪造位置、具体生成器来源或司法鉴定结论。
```
