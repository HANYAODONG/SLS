# XLS-R + SLS 大模型可解释性零训练可执行方案

## 1. 目标与硬约束

本方案用于现有 `SLSforASVspoof-2021-DF` 项目，在不重新训练主模型的前提下实现可解释性报告。

硬约束：

- 不修改 XLS-R 前端结构；
- 不新增可训练参数；
- 不改变原 checkpoint 的 `state_dict` 键名和张量形状；
- 不重新训练；
- 原训练、评估和推理入口保持可用；
- 兼容 Python 3.7、PyTorch 1.12.1；
- CPU 模式可执行；
- 大模型只负责将结构化证据改写成自然语言，不参与真假判定。

最终功能：

1. 原始整段检测；
2. SLS 层权重导出；
3. 时间遮挡分析；
4. 归一化层遮挡分析；
5. 平滑频带遮挡分析；
6. 扰动稳定性分析；
7. 可选原型距离和 OOD 检测；
8. 结构化证据 JSON；
9. 固定模板或本地大模型报告；
10. 报告事实校验与自动回退。


## 2. 推荐实施顺序

### 第一阶段

```text
原始整段预测
→ SLS层权重导出
→ 时间遮挡
→ 归一化层遮挡
→ 扰动稳定性
→ 结构化JSON
→ 固定模板报告
```

### 第二阶段

```text
平滑频带遮挡
→ 开发集原型库
→ OOD检测
→ Ollama大模型改写
→ 报告事实校验
```

暂时不把短窗口独立分类作为核心解释证据，除非确认原模型真正支持变长输入。若原模型会把短片段重复到固定长度，短窗结果可能混入重复伪影。


## 3. 项目目录

```text
SLSforASVspoof-2021-DF/
├── model.py
├── main.py
├── explainability/
│   ├── __init__.py
│   ├── model_adapter.py
│   ├── audio_utils.py
│   ├── occlusion.py
│   ├── layer_ablation.py
│   ├── frequency_ablation.py
│   ├── stability.py
│   ├── prototype_bank.py
│   ├── evidence_builder.py
│   ├── llm_report.py
│   └── validator.py
├── scripts/
│   ├── inspect_model_output.py
│   ├── compare_original_output.py
│   ├── build_prototype_bank.py
│   └── explain_audio.py
├── configs/
│   └── explainability.json
├── artifacts/
│   ├── prototype_bank/
│   ├── reports/
│   └── cache/
└── requirements_explain.txt
```

Linux：

```bash
mkdir -p explainability scripts configs
mkdir -p artifacts/prototype_bank artifacts/reports artifacts/cache
touch explainability/__init__.py
```

Windows CMD：

```bat
mkdir explainability
mkdir scripts
mkdir configs
mkdir artifacts
mkdir artifacts\prototype_bank
mkdir artifacts\reports
mkdir artifacts\cache
type nul > explainability\__init__.py
```


## 4. 附加依赖

创建 `requirements_explain.txt`：

```text
numpy==1.21.6
scipy==1.7.3
scikit-learn==1.0.2
librosa==0.9.2
soundfile==0.12.1
matplotlib==3.5.3
requests==2.31.0
tqdm==4.66.1
```

安装：

```bash
conda activate SLS
pip install -r requirements_explain.txt
```


## 5. 配置文件

创建 `configs/explainability.json`：

```json
{
  "sample_rate": 16000,
  "fake_class_index": 1,
  "time_occlusion_window": 0.5,
  "time_occlusion_hop": 0.25,
  "time_occlusion_modes": ["zero", "noise"],
  "frequency_bands": [
    [0, 1000],
    [1000, 2000],
    [2000, 4000],
    [4000, 8000]
  ],
  "frequency_attenuation_db": -24.0,
  "top_layer_ablation": 6,
  "top_segments": 5,
  "top_frequency_bands": 3,
  "stability_tests": [
    "volume",
    "noise",
    "resample",
    "mp3_like",
    "trim"
  ],
  "stability_noise_snr_db": 30,
  "prototype_enabled": false,
  "prototype_top_k": 5,
  "ood_percentile": 95,
  "llm_provider": "template",
  "ollama_url": "http://127.0.0.1:11434/api/generate",
  "ollama_model": "qwen2.5:3b",
  "llm_temperature": 0.1,
  "llm_timeout": 120,
  "output_language": "zh-CN"
}
```

第一轮保持：

```json
"prototype_enabled": false,
"llm_provider": "template"
```


## 6. 修改 `model.py`

### 6.1 修改原则

不得修改：

```text
XLS-R checkpoint读取
XLS-R前端结构
SLS已有参数
池化层参数
全连接分类层参数
state_dict键名
原forward默认输出
```

只增加：

```python
return_details=False
layer_mask=None
```

### 6.2 前向接口

根据原项目真实变量名适配：

```python
def forward(self, x, return_details=False, layer_mask=None):
    hidden_states = ...
    hidden_states = self._to_b_l_t_d(hidden_states)

    layer_weights = ...

    if layer_weights.dim() == 3:
        layer_weights = layer_weights.squeeze(-1)

    original_layer_weights = layer_weights

    if layer_mask is not None:
        if layer_mask.dim() == 1:
            layer_mask = layer_mask.unsqueeze(0)

        if torch.all(layer_mask == 0):
            raise ValueError("layer_mask cannot disable all layers")

        layer_mask = layer_mask.to(
            device=layer_weights.device,
            dtype=layer_weights.dtype,
        )

        masked_weights = layer_weights * layer_mask

        original_sum = layer_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)

        masked_sum = masked_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)

        layer_weights = masked_weights * (
            original_sum / masked_sum
        )

    weighted_hidden = (
        hidden_states
        * layer_weights.unsqueeze(-1).unsqueeze(-1)
    )

    fused_sequence = torch.sum(
        weighted_hidden,
        dim=1,
    )

    pooled_embedding = ...
    logits = ...

    if not return_details:
        return logits

    return {
        "logits": logits,
        "embedding": pooled_embedding,
        "hidden_states": hidden_states,
        "fused_sequence": fused_sequence,
        "layer_weights": original_layer_weights
    }
```

必须从原代码确认：

- `hidden_states` 的真实维度；
- `pooled_embedding` 对应的真实变量；
- 真伪标签顺序；
- `fake_class_index`；
- 原始固定输入长度；
- 是否支持变长输入。


## 7. 真实输出与一致性检查

创建 `scripts/inspect_model_output.py`，直接复用原 `main.py` 中：

- 参数解析；
- 模型初始化；
- checkpoint 加载；
- 音频预处理；
- 裁剪或补齐；
- 设备选择。

运行：

```bash
python scripts/inspect_model_output.py   --audio test.wav   --checkpoint weights/model.pth   --device cpu
```

期望输出：

```text
logits: (1, 2)
embedding: (1, D)
hidden_states: (1, L, T, D)
fused_sequence: (1, T, D)
layer_weights: (1, L)
fake_class_index: ...
fixed_input_length: ...
variable_length_supported: ...
```

创建 `scripts/compare_original_output.py`：

```python
with torch.no_grad():
    output_a = model(audio, return_details=False)
    output_b = model(audio, return_details=True)["logits"]

difference = torch.max(
    torch.abs(output_a - output_b)
).item()

print("max_abs_difference:", difference)
```

要求：

```text
CPU：小于1e-7
GPU：小于1e-5
```

随后比较修改前后完整评估集：

```text
分数文件一致
EER一致
标签顺序一致
```


## 8. 时间遮挡

时间遮挡不得改变总长度，同时使用两种替换方式：

```text
zero：目标片段置零
noise：目标片段替换为低幅随机噪声
```

每个区间分别计算：

```text
原始伪造分数
遮挡后伪造分数
分数下降值
两种遮挡方式的平均下降值
```

输出示例：

```json
{
  "start": 2.0,
  "end": 2.5,
  "average_probability_drop": 0.341,
  "modes": [
    {
      "mode": "zero",
      "masked_probability": 0.57,
      "probability_drop": 0.35
    },
    {
      "mode": "noise",
      "masked_probability": 0.59,
      "probability_drop": 0.33
    }
  ]
}
```

按 `average_probability_drop` 从高到低排序。


## 9. 归一化层遮挡

按 SLS 权重选取前 `top_layer_ablation` 层，逐层关闭。

层遮挡必须在 `model.py` 内保持遮挡前后的总权重尺度一致，禁止只置零不归一化。

输出：

```json
{
  "layer_number": 21,
  "sls_weight": 0.914,
  "ablated_probability": 0.641,
  "probability_drop": 0.287
}
```

禁止关闭全部层。


## 10. 平滑频带遮挡

禁止直接执行：

```python
spectrum[mask] = 0
```

使用平滑边缘衰减：

```text
核心频带：衰减24 dB
左右过渡带：线性或余弦渐变
总音频长度：保持不变
```

推荐频带：

```text
0–1 kHz
1–2 kHz
2–4 kHz
4–8 kHz
```

输出：

```json
{
  "low_hz": 2000,
  "high_hz": 4000,
  "attenuation_db": -24.0,
  "ablated_probability": 0.705,
  "probability_drop": 0.223
}
```


## 11. 扰动稳定性

扰动类型：

```text
轻微音量变化
30 dB信噪比加噪
重采样往返
近似压缩退化
轻微首尾裁剪后补零
```

要求：

- 扰动后总长度不变；
- 固定随机种子；
- 加噪至少运行两次取平均；
- 保存每个版本的伪造分数。

稳定性等级：

```text
range <= 0.08：high
0.08 < range <= 0.20：medium
range > 0.20：low
```

输出：

```json
{
  "scores": [],
  "mean": 0.901,
  "std": 0.031,
  "range": 0.082,
  "level": "medium"
}
```


## 12. 原型库与 OOD

第二阶段实现。

原型库使用独立开发集：

```text
真实音频500～2000条
伪造音频500～2000条
```

manifest：

```text
/path/to/real_001.wav 0
/path/to/real_002.wav 0
/path/to/fake_001.wav 1
/path/to/fake_002.wav 1
```

使用分类头前的 `pooled_embedding`，保存：

```text
embeddings.npy
labels.npy
metadata.json
```

OOD 阈值：

```text
独立开发集最近邻距离的95百分位
```

输出：

```json
{
  "real_center_similarity": 0.42,
  "fake_center_similarity": 0.81,
  "nearest_distance": 0.13,
  "is_ood": false,
  "ood_threshold": 0.21,
  "nearest_examples": []
}
```

只能表述为：

```text
“在模型特征空间中与某类样本较相似”
```

禁止表述为：

```text
“确定由某个生成器生成”
```


## 13. 结构化证据 JSON

最终 JSON：

```json
{
  "schema_version": "1.0",
  "audio": {
    "path": "test.wav",
    "sample_rate": 16000,
    "duration_seconds": 4.03
  },
  "decision": {
    "label": "fake",
    "fake_probability": 0.9284,
    "raw_logits": []
  },
  "layer_evidence": {
    "top_sls_layers": [],
    "layer_ablation": []
  },
  "temporal_evidence": {
    "top_occlusion_segments": []
  },
  "frequency_evidence": {
    "top_frequency_bands": []
  },
  "stability": {},
  "prototype_and_ood": null,
  "report_constraints": {
    "must_not_claim_generator_identity": true,
    "must_not_claim_causality": true,
    "must_include_ood_warning": false,
    "must_include_stability": true,
    "must_state_detection_limitations": true
  }
}
```


## 14. 固定模板报告

第一阶段只使用模板：

```text
检测结论：
当前模型将该音频判定为疑似伪造，伪造类别输出分数为0.9284。

关键证据：
遮挡2.00至2.50秒后，模型伪造分数平均下降0.3410。关闭XLS-R第21层贡献后，伪造分数下降0.2870。平滑衰减2000至4000Hz频带后，伪造分数下降0.2230。

稳定性：
在音量变化、轻微加噪、重采样和近似压缩处理下，分数波动范围为0.0820，稳定性为中等。

风险提示：
当前音频未触发明显的分布外警告，但仍建议结合音频来源和人工复核。

限制说明：
以上内容仅描述模型预测及干预实验造成的输出变化，不能单独证明音频的具体生成方式，也不能替代司法鉴定。
```


## 15. 本地大模型与事实校验

第二阶段使用独立 Ollama 服务：

```bash
ollama serve
ollama pull qwen2.5:3b
```

提示词必须限制：

```text
只能使用JSON中存在的证据。
禁止猜测具体生成器。
禁止把关注区域描述为已证实的伪造位置。
禁止把softmax分数描述为真实世界概率。
必须输出检测结论、关键证据、稳定性、风险提示、限制说明。
```

事实校验必须检查：

```text
是否出现“确定由某生成器生成”
是否出现“已经证明”
是否出现“百分之百”或“100%”
是否引用不存在的层号
是否引用不存在的时间段
是否遗漏稳定性
是否遗漏限制说明
OOD为true时是否遗漏分布外提示
```

未通过校验时自动回退固定模板。


## 16. 主执行流程

`scripts/explain_audio.py` 固定执行：

```text
1. 读取配置
2. 使用原项目预处理加载音频
3. 加载原模型和checkpoint
4. 原始整段预测
5. 导出SLS层权重
6. 时间遮挡
7. 归一化层遮挡
8. 平滑频带遮挡
9. 扰动稳定性
10. 可选原型/OOD
11. 构建evidence.json
12. 模板或Ollama生成报告
13. 事实校验
14. 保存结果
```

输出：

```text
artifacts/reports/<audio_name>_evidence.json
artifacts/reports/<audio_name>_report.txt
```


## 17. Codex执行指令

```text
请在当前 SLSforASVspoof-2021-DF 仓库内完成零训练可解释性功能。

硬约束：
1. 不修改 XLS-R 前端结构。
2. 不新增任何可训练参数。
3. 不改变原 checkpoint 的 state_dict 键和张量形状。
4. 不重新训练。
5. 原训练、评估和推理入口必须继续可用。
6. Python 3.7、PyTorch 1.12.1兼容。
7. 不使用 X | None，全部使用 Optional[X]。
8. CPU模式必须运行。
9. 所有代码必须通过 py_compile。
10. 不允许凭空假设张量维度、标签顺序和模型初始化参数。

任务：
1. 阅读 main.py、model.py、data_utils_SSL.py 和原评估代码。
2. 找出模型真实初始化方式、checkpoint字段、音频预处理和真假标签映射。
3. 在 model.py 的 forward 中增加 return_details=False 和 layer_mask=None。
4. return_details=True 时返回 logits、embedding、hidden_states、fused_sequence、layer_weights。
5. 运行真实样本并将 hidden_states 统一为 [B,L,T,D]。
6. 层遮挡后按原权重总和重新归一化。
7. return_details=False 时输出必须与修改前一致。
8. 新建 explainability 和 scripts 中全部文件。
9. 时间遮挡同时使用 zero 和 noise。
10. 频带遮挡使用平滑频率衰减，不允许硬置零。
11. 稳定性扰动后保持音频总长度不变。
12. 第一轮默认关闭 prototype 和 Ollama，只生成固定模板。
13. 完成单样本输出一致性和完整评估EER一致性测试。
14. 执行 python -m compileall explainability scripts。
15. 执行 inspect_model_output.py。
16. 执行 compare_original_output.py。
17. 执行 explain_audio.py。
18. 完成后输出修改文件列表、执行命令、中间张量真实形状、fake_class_index、最大输出误差、生成文件路径和问题修复结果。
```


## 18. 执行命令

编译：

```bash
python -m py_compile model.py
python -m compileall explainability scripts
```

检查输出：

```bash
python scripts/inspect_model_output.py   --audio test.wav   --checkpoint weights/model.pth   --device cpu
```

检查一致性：

```bash
python scripts/compare_original_output.py   --audio test.wav   --checkpoint weights/model.pth   --device cpu
```

生成解释：

```bash
python scripts/explain_audio.py   --audio test.wav   --checkpoint weights/model.pth   --config configs/explainability.json   --device cpu
```

第二阶段构建原型库：

```bash
python scripts/build_prototype_bank.py   --manifest prototype_manifest.txt   --checkpoint weights/model.pth   --output artifacts/prototype_bank   --device cpu
```


## 19. CPU调试配置

```json
{
  "time_occlusion_window": 1.0,
  "time_occlusion_hop": 1.0,
  "time_occlusion_modes": ["zero"],
  "top_layer_ablation": 3,
  "frequency_bands": [
    [0, 2000],
    [2000, 8000]
  ],
  "stability_tests": [
    "volume",
    "noise"
  ],
  "prototype_enabled": false,
  "llm_provider": "template"
}
```


## 20. 验收条件

```text
[ ] 原checkpoint strict=True加载成功
[ ] 原模型评估入口仍可执行
[ ] return_details=False与修改前输出一致
[ ] 完整评估集EER一致
[ ] fake_class_index已从原项目确认
[ ] hidden_states真实维度已确认并统一为[B,L,T,D]
[ ] layer_weights形状为[B,L]
[ ] 层遮挡后执行权重总和归一化
[ ] 禁止关闭全部层
[ ] 时间遮挡不改变音频长度
[ ] 时间遮挡验证zero和noise两种方式
[ ] 频带遮挡使用平滑衰减
[ ] 稳定性结果可重复
[ ] 无原型库时仍可执行
[ ] Ollama不可用时自动回退模板
[ ] LLM引用不存在证据时自动回退模板
[ ] evidence.json支持报告中的全部数字
```

可靠性边界：

```text
本方案输出模型决策依据、干预后的输出变化、稳定性和分布外风险。
不得作为真实伪造位置的因果证明、具体生成器识别或司法鉴定结论。
```
