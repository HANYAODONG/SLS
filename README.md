# SLS 音频深度伪造检测复现与扩展

本仓库基于 `SLSforASVspoof-2021-DF`，用于完成 SLS 主模型复现、ASVspoof 2021 DF / In-the-Wild 数据集测试，以及不改动模型结构的横向扩展功能验证。

当前项目原则：

- 不修改 `model.py` 中的主模型结构；
- 使用原始预训练权重 `MMpaper_model.pth`；
- 评测优先统一使用 20000 条子集；
- 大文件数据集、模型权重、API Key 不上传 GitHub；
- 扩展功能独立于算法层，便于后续回滚。

## 一、队友快速部署

### 1. 克隆仓库

```bash
git clone https://github.com/HANYAODONG/SLS.git
cd SLS
```

### 2. 创建并激活环境

推荐使用 Python 3.10：

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

安装主模型依赖：

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

解压并安装 fairseq：

```bash
unzip -n fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1.zip
pip install --editable ./fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
```

如果本机没有 CUDA，也可以先用 CPU 检查脚本，但正式跑 20000 条建议使用 GPU。

### 3. 准备模型权重

以下文件不在 GitHub 中，需要自行放到项目根目录：

```text
MMpaper_model.pth
xlsr2_300m.pt
```

`xlsr2_300m.pt` 可从 fairseq XLS-R 官方地址下载：

```bash
wget -O xlsr2_300m.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt
```

`MMpaper_model.pth` 使用项目组内部共享的原始 SLS 权重。

### 4. 准备数据集

DF 数据集放置为：

```text
data/ASVspoof2021_DF_eval/flac/
```

In-the-Wild 数据集解压为：

```text
release_in_the_wild/
```

这些目录不上传 GitHub，需要本地自行准备。

## 二、环境检查

激活环境后运行：

```bash
source venv/bin/activate
python - <<'PY'
import torch, torchaudio, google.protobuf
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("protobuf:", google.protobuf.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

当前主环境应保持：

```text
torch==1.12.1+cu113
torchaudio==0.12.1+cu113
protobuf==3.20.3
```

不要在主 `venv` 中直接安装 WeSpeaker 或 pymilvus，它们会拉高 `torch/protobuf` 版本，可能破坏主模型复现环境。

## 三、20000 条数据集评测

本项目组统一优先跑 20000 条。当前已准备好 DF 和 In-the-Wild 的 20000 条协议与 key。

### 1. DF 20000

运行：

```bash
source venv/bin/activate
bash scripts/eval_df_20000.sh
bash scripts/eer_df_20000.sh
```

输出：

```text
scores/scores_DF_20000.txt
```

当前本地已跑结果：

```text
EER = 1.99%
```

### 2. In-the-Wild 20000

运行：

```bash
source venv/bin/activate
bash scripts/eval_wild_20000.sh
bash scripts/eer_wild_20000.sh
```

输出：

```text
scores/scores_Wild_20000.txt
```

当前本地已跑结果：

```text
EER ≈ 7.47%
```

### 3. LA 20000 可选说明

仓库中保留了 LA 评测入口和原始协议文件，但当前本地没有完整整理好的 LA 20000 key 子集与本地 LA eval 音频目录。

如果队友需要补跑 LA 20000，需要准备：

```text
ASVspoof2021_LA_eval/flac/
ASVspoof2021 LA 对应官方 key / metadata
```

然后仿照 DF/Wild 生成：

```text
database/...LA.first20000...
keys/LA_20000/...
scores/scores_LA_20000.txt
```

本轮交付的可直接运行 20000 条脚本是：

```text
DF 20000
In-the-Wild 20000
```

## 四、新增横向扩展功能

新增功能不改变主模型，只在工程层扩展：

- 视频音画分离；
- Whisper ASR 转写；
- 文本风险审计；
- WeSpeaker 声纹注册与匹配；
- VLM 图像审计接口；
- 本地模拟 TSA 时间戳存证；
- DeepSeek 综合风险报告；
- 高风险样本清单导出。

相关入口：

```text
extension_audit.py
extensions/
scripts/check_extension_deps.sh
scripts/preprocess_video_demo.sh
scripts/certify_file_demo.sh
scripts/risk_wild_20000.sh
```

### 1. 检查扩展依赖

```bash
bash scripts/check_extension_deps.sh
```

理想状态：

```text
[ok] command: ffmpeg
[ok] command: ffprobe
[ok] python fallback: imageio_ffmpeg
[missing] python module: wespeaker
[ok] speaker env module: wespeaker (venv_speaker/bin/python)
[ok] python module: whisper
[missing] python module: pymilvus
```

说明：

- 主 `venv` 中 `wespeaker` missing 是正常的；
- WeSpeaker 应通过 `venv_speaker` 隔离环境使用；
- pymilvus 当前不启用，小规模声纹库用 JSONL 代替。

### 2. 安装扩展依赖

主环境可安装：

```bash
source venv/bin/activate
pip install -r requirements-extensions.txt
```

系统 FFmpeg：

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

WeSpeaker 隔离环境：

```bash
bash scripts/setup_speaker_env.sh
```

不要在主 `venv` 中运行：

```bash
pip install git+https://github.com/wenet-e2e/wespeaker.git
pip install pymilvus
```

### 3. 视频音画分离

队友准备：

```text
samples/video/demo_video.mp4
```

运行：

```bash
CASE_ID=demo_video bash scripts/preprocess_video_demo.sh samples/video/demo_video.mp4
```

输出：

```text
artifacts/audit/demo_video/audio.wav
artifacts/audit/demo_video/frames/
artifacts/audit/demo_video/preprocess.json
```

### 4. Whisper ASR

运行：

```bash
venv/bin/python extension_audit.py asr \
  --audio artifacts/audit/demo_video/audio.wav \
  --model-name base \
  --output artifacts/audit/demo_video/asr.json
```

第一次运行可能下载 Whisper 模型权重。

### 5. 文本风险审计

需要 `.env` 中配置 DeepSeek：

```text
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

运行：

```bash
venv/bin/python extension_audit.py audit-text \
  --text-file samples/text/demo_transcript.txt \
  --output artifacts/audit/demo_video/text_audit.json
```

### 6. 声纹注册与匹配

注册：

```bash
SPEAKER_PYTHON=venv_speaker/bin/python \
venv/bin/python extension_audit.py enroll-speaker \
  --name "target_speaker" \
  --audio samples/speaker/enroll/target_clean.wav \
  --enrollment artifacts/speaker/enrollment.jsonl
```

匹配：

```bash
SPEAKER_PYTHON=venv_speaker/bin/python \
venv/bin/python extension_audit.py match-speaker \
  --audio samples/speaker/query/query_same.wav \
  --enrollment artifacts/speaker/enrollment.jsonl
```

### 7. VLM 图像审计

需要额外准备支持图片输入的 VLM API：

```text
VLM_BASE_URL
VLM_API_KEY
VLM_MODEL
```

运行：

```bash
VLM_BASE_URL="https://your-vlm-endpoint/v1" \
VLM_API_KEY="your_vlm_key" \
VLM_MODEL="your_vision_model" \
venv/bin/python extension_audit.py audit-image \
  --image samples/images/frame_public_person.jpg \
  --output artifacts/audit/demo_video/frame_audit.json
```

### 8. 本地模拟存证

```bash
bash scripts/certify_file_demo.sh artifacts/audit/demo_video/audio.wav
```

输出：

```text
artifacts/audit/demo_case/timestamp.json
```

说明：当前是 `local_simulation`，用于项目演示，不等价于正式法律 TSA 证书。

### 9. 综合风险报告

```bash
venv/bin/python extension_audit.py report \
  --input samples/audit_payload/demo_payload.json \
  --output artifacts/audit/demo_video/final_report.json
```

## 五、样本收集交付

交给队友的数据收集文件：

```text
样本数据收集清单.md
```

部署和验证说明文件：

```text
扩展依赖安装与验证清单.md
```

综合说明文件：

```text
实现说明.md
```

建议队友收集目录：

```text
samples/
  video/
    demo_video.mp4
  text/
    demo_transcript.txt
  speaker/
    enroll/
      target_clean.wav
    query/
      query_same.wav
      query_other.wav
  images/
    frame_public_person.jpg
    frame_normal_scene.jpg
  audio/
    real_demo.wav
    fake_demo.wav
  audit_payload/
    demo_payload.json
```

`samples/` 不建议上传 GitHub。

## 六、网页助手

配置 `.env`：

```text
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

启动：

```bash
source venv/bin/activate
bash scripts/run_llm_web.sh
```

打开：

```text
http://127.0.0.1:7860
```

如果端口占用，程序会自动切换，也可以手动指定：

```bash
PORT=7861 bash scripts/run_llm_web.sh
```

## 七、重要注意事项

- `.env` 不要上传 GitHub；
- `MMpaper_model.pth` 不要上传 GitHub；
- `xlsr2_300m.pt` 不要上传 GitHub；
- `data/`、`release_in_the_wild/`、`samples/` 不要上传 GitHub；
- 不要在主 `venv` 中安装 WeSpeaker 或 pymilvus；
- 如果主模型环境被破坏，优先检查 `torch/torchaudio/protobuf` 版本；
- 当前主模型复现结果以 DF 20000 和 In-the-Wild 20000 为主。

## 八、当前结果参考

```text
DF 20000: EER = 1.99%
In-the-Wild 20000: EER ≈ 7.47%
```

完整复现和扩展过程见：

```text
复现报告.md
实现说明.md
扩展依赖安装与验证清单.md
样本数据收集清单.md
```
