const experimentSelect = document.getElementById("experiment");
const summaryEl = document.getElementById("summary");
const questionEl = document.getElementById("question");
const answerEl = document.getElementById("answer");
const statusEl = document.getElementById("status");
const askButton = document.getElementById("ask");
const startRecordingButton = document.getElementById("start-recording");
const stopRecordingButton = document.getElementById("stop-recording");
const analyzeAudioButton = document.getElementById("analyze-audio");
const audioFileInput = document.getElementById("audio-file");
const audioPreview = document.getElementById("audio-preview");
const audioStatus = document.getElementById("audio-status");
const audioResult = document.getElementById("audio-result");
const voiceSimilarityInput = document.getElementById("voice-similarity");

let mediaRecorder = null;
let recordedChunks = [];
let selectedAudioBlob = null;
let selectedAudioName = "recording.webm";

function fmt(value, digits = 2) {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "number") return value.toFixed(digits);
  return value;
}

function metric(label, value) {
  return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

async function loadSummary() {
  summaryEl.textContent = "加载中...";
  const exp = experimentSelect.value;
  try {
    const response = await fetch(`/api/summary?experiment=${encodeURIComponent(exp)}`);
    const data = await response.json();
    if (!response.ok) {
      summaryEl.textContent = data.error || "加载失败";
      return;
    }
    const dist = data.score_distribution || {};
    const metrics = [
      metric("实验", data.name),
      metric("EER", `${fmt(data.eer_percent)}%`),
      metric("score 行数", data.num_scores),
      metric("eval 样本", data.num_phase_rows),
      metric("eval spoof", (data.label_counts_phase || {}).spoof || 0),
      metric("eval bonafide", (data.label_counts_phase || {}).bonafide || 0),
      metric("score 均值", fmt(dist.mean, 4)),
    ];
    if (data.reported_full_df_eer !== undefined) {
      metrics.push(metric("仓库完整 DF EER", `${fmt(data.reported_full_df_eer)}%`));
    }
    summaryEl.innerHTML = metrics.join("");
  } catch (error) {
    summaryEl.innerHTML = [
      metric("服务状态", "未连接"),
      metric("启动方式", "bash scripts/run_llm_web.sh"),
      metric("错误", error.message),
      metric("提示", "不要直接双击打开 HTML"),
    ].join("");
  }
}

async function ask(question) {
  const finalQuestion = question || questionEl.value.trim();
  if (!finalQuestion) return;
  questionEl.value = finalQuestion;
  statusEl.textContent = "请求 DeepSeek 中...";
  answerEl.textContent = "";
  askButton.disabled = true;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        experiment: experimentSelect.value,
        question: finalQuestion,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "请求失败");
    }
    answerEl.textContent = data.answer;
    statusEl.textContent = "完成";
  } catch (error) {
    answerEl.textContent = `出错：${error.message}`;
    statusEl.textContent = "失败";
  } finally {
    askButton.disabled = false;
  }
}

experimentSelect.addEventListener("change", loadSummary);
askButton.addEventListener("click", () => ask());
document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => ask(button.dataset.question));
});

function setAudioBlob(blob, name) {
  selectedAudioBlob = blob;
  selectedAudioName = name;
  audioPreview.src = URL.createObjectURL(blob);
  analyzeAudioButton.disabled = false;
  audioStatus.textContent = `已选择音频：${name}，大小 ${(blob.size / 1024).toFixed(1)} KB`;
}

function renderAudioResult(data) {
  const decision = data.decision || {};
  const input = data.input_audio || {};
  const processed = data.processed_audio || {};
  const processing = data.processing || {};
  const threshold = data.threshold_reference || {};
  const probabilities = data.probabilities || {};
  const joint = data.joint_decision || {};
  const speaker = data.speaker_match || {};
  const speakerBest = speaker.best || {};
  const cards = [
    metric("判别倾向", decision.label || "N/A"),
    metric("风险等级", decision.risk_level || "N/A"),
    metric("模型分数", fmt(data.score, 4)),
    metric("伪造概率", fmt(probabilities.fake_probability, 4)),
    metric("阈值 margin", fmt(decision.margin, 4)),
    metric("参考 EER", threshold.eer_percent === undefined ? "N/A" : `${fmt(threshold.eer_percent)}%`),
    metric("联合象限", joint.quadrant || "未启用"),
    metric("联合风险", joint.risk_level || joint.fake_only_level || "N/A"),
    metric("声纹对象", speakerBest.name || "N/A"),
    metric("声纹相似度", fmt(joint.raw_voice_similarity ?? speakerBest.similarity, 4)),
    metric("音频时长", processed.duration_seconds === undefined ? "N/A" : `${fmt(processed.duration_seconds)}s`),
    metric("采样率", processed.sample_rate || "N/A"),
    metric("推理设备", (data.model || {}).device || "N/A"),
  ];
  function shortText(value, keep = 12) {
    if (!value) return "N/A";
    if (value.length <= keep * 2 + 3) return value;
    return `${value.slice(0, keep)}...${value.slice(-keep)}`;
  }

  function basename(path) {
    if (!path) return "N/A";
    return path.split(/[\\/]/).pop();
  }

  const detail = {
    sample_id: data.sample_id,
    original_filename: data.original_filename,
    input_sha256_short: shortText(input.sha256),
    processed_sha256_short: shortText(processed.sha256),
    processed_file: basename(processed.path),
    joint_decision: joint,
    speaker_match_error: speaker.error,
    elapsed_seconds: fmt(processing.elapsed_seconds, 3),
    note: data.note,
  };
  audioResult.innerHTML = `<div class="summary result-summary">${cards.join("")}</div><code>${JSON.stringify(detail, null, 2)}</code>`;
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    });
    mediaRecorder.addEventListener("stop", () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      setAudioBlob(blob, `recording-${Date.now()}.webm`);
      startRecordingButton.disabled = false;
      stopRecordingButton.disabled = true;
    });
    mediaRecorder.start();
    startRecordingButton.disabled = true;
    stopRecordingButton.disabled = false;
    analyzeAudioButton.disabled = true;
    audioStatus.textContent = "录音中...";
    audioResult.textContent = "暂无单条语音分析结果。";
  } catch (error) {
    audioStatus.textContent = `无法开始录音：${error.message}`;
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
}

async function analyzeAudio() {
  if (!selectedAudioBlob) return;
  analyzeAudioButton.disabled = true;
  audioStatus.textContent = "正在上传并分析...";
  audioResult.textContent = "";
  try {
    const formData = new FormData();
    formData.append("audio", selectedAudioBlob, selectedAudioName);
    const voiceSimilarity = voiceSimilarityInput.value.trim();
    if (voiceSimilarity) {
      formData.append("voice_similarity", voiceSimilarity);
    }
    const response = await fetch("/api/analyze-recording", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "分析失败");
    }
    renderAudioResult(data);
    audioStatus.textContent = "单条语音分析完成";
  } catch (error) {
    audioResult.textContent = `出错：${error.message}`;
    audioStatus.textContent = "分析失败";
  } finally {
    analyzeAudioButton.disabled = false;
  }
}

startRecordingButton.addEventListener("click", startRecording);
stopRecordingButton.addEventListener("click", stopRecording);
analyzeAudioButton.addEventListener("click", analyzeAudio);
audioFileInput.addEventListener("change", () => {
  const file = audioFileInput.files && audioFileInput.files[0];
  if (file) {
    setAudioBlob(file, file.name);
    audioResult.textContent = "暂无单条语音分析结果。";
  }
});

loadSummary();
