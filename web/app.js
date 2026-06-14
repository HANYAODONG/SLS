const experimentSelect = document.getElementById("experiment");
const summaryEl = document.getElementById("summary");
const questionEl = document.getElementById("question");
const answerEl = document.getElementById("answer");
const statusEl = document.getElementById("status");
const askButton = document.getElementById("ask");

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
  const response = await fetch(`/api/summary?experiment=${encodeURIComponent(exp)}`);
  const data = await response.json();
  if (!response.ok) {
    summaryEl.textContent = data.error || "加载失败";
    return;
  }
  const dist = data.score_distribution || {};
  summaryEl.innerHTML = [
    metric("实验", data.name),
    metric("EER", `${fmt(data.eer_percent)}%`),
    metric("score 行数", data.num_scores),
    metric("eval 样本", data.num_phase_rows),
    metric("eval spoof", (data.label_counts_phase || {}).spoof || 0),
    metric("eval bonafide", (data.label_counts_phase || {}).bonafide || 0),
    metric("score 均值", fmt(dist.mean, 4)),
    metric("仓库完整 DF EER", `${fmt(data.reported_full_df_eer)}%`),
  ].join("");
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

loadSummary();
