const buttons = Array.from(document.querySelectorAll(".tab-button"));
const panels = Array.from(document.querySelectorAll(".panel"));

function showPanel(name) {
  buttons.forEach((button) => {
    button.classList.toggle("active", button.dataset.panel === name);
  });
  panels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${name}-panel`);
  });
}

buttons.forEach((button) => {
  button.addEventListener("click", () => showPanel(button.dataset.panel));
});

showPanel(window.initialPanel || "predict");

async function refreshLog(logBox) {
  const url = logBox.dataset.statusUrl;
  if (!url) return;

  try {
    const response = await fetch(url);
    const payload = await response.json();
    logBox.textContent = payload.log || "";
    if (payload.running) {
      logBox.classList.add("running");
    } else {
      logBox.classList.remove("running");
    }
    if (url.includes("/status/evaluate")) {
      renderEvaluationArtifacts(payload.artifacts);
    }
  } catch {
    logBox.classList.remove("running");
  }
}

function renderEvaluationArtifacts(artifacts) {
  const results = document.querySelector("#evaluation-results");
  if (!results || !artifacts || !artifacts.available) return;

  const summary = document.querySelector("#evaluation-summary");
  const reportText = document.querySelector("#evaluation-report-text");
  const matrixImage = document.querySelector("#confusion-matrix-image");

  summary.innerHTML = "";
  for (let i = 0; i < 2; i++) {
    const item = artifacts.summary[i];
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `
      <span>${item.label}</span>
      <strong>${item.value}</strong>
      <p>${item.detail}</p>
    `;
    summary.appendChild(card);
  }

  reportText.textContent =
    artifacts.report_text || "No evaluation_report.txt was found yet.";

  if (artifacts.confusion_matrix_url) {
    matrixImage.src = artifacts.confusion_matrix_url;
    matrixImage.closest(".confusion-figure").hidden = false;
  } else {
    matrixImage.closest(".confusion-figure").hidden = true;
  }

  results.hidden = false;
}

const logBoxes = Array.from(document.querySelectorAll(".log-box"));
logBoxes.forEach((logBox) => refreshLog(logBox));
setInterval(() => {
  logBoxes.forEach((logBox) => refreshLog(logBox));
}, 2500);
