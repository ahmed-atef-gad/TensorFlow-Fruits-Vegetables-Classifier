"""Local web interface for the fruits and vegetables CNN project."""

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from src.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DATASET_DIR,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    PLOTS_DIR,
    PROJECT_ROOT,
)


UPLOAD_DIR = PROJECT_ROOT / "uploads"
RUNS_DIR = PROJECT_ROOT / "web_runs"
TRAIN_LOG_PATH = RUNS_DIR / "training.log"
EVALUATE_LOG_PATH = RUNS_DIR / "evaluation.log"
EVALUATION_REPORT_PATH = MODELS_DIR / "evaluation_report.txt"
CONFUSION_MATRIX_IMAGE_PATH = PLOTS_DIR / "confusion_matrix_heatmap.png"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


@dataclass
class ManagedRun:
    process: subprocess.Popen[str] | None = None
    command: list[str] | None = None
    log_path: Path | None = None


training_run = ManagedRun()
evaluation_run = ManagedRun()


def default_settings() -> dict[str, Any]:
    return {
        "dataset_dir": str(DATASET_DIR),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "image_size": IMAGE_SIZE[0],
        "resume_from": "none",
        "workers": 2,
        "max_queue_size": 8,
        "model_path": str(BEST_MODEL_PATH),
    }


def read_form_value(name: str, default: Any, caster: type = str) -> Any:
    value = request.form.get(name, "").strip()
    if value == "":
        return default
    return caster(value)


def image_file_is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_metric(value: str | None) -> str:
    number = parse_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"


def format_support(value: str | None) -> str:
    number = parse_float(value)
    if number is None:
        return "-"
    return f"{int(round(number)):,}"


def evaluation_summary_from_report(report_text: str) -> list[dict[str, str]]:
    summary_cards: list[dict[str, str]] = []
    lines = [line.strip() for line in report_text.splitlines() if line.strip()]

    for line in lines:
        if line.startswith("Test loss:"):
            summary_cards.append(
                {
                    "label": "Test loss",
                    "value": line.split(":", 1)[1].strip(),
                    "detail": "Lower is better",
                }
            )
        elif line.startswith("Test accuracy:"):
            summary_cards.append(
                {
                    "label": "Test accuracy",
                    "value": format_metric(line.split(":", 1)[1].strip()),
                    "detail": "Overall test performance",
                }
            )
        elif line.startswith("macro avg"):
            parts = line.split()
            if len(parts) >= 6:
                summary_cards.append(
                    {
                        "label": "Macro F1",
                        "value": format_metric(parts[-2]),
                        "detail": "Unweighted class average",
                    }
                )
        elif line.startswith("weighted avg"):
            parts = line.split()
            if len(parts) >= 6:
                summary_cards.append(
                    {
                        "label": "Weighted F1",
                        "value": format_metric(parts[-2]),
                        "detail": f"{format_support(parts[-1])} test images",
                    }
                )

    return summary_cards


def evaluation_artifacts() -> dict[str, Any]:
    report_text = ""
    if EVALUATION_REPORT_PATH.exists():
        report_text = EVALUATION_REPORT_PATH.read_text(encoding="utf-8", errors="replace")

    summary_cards = evaluation_summary_from_report(report_text)

    confusion_matrix_url = ""
    if CONFUSION_MATRIX_IMAGE_PATH.exists():
        confusion_matrix_url = f"/artifacts/confusion-matrix?v={int(CONFUSION_MATRIX_IMAGE_PATH.stat().st_mtime)}"

    return {
        "available": bool(summary_cards or confusion_matrix_url or report_text),
        "summary": summary_cards,
        "confusion_matrix_url": confusion_matrix_url,
        "report_text": report_text,
    }


def start_managed_run(run: ManagedRun, command: list[str], log_path: Path) -> None:
    if run.process is not None and run.process.poll() is None:
        raise RuntimeError("A run is already active.")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"$ {' '.join(command)}\n\n", encoding="utf-8")
    log_file = log_path.open("a", encoding="utf-8")

    run.process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    run.command = command
    run.log_path = log_path

    def close_when_done() -> None:
        if run.process is not None:
            run.process.wait()
        log_file.close()

    threading.Thread(target=close_when_done, daemon=True).start()


def run_status(run: ManagedRun) -> dict[str, Any]:
    process = run.process
    return_code = None if process is None else process.poll()
    running = process is not None and return_code is None
    log_text = ""
    if run.log_path is not None and run.log_path.exists():
        log_text = run.log_path.read_text(encoding="utf-8", errors="replace")[-12000:]

    return {
        "running": running,
        "return_code": return_code,
        "command": run.command or [],
        "log": log_text,
    }


@app.get("/")
def index():
    return render_template("index.html", defaults=default_settings())


@app.post("/predict")
def predict():
    defaults = default_settings()
    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return render_template(
            "index.html",
            defaults=defaults,
            error="Choose an image first.",
            active_panel="predict",
        )

    filename = secure_filename(uploaded_file.filename)
    if not image_file_is_allowed(filename):
        return render_template(
            "index.html",
            defaults=defaults,
            error="Unsupported image type. Use jpg, jpeg, png, bmp, gif, or webp.",
            active_panel="predict",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    image_path = UPLOAD_DIR / filename
    uploaded_file.save(image_path)

    model_path = read_form_value("predict_model_path", defaults["model_path"])
    image_size = read_form_value("predict_image_size", defaults["image_size"], int)

    try:
        from src.predict import predict_single_image

        predicted_class, confidence = predict_single_image(
            image_path=image_path,
            model_path=model_path,
            image_size=(image_size, image_size),
        )
    except Exception as exc:  # noqa: BLE001 - show model/runtime errors in the UI.
        return render_template(
            "index.html",
            defaults=defaults,
            error=str(exc),
            active_panel="predict",
        )

    return render_template(
        "index.html",
        defaults=defaults,
        active_panel="predict",
        prediction={
            "class_name": predicted_class,
            "confidence": f"{confidence:.2%}",
            "image_name": filename,
        },
    )


@app.post("/train")
def train():
    defaults = default_settings()
    try:
        dataset_dir = read_form_value("dataset_dir", defaults["dataset_dir"])
        epochs = read_form_value("epochs", defaults["epochs"], int)
        batch_size = read_form_value("batch_size", defaults["batch_size"], int)
        learning_rate = read_form_value("learning_rate", defaults["learning_rate"], float)
        image_size = read_form_value("image_size", defaults["image_size"], int)
        resume_from = read_form_value("resume_from", defaults["resume_from"])
        workers = read_form_value("workers", defaults["workers"], int)
        max_queue_size = read_form_value("max_queue_size", defaults["max_queue_size"], int)
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--dataset-dir",
            dataset_dir,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--image-size",
            str(image_size),
            "--resume-from",
            resume_from,
            "--workers",
            str(workers),
            "--max-queue-size",
            str(max_queue_size),
        ]
        if request.form.get("use_multiprocessing") == "on":
            command.append("--use-multiprocessing")
        start_managed_run(training_run, command, TRAIN_LOG_PATH)
    except Exception as exc:  # noqa: BLE001 - user-facing local tool.
        return render_template(
            "index.html",
            defaults=defaults,
            error=str(exc),
            active_panel="train",
            training=run_status(training_run),
        )

    return render_template(
        "index.html",
        defaults=defaults,
        active_panel="train",
        message="Training started. Logs will update below.",
        training=run_status(training_run),
    )


@app.post("/evaluate")
def evaluate():
    defaults = default_settings()
    try:
        model_path = read_form_value("evaluate_model_path", defaults["model_path"])
        batch_size = read_form_value("evaluate_batch_size", defaults["batch_size"], int)
        image_size = read_form_value("evaluate_image_size", defaults["image_size"], int)
        command = [
            sys.executable,
            "-m",
            "src.evaluate",
            "--model-path",
            model_path,
            "--batch-size",
            str(batch_size),
            "--image-size",
            str(image_size),
        ]
        start_managed_run(evaluation_run, command, EVALUATE_LOG_PATH)
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "index.html",
            defaults=defaults,
            error=str(exc),
            active_panel="evaluate",
            evaluation=run_status(evaluation_run),
        )

    return render_template(
        "index.html",
        defaults=defaults,
        active_panel="evaluate",
        message="Evaluation started. Logs will update below.",
        evaluation=run_status(evaluation_run),
    )


@app.get("/status/<run_name>")
def status(run_name: str):
    if run_name == "train":
        return jsonify(run_status(training_run))
    if run_name == "evaluate":
        status_payload = run_status(evaluation_run)
        status_payload["artifacts"] = evaluation_artifacts()
        return jsonify(status_payload)
    return jsonify({"error": "Unknown run."}), 404


@app.get("/artifacts/confusion-matrix")
def confusion_matrix_image():
    if not CONFUSION_MATRIX_IMAGE_PATH.exists():
        abort(404)
    return send_file(CONFUSION_MATRIX_IMAGE_PATH, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
