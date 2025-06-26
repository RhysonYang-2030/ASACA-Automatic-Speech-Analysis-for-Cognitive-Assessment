# ASACA — Automatic Speech and Cognition Analysis  
[![CI](https://github.com/your-org/asaca/actions/workflows/ci.yml/badge.svg)](../../actions)
[![PyPI](https://img.shields.io/pypi/v/asaca?logo=pypi)](https://pypi.org/project/asaca/)
[![License](https://img.shields.io/github/license/your-org/asaca)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](#)

ASACA is an **end-to-end pipeline** that converts raw speech into
*clinical-grade* neuro-cognitive biomarkers and interpretable predictions
(`HC` vs `MCI` vs `AD`).  
It couples a fine-tuned *wav2vec 2.0* ASR backbone with linguistics,
acoustic prosody, pause/syllable analysis and a SHAP-explainable
logistic-regression head.

<div align="center">
  <img alt="ASACA overview" src="docs/img/architecture.svg" width="720">
</div>

---

## ✨ Key Features
* **Single command analysis** – `asaca run audio.wav`
* **Multimodal features** – lexical, temporal, prosodic & spectral
* **Explainability out-of-the-box** – SHAP plots per prediction
* **Real-time GUI** – PyQt5 dashboard with waveform & segmentation panes
* **Docker / VS Code dev-container** for bullet-proof reproducibility
* **Extensible YAML config** – tune VAD gaps, pause thresholds, LM text
* **MIT-licensed model weights via 🤗 Hub** (downloaded on first run)

---

## 🚀 Quick start

```bash
# 1. clone and install
git clone https://github.com/your-org/asaca.git
cd asaca
pip install -e ".[full]"        # full = runtime + [gui,dev]

# 2. authenticate Hugging Face once (stores token in ~/.cache/huggingface)
huggingface-cli login

# 3. run inference
asaca run samples/patient07.wav         # CLI
python -m speech_tools.app              # GUI

# Expected output
# ──────────────────────────────
# Transcription WER   : 5.2 %
# Syllable rate       : 4.1 Hz
# Mean pause duration : 0.58 s
# Predicted label     : MCI   (p = 0.71) ✓
# Full report written : reports/patient07.pdf
