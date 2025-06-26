# ASACA — Automatic Speech & Cognition Assessment  
[![CI](https://github.com/ProfYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment/actions/workflows/ci.yml/badge.svg)](../../actions)
//[![PyPI](https://img.shields.io/pypi/v/asaca?logo=pypi)](https://pypi.org/project/asaca/)
[![License](https://img.shields.io/github/license/ProfYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)

> **Tagline:** *An end-to-end toolkit that converts raw speech into multimodal neuro-cognitive biomarkers & transparent predictions.*

---

## ✨ Features
* **One-command inference:** `asaca run input.wav`
* Fine-tuned **wav2vec 2.0** ASR backbone + linguistic, prosodic & pause features
* **Explainability built-in** via SHAP
* **Docker & VS Code Dev-Container** for reproducible research
* **Config-driven** (YAML) thresholds and model paths
* **Lightweight demo** – works offline with the single `demo.wav` shipped in `samples/`

---

## 🚀 Quick start

```bash
# clone & install (editable)
git clone https://github.com/ProfYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment.git
cd ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment
pip install -e ".[full]"          # full = runtime + dev + gui extras

# run demo inference
asaca run samples/demo.wav
