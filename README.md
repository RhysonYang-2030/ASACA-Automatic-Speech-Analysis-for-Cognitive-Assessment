# ASACA — Automatic Speech & Cognition Assessment  
[![CI](https://github.com/ProfYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment/actions/workflows/ci.yml/badge.svg)](../../actions) 
[![PyPI](https://img.shields.io/pypi/v/asaca?logo=pypi)](https://pypi.org/project/asaca/) 
[![License](https://img.shields.io/github/license/ProfYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment)](LICENSE) 
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)

> **An end-to-end toolkit that converts raw speech into multimodal neuro-cognitive biomarkers, transcriptions and transparent predictions.**

---

## ✨ Features
| Category | Highlights |
|----------|------------|
| **Turn-key inference** | `asaca run input.wav` outputs transcript, feature sheet, SHAP explanations & PDF report |
| **Multimodal features** | Fine-tuned *wav2vec 2.0* ASR + lexical, prosodic, pause & temporal measures |
| **Explainability built-in** | Global & per-sample SHAP visualisations for model transparency |
| **Reproducible** | Dockerfile & VS Code *dev-container* ready; CI on GitHub Actions |
| **Config-driven** | All thresholds, paths & tunables in a single `config.yaml` |
| **Lightweight demo** | Works completely offline with `samples/demo.wav` |

---

## 🚀 Quick start

```bash
# 1 clone & editable install
git clone https://github.com/ProfYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment.git
cd ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment
pip install -e ".[full]"        # full = runtime + dev + gui extras

# 2 run demo inference
asaca run samples/demo.wav
```

<details>
<summary>Expected terminal output</summary>

```text
───────────────────────────────
Transcription WER   : 4.8 %
Syllable rate       : 4.2 Hz
Mean pause duration : 0.55 s
Prediction          : MCI   (p = 0.69)
Full report         : reports/demo.pdf
```
</details>

---

### Model weights

The fine-tuned *wav2vec 2.0* checkpoint (≈ 360 MB) is version-controlled via **Git LFS** in `checkpoints/`.  
The first clone will therefore download `checkpoints/asaca_asr.pt`.

Prefer hosting on Hugging Face instead?

1. `git lfs uninstall` – remove the pointer  
2. `huggingface-cli upload checkpoints/asaca_asr.pt xinboyang/asaca-asr`  
3. Edit `config.yaml` → `model_path: hf::xinboyang/asaca-asr`

---

## 🐳 Docker

```bash
docker build -t asaca .
docker run --rm -v $PWD/samples:/data asaca asaca run /data/demo.wav
```

*CUDA 12 build – see inline comments in the [`Dockerfile`](Dockerfile).*

---

## 🏗️ Directory layout


speech_tools/               # audio utils, diarisation, syllable counter, …
asaca_cognition/            # feature extractor, ML head, SHAP explainers
config.yaml                 # editable hyper-parameters
samples/                    # demo.wav (public example)
checkpoints/                # wav2vec2-ft + cognitive head (via Git LFS)
docs/                       # optional in-depth docs & figures


## 📖 Training workflow

1. **ASR fine-tune** – `asaca train-asr ...`
2. **Feature export** – `python -m asaca_cognition.feature_extractor ...`
3. **Classifier grid-search** – `python -m asaca_cognition.model_training_improved ...`

Full commands live in `docs/TRAINING.md`.


## 🛡️ License

This project is distributed under the **Apache License 2.0** – see [`LICENSE`](LICENSE).


## 🤝 Contributing

We welcome issues & pull requests!  
Please review [`CONTRIBUTING.md`](CONTRIBUTING.md) before submitting code.


## 📚 Citation

If you use **ASACA** in academic work, please cite the MSc thesis:

```bibtex
@mastersthesis{yang2025asaca,
  author  = {Xinbo Yang},
  title   = {ASACA: Automatic Speech and Cognition Assessment},
  school  = {Trinity College Dublin},
  year    = {2025},
  month   = {September},
  url     = {https://github.com/ProfYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment}
}

## 📬 Contact & support

* **Issue tracker** – preferred channel for bugs & feature requests.  
* **E-mail** – <xyang2@tcd.ie> for private queries or security disclosures.

