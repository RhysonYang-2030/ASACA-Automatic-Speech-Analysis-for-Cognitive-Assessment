# ASACA – Automatic Speech Analysis for Cognitive Assessments
[![CI](https://github.com/RhysonYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment/actions/workflows/ci.yml/badge.svg)](../../actions) 
[![PyPI](https://img.shields.io/pypi/v/asaca?logo=pypi)](https://pypi.org/project/asaca/) 
[![License](https://img.shields.io/github/license/RhysonYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment)](LICENSE) 
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
![GUI](docs/img/asaca_gui.gif)

ASACA is an **end-to-end toolkit** that transforms raw speech into
multimodal biomarkers — lexical, prosodic and pause-based — and returns
an interpretable prediction ( *HC / MCI / AD* ).

---

## ✨ Key Features
| Capability | Detail |
|------------|--------|
| **Single-command inference** | `asaca run audio.wav` outputs JSON + PDF report |
| **Fine-tuned wav2vec 2.0 ASR** | < 2 % WER on in-domain test set |
| **Explainability** | SHAP plots per prediction |
| **Rich feature set** | word-error rate, syllable rate, pause stats, spectral cues |
| **Offline-ready** | Model weights stored under `Models/` via Git LFS |
| **PEP 517/621 packaging** | `pip install asaca` or editable mode |
| **Future-proof docs** | MkDocs with Material theme |

---

## 🚀 Quick start
```bash
# 1 – clone
git clone https://github.com/ProfYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment.git
cd ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment

# 2 – install (Python ≥ 3.11)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 3 – run demo
asaca run samples/demo.wav
Expected console output:

text
Copy
Edit
Transcription WER   : 5.1 %
Syllable rate       : 4.0 Hz
Mean pause duration : 0.57 s
Prediction          : MCI  (p = 0.71)

---

## Documentation
Full API reference and user guide live in the [`docs/`](docs/) directory and on [Read the Docs](https://example.com/).

## License
Released under the Apache-2.0 license.
