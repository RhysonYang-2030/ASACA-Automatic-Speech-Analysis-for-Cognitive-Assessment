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

## Documentation
Full API reference and user guide live in the [`docs/`](docs/) directory and on [Read the Docs](https://example.com/).

## License
Released under the Apache-2.0 license.
