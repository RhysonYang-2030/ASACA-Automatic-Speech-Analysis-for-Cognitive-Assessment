# ASACA – Automatic Speech Analysis for Cognitive Assessments
[![CI](https://github.com/RhysonYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment/actions/workflows/ci.yml/badge.svg)](../../actions) 
[![PyPI](https://img.shields.io/pypi/v/asaca?logo=pypi)](https://pypi.org/project/asaca/) 
[![License](https://img.shields.io/github/license/RhysonYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment)](LICENSE) 
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
![GUI](docs/img/asaca_gui.gif)

ASACA is a toolkit that extracts linguistic and acoustic biomarkers from raw speech and provides an optional graphical interface for quick analysis.

## Why ASACA
* **Turn–key** inference from a single command.
* **Explainable** cognitive scoring with SHAP.
* **Modular** components for diarisation, pause detection and feature extraction.

## Quick start
```bash
pip install asaca
asaca-cli infer samples/demo.wav -o out/
```

See the [examples](examples/) folder for a notebook demo.

## Documentation
Full API reference and user guide live in the [`docs/`](docs/) directory and on [Read the Docs](https://example.com/).

## License
Released under the Apache-2.0 license.
