# Contributing to ASACA

First, thank you ❤️ for taking the time to contribute!  
ASACA (Automatic Speech & Cognition Assessment) relies on the open-source community to stay reliable, transparent, and cutting-edge.

---

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Style Guide](#style-guide)
5. [Running the Test Suite](#running-the-test-suite)
6. [Documentation](#documentation)
7. [Pull-Request Checklist](#pull-request-checklist)
8. [Releasing](#releasing)
9. [Contact](#contact)

---

## Code of Conduct
Participating in this project means you agree to abide by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

---

## Getting Started

```bash
git clone https://github.com/ProfYang-2030/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment.git
cd ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment
python -m venv .venv           # or conda create -n asaca python=3.11
source .venv/bin/activate
pip install -e ".[dev]"        # runtime + linters + tests
pre-commit install             # auto-format on commit
Development Workflow
Step	Command / Action	Purpose
1	git checkout -b feat/awesome-feature	Create a topic branch
2	Code + ruff check . + pytest -q	Keep it clean & tested
3	git add -p && git commit -m "feat: add awesome feature"	Conventional commits
4	git push --set-upstream origin feat/awesome-feature	Push branch
5	Open Pull Request on GitHub	PR triggers full CI

Style Guide
PEP 8 enforced by ruff (no unused imports, 120-char lines).

Black auto-formats code (installed in pre-commit).

Docstrings follow NumPy style.

Type hints are mandatory for new public functions & methods.

Keep functions pure when feasible; heavy I/O belongs in CLI wrappers.

Running the Test Suite
bash
Copy
Edit
pytest -q          # run all tests
pytest tests/xyz.py::TestClass::test_one   # single test
pytest --cov=asaca_cognition --cov-report=term-missing   # coverage
Add tests for every new public API or bugfix.

Documentation
User-facing changes require updates to README.md and any relevant
tutorial notebooks under notebooks/.
Large additions should get a new page in docs/ (MkDocs).

Pull-Request Checklist
 I ran ruff check . and fixed all issues.

 pytest -q passes locally.

 I added/updated unit tests.

 I updated docs / README examples.

 I described why the change is needed in the PR body.

 I checked that CI passes after pushing.

Releasing
Project maintainers only:

bash
Copy
Edit
git switch main && git pull
bumpver update --patch           # or --minor / --major
git push && git push --tags      # tag triggers PyPI & Docker publish
Contact
Questions? Ping the issue tracker or e-mail xyang2@tcd.ie.

pgsql
Copy
Edit
