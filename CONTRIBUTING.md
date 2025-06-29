# Contributing to ASACA
*A toolkit for Automatic Speech & Cognition Assessment*  

Thank you for considering a contribution!  
ASACA welcomes pull-requests for bug fixes, new features, documentation, and tests.

---

## Table of Contents
1. [Ground Rules](#ground-rules)  
2. [Project Setup](#project-setup)  
3. [Development Workflow](#development-workflow)  
4. [Code Style & Linting](#code-style--linting)  
5. [Running Tests](#running-tests)  
6. [Documentation](#documentation)  
7. [Pull-Request Checklist](#pull-request-checklist)  
8. [Release Process (maintainers)](#release-process-maintainers)  
9. [Contact](#contact)  

---

## Ground Rules
* **Be kind & inclusive** – we follow the [Code of Conduct](CODE_OF_CONDUCT.md).  
* Keep the *main* branch **green** (all tests passing).  
* One logical change per PR.  
* Large changes? Open an issue first so we can discuss direction.

---

## Project Setup

```bash
# Clone your fork
git clone https://github.com/<your-user>/ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment.git
cd ASACA-Automatic-Speech-Analysis-for-Cognitive-Assessment

# Create virtual-env
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dev dependencies
pip install -e ".[dev]"          # runtime + linters + tests
pre-commit install               # auto-format staged files
```

---

## Development Workflow

| Step | Command / Action | Notes |
|------|------------------|-------|
| **1** | `git checkout -b feat/<short-description>` | Use `fix/…` for bugs. |
| **2** | Code your change | ⌨ |
| **3** | `ruff check .`   | Lints & auto-fixable issues. |
| **4** | `pytest -q`      | All tests green. |
| **5** | `git add -p` & `git commit -m "feat: …"` | Conventional commit style. |
| **6** | `git push --set-upstream origin feat/<…>` | Push branch. |
| **7** | Open Pull Request | PR triggers full CI. |

> **Tip:** commit messages should start with  
> `feat:`, `fix:`, `docs:`, `refactor:`, `test:` or `ci:`.

---

## Code Style & Linting
* **Black** – auto-format (`pre-commit` runs it).  
* **Ruff** – static analysis; 0 errors required.  
* **Type hints** – mandatory for new public functions/classes.  
* Public APIs need **NumPy-style docstrings** with examples.

---

## Running Tests

```bash
pytest -q                      # all tests
pytest tests/test_audio.py -k pause   # subset
pytest --cov=asaca_cognition --cov-report=term-missing
```

Add/extend tests for every new feature or bug fix.

---

## Documentation
* Update **README.md** if behaviour changes or new CLI flags are added.  
* Large additions → add a Markdown page under `docs/` and link from README.  
* Example notebooks live in `notebooks/`; keep them lightweight (< 5 MB).

---

## Pull-Request Checklist
- [ ] Code compiles and runs locally.  
- [ ] `ruff check .` passes (no warnings).  
- [ ] `pytest -q` passes (existing + new tests).  
- [ ] Docs / examples updated.  
- [ ] Linked related issue(s) in PR description (`Fixes #42`).  
- [ ] PR follows [Ground Rules](#ground-rules).

---

## Release Process (maintainers)

```bash
git checkout main
git pull
bumpver update --patch           # or --minor / --major
git push && git push --tags      # tag triggers PyPI + Docker publish
```

* `CI` must be green before cutting a release.  
* Draft release notes from merged PR titles (`Generate release notes` button).

---

## Contact
Open an issue for questions & proposals.  
Security concerns? See [SECURITY.md](SECURITY.md) or e-mail **xyang2@tcd.ie**.
