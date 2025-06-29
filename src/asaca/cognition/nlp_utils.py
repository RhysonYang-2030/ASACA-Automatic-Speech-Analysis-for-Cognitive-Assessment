"""
Light-weight NLP helpers shared by all cognition-modules.
"""
from __future__ import annotations
import re, string, pkg_resources
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords                # noqa: E402
from nltk.stem.snowball import SnowballStemmer   # noqa: E402
# ----------------------------------------------------------------------
# ensure NLTK corpora are available on first run  ----------------------
import nltk
for pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
_FILLERS = {
    "um","o","erm","uh","er","emm","em","umm","hmm","eh","ed","s","mm",
    "oh","ah","hm","e","amm","ur","m","si","om","oi","t","ar","th","emmm"
}

_STEMMER = SnowballStemmer("english")
from nltk.corpus import stopwords
_STOPWORDS = set(stopwords.words("english")) | set(string.punctuation)

# ----------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z']+")

@lru_cache(maxsize=16)
def preprocess(text: str) -> List[str]:
    """
    Lower-case, tokenize, remove stop-words, and stem (Snowball) **except** for
    filler words, which are kept verbatim.

    Returns the cleaned token list.
    """
    if not text:
        return []

    tokens: Iterable[str] = TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for tok in tokens:
        if tok in _FILLERS:          # keep untouched
            out.append(tok)
        elif tok not in _STOPWORDS:
            out.append(_STEMMER.stem(tok))
    return out
