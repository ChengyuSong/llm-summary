"""Cheap, post-processing struggle signals for code-contract summaries.

Three signals run on the model's `analysis` text and the produced
contract:

  - **hedge_density**: fraction of tokens that are uncertainty markers
    ("may", "might", "could", ...). Captures verbal hedging even when
    the model self-rates `confidence: high`.
  - **ub_mismatch**: the analysis names a UB class for the property
    (e.g. "null deref" for memsafe) but `requires[prop]` is empty.
    Catches "model walked through a UB and published no constraint".
  - **length_anomaly**: `|log10(len(analysis) / len(prompt_context))|`.
    UNDER (analysis ≪ context) is the orchestrator-skimmed-callees
    failure mode; OVER (analysis ≫ context) is the rambling-on-leaf
    failure mode. UNDER is weighted heavier in the composite score.

All three combine into a composite `score` (higher = more struggle).
The threshold is intentionally NOT baked in — callers (refinement
gate) pick one. Whole module runs in microseconds; no LLM calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# ── Hedge words ────────────────────────────────────────────────────────

# Single-word hedges. Multi-word phrases ("I think", "I believe", "not
# sure") are checked separately because the tokenizer is a word-level
# regex and would split them.
_HEDGE_WORDS: frozenset[str] = frozenset({
    "may", "might", "maybe", "possibly", "perhaps", "could",
    "probably", "presumably", "seems", "appears", "apparently",
    "unclear", "uncertain", "unsure", "ambiguous",
    "likely", "unlikely",
    # Approximation markers — model is widening because it can't pin a bound.
    "approximately", "roughly", "around", "somewhat",
})

_HEDGE_PHRASES: tuple[str, ...] = (
    "i think", "i believe", "not sure", "hard to say", "cannot tell",
    "can't tell", "difficult to determine",
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def hedge_density(text: str) -> float:
    """Return fraction of tokens in `text` that are hedge words/phrases.

    0.0 for empty/no-hedge text. Typical confident analysis: <0.01.
    Hedgy analysis: >0.03.
    """
    if not text:
        return 0.0
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    word_hits = sum(1 for t in tokens if t in _HEDGE_WORDS)
    # Each phrase match counts as len(phrase.split()) hits so densities
    # are comparable to the single-word ones.
    lower = text.lower()
    phrase_hits = 0
    for phrase in _HEDGE_PHRASES:
        if phrase in lower:
            phrase_hits += lower.count(phrase) * len(phrase.split())
    return (word_hits + phrase_hits) / len(tokens)


# ── UB-class keywords (per property) ──────────────────────────────────

# Lowercased substrings. We use word-boundary matching so e.g. "ovf"
# isn't a hit and "buffer" doesn't trigger "buf".
UB_KEYWORDS_BY_PROP: dict[str, frozenset[str]] = {
    "memsafe": frozenset({
        "null", "nullptr",
        "deref", "dereference", "dereferences", "dereferenced",
        "out-of-bounds", "out of bounds", "oob", "bounds",
        "uninitialized", "uninit", "use-before-init", "ubi",
        "use-after-free", "uaf", "double-free", "freed",
        "dangling",
    }),
    "memleak": frozenset({
        "leak", "leaks", "leaked", "leaking",
        "alloc", "allocate", "allocated", "allocation",
        "release", "released", "free", "freed",
        "ownership", "acquired", "acquire",
    }),
    "overflow": frozenset({
        "overflow", "overflows", "overflowed", "overflowing",
        "underflow", "wrap", "wraps", "wrapping",
        "shift", "shifts", "shifting",
        "divide", "division", "divided", "divisor",
        "signed",  # mentions of signed arithmetic suggest UB risk
        "undefined", "ub",
    }),
}

_KEYWORD_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _ub_pattern(prop: str) -> re.Pattern[str]:
    pat = _KEYWORD_RE_CACHE.get(prop)
    if pat is not None:
        return pat
    keywords = UB_KEYWORDS_BY_PROP.get(prop, frozenset())
    if not keywords:
        # Match nothing.
        compiled = re.compile(r"(?!.*)")
    else:
        # Sort longest first so multi-word keywords match before their
        # substrings ("out-of-bounds" before "bounds"). Use lookarounds
        # for word boundaries to handle hyphens and adjacent punctuation.
        parts = sorted((re.escape(k) for k in keywords), key=len, reverse=True)
        compiled = re.compile(
            r"(?<![A-Za-z])(?:" + "|".join(parts) + r")(?![A-Za-z])",
            re.IGNORECASE,
        )
    _KEYWORD_RE_CACHE[prop] = compiled
    return compiled


def count_ub_mentions(analysis: str, prop: str) -> int:
    """Count UB-class keyword hits in `analysis` for `prop`."""
    if not analysis:
        return 0
    return len(_ub_pattern(prop).findall(analysis))


def ub_mismatch(analysis: str, requires: list[str], prop: str) -> bool:
    """True iff `analysis` mentions a UB class for `prop` but `requires`
    is empty (no non-trivial entries).

    The intent: when the model verbalizes a UB risk, it should either
    rule it out explicitly (we can't reliably detect that cheaply) or
    publish a precondition that excludes it. Empty requires + UB
    mention is the cheapest "model raised a concern and dropped it" tell.
    """
    if count_ub_mentions(analysis, prop) == 0:
        return False
    # Treat trivial entries ("true", "") as empty.
    from .models import is_nontrivial
    return not any(is_nontrivial(r) for r in requires)


# ── Length anomaly ────────────────────────────────────────────────────

# Cap on the log-ratio magnitude. ratio < 1e-_LR_CAP or > 10^_LR_CAP
# clamps to this. Prevents empty-analysis or empty-context from
# dominating the composite score.
_LR_CAP: float = 2.0


def _length_log_ratio(analysis_chars: int, context_chars: int) -> float:
    """Return signed log10(analysis_chars / context_chars), clamped.

    Negative = analysis shorter than context (under-engagement);
    positive = analysis longer than context (over-engagement);
    zero = balanced. Empty inputs map to ±_LR_CAP rather than ±inf.
    """
    if analysis_chars <= 0:
        return -_LR_CAP
    if context_chars <= 0:
        return _LR_CAP
    log_r = math.log10(analysis_chars / context_chars)
    return max(-_LR_CAP, min(_LR_CAP, log_r))


def length_anomaly(analysis_chars: int, context_chars: int) -> float:
    """Return `|log10(analysis_chars / context_chars)|`, clamped.

    Direction-agnostic magnitude. Use the score formula in `compute()`
    if you care about UNDER vs OVER weighting.
    """
    return abs(_length_log_ratio(analysis_chars, context_chars))


# ── Composite score ───────────────────────────────────────────────────


@dataclass
class StruggleScore:
    """Per-property struggle signals plus a combined composite.

    Components are kept separate so callers can debug WHY a summary was
    flagged. `score` is a weighted sum; calibrate the gate threshold
    against your data — the formula is intentionally simple, not magic.

    `ub_mismatch` is recorded for inspection but does NOT contribute to
    `score`: empirically (zlib, ~250 pairs), it fires almost entirely on
    correct contracts where the analysis explains why no UB exists
    ("operands are unsigned, well-defined") and `requires` is rightly
    empty. Keep it as a diagnostic flag in case a future model-prompt
    combination produces real "raised concern then dropped it" cases.
    """

    hedge_density: float
    ub_mismatch: bool
    length_anomaly: float  # |log10(analysis_chars / context_chars)|, clamped
    score: float


# Asymmetric weighting: UNDER (analysis ≪ context) is the more dangerous
# failure mode (orchestrator skimmed callees → incomplete contract), so
# weight it 4×. OVER (analysis ≫ context) is the rambling-on-leaf mode,
# weighted 1× — usually only minor inaccuracies.
_W_UNDER: float = 4.0
_W_OVER: float = 1.0


def compute(
    analysis: str,
    requires: list[str],
    prop: str,
    context_chars: int,
) -> StruggleScore:
    """Compute the composite struggle score for one property's response.

    `analysis` and `requires` come straight off `CodeContractSummary`.
    `context_chars` is the size of the prompt context the model saw
    when producing this contract (function body + rendered callee
    contracts). The pass already builds this string; pass `len(prompt)`.
    """
    hd = hedge_density(analysis)
    mm = ub_mismatch(analysis, requires, prop)
    log_r = _length_log_ratio(len(analysis), context_chars)
    under = max(0.0, -log_r)
    over = max(0.0, log_r)
    score = hd * 20.0 + under * _W_UNDER + over * _W_OVER
    return StruggleScore(
        hedge_density=hd,
        ub_mismatch=mm,
        length_anomaly=abs(log_r),
        score=score,
    )
