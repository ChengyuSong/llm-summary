"""Tests for `code_contract.struggle` — cheap struggle-signal score."""

from __future__ import annotations

from llm_summary.code_contract.struggle import (
    compute,
    count_ub_mentions,
    hedge_density,
    length_anomaly,
    ub_mismatch,
)


class TestHedgeDensity:
    def test_empty_text(self) -> None:
        assert hedge_density("") == 0.0

    def test_no_hedges(self) -> None:
        text = "The function dereferences buf at index 0 and returns the result."
        assert hedge_density(text) == 0.0

    def test_single_hedge(self) -> None:
        text = "The function may dereference buf at index zero."
        # 1 hedge ("may") in 8 word tokens.
        assert hedge_density(text) == 1 / 8

    def test_phrase_hedge_counts_words(self) -> None:
        # "I think" should count as 2 hits so phrases compete fairly with
        # single-word hedges.
        text = "I think buf is non-null on this path."
        # 8 tokens (I, think, buf, is, non, null, on, this, path) — wait let
        # me recount: tokenizer matches [A-Za-z][A-Za-z']*. "non-null" splits
        # into "non" and "null". So tokens: I think buf is non null on this
        # path = 9 tokens. Phrase "i think" = 2 hits.
        assert hedge_density(text) == 2 / 9

    def test_case_insensitive(self) -> None:
        # 4 tokens: MAYBE, buf, is, null. 1 hedge / 4.
        assert hedge_density("MAYBE buf is null.") == 1 / 4


class TestUbKeywords:
    def test_memsafe_hits(self) -> None:
        text = "The function dereferences buf and may cause null deref."
        # "dereferences", "null", "deref" — three matches.
        assert count_ub_mentions(text, "memsafe") == 3

    def test_overflow_hits(self) -> None:
        text = "Signed shift overflow could occur if n exceeds 31."
        # "signed", "shift", "overflow" — three matches.
        assert count_ub_mentions(text, "overflow") == 3

    def test_property_isolation(self) -> None:
        # "overflow" mention should NOT register under memsafe.
        text = "Integer overflow happens when product exceeds INT_MAX."
        assert count_ub_mentions(text, "memsafe") == 0
        assert count_ub_mentions(text, "overflow") >= 1

    def test_word_boundary(self) -> None:
        # "overload" should not match "overflow"; "buffered" should not
        # match "buf*" keywords. (None of our keywords are "buf", but
        # check the pattern handles word boundaries.)
        text = "The function uses overload resolution on buffered streams."
        assert count_ub_mentions(text, "overflow") == 0
        assert count_ub_mentions(text, "memsafe") == 0


class TestUbMismatch:
    def test_no_ub_mention_no_mismatch(self) -> None:
        # No UB keyword, no mismatch even if requires is empty.
        text = "The function returns the input unchanged."
        assert ub_mismatch(text, [], "memsafe") is False

    def test_ub_mention_empty_requires_is_mismatch(self) -> None:
        text = "The function may dereference a null pointer."
        assert ub_mismatch(text, [], "memsafe") is True

    def test_ub_mention_with_requires_no_mismatch(self) -> None:
        text = "The function dereferences buf; require buf != NULL."
        assert ub_mismatch(text, ["buf != NULL"], "memsafe") is False

    def test_trivial_requires_count_as_empty(self) -> None:
        # `is_nontrivial` filters "true" and similar — they shouldn't save
        # the model from a mismatch flag.
        text = "Null deref possible at line 5."
        assert ub_mismatch(text, ["true"], "memsafe") is True


class TestLengthAnomaly:
    def test_balanced_returns_zero(self) -> None:
        # ratio == 1 → log10(1) == 0
        assert length_anomaly(500, 500) == 0.0

    def test_under_engagement_positive(self) -> None:
        # 600 chars of analysis on 60K context → log10(0.01) → 2.0
        a = length_anomaly(600, 60_000)
        assert 1.9 < a <= 2.0

    def test_over_engagement_positive(self) -> None:
        # 1000 chars of analysis on 100 chars of context → log10(10) == 1.0
        assert abs(length_anomaly(1000, 100) - 1.0) < 1e-9

    def test_empty_analysis_caps(self) -> None:
        # No-analysis case shouldn't blow up; returns the cap.
        assert length_anomaly(0, 1000) == 2.0

    def test_empty_context_caps(self) -> None:
        # Defensive: zero context shouldn't divide-by-zero.
        assert length_anomaly(500, 0) == 2.0

    def test_clamped_at_cap(self) -> None:
        # Extreme over-engagement is clamped, not unbounded.
        assert length_anomaly(10**8, 1) == 2.0


class TestComposite:
    def test_clean_confident_analysis_low_score(self) -> None:
        # Real-world: adler32_z overflow analysis. Analysis length
        # roughly matches body+callee context for a leaf function.
        analysis = (
            "The function performs arithmetic on uLong (typedef'd to "
            "unsigned long). All additions, subtractions, shifts, and "
            "modulos operate on unsigned types, which wrap modulo 2^N "
            "in C and are well-defined."
        )
        s = compute(analysis, [], "overflow", context_chars=len(analysis))
        assert s.hedge_density == 0.0
        # Balanced length → no length penalty.
        assert s.length_anomaly < 0.1
        # ub_mismatch may still fire (mentions "shift"/"wrap"); score bounded.
        assert s.score < 4.0

    def test_under_engagement_high_score(self) -> None:
        # Real failure mode: orchestrator function gets a tiny analysis
        # for a huge prompt context (body + many callee contracts).
        # Hedgy + UNDER weighted 4× → big score (mm flagged but not scored).
        analysis = (
            "I think buf could be null on this path. "
            "Maybe the bounds check is unreachable, unclear."
        )
        s = compute(analysis, [], "memsafe", context_chars=50_000)
        assert s.hedge_density > 0.05
        assert s.ub_mismatch is True  # diagnostic, doesn't add to score
        assert s.length_anomaly > 1.5  # log10(100/50000) ≈ -2.7 → clamp to 2
        # under * 4 + hedge contribution clears the threshold.
        assert s.score > 8.0

    def test_over_engagement_lower_weight(self) -> None:
        # OVER (rambling on a leaf) should fire but be weighted less
        # than UNDER. Same magnitude of length anomaly, smaller score.
        long_analysis = "x" * 1000
        short_context = 100
        s = compute(long_analysis, ["true"], "overflow",
                    context_chars=short_context)
        # log10(1000/100) == 1.0 → length_anomaly == 1.0
        assert abs(s.length_anomaly - 1.0) < 1e-9
        # over * 1 = 1.0 contribution from length; no hedges, no UB.
        assert s.score < 2.0

    def test_components_are_separable(self) -> None:
        # Each component reflects the underlying signal regardless of
        # the others — useful for debugging why something was flagged.
        analysis = "buf may be null."
        s = compute(analysis, ["buf != NULL"], "memsafe",
                    context_chars=len(analysis))
        assert s.hedge_density > 0.0
        assert s.ub_mismatch is False  # requires is non-empty
