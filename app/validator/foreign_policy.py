"""Foreign-plate rejection policy (Phase 10 - Step 2).

This is the **policy layer** that sits on top of the Phase 9 validator. Phase 9
*classifies* a text string (``VALID_JORDANIAN`` / ``VALID_GOVERNMENT`` /
``FOREIGN_LIKELY`` / ``INVALID_FORMAT`` / ``UNKNOWN``); this module turns that
classification into an actionable **decision**. There is no ML, no image input,
no confidence score and no third-party dependency.

Owner-approved policy (``logs/phase10_foreign/DESIGN.md`` section C + the Phase
10 Step 2 decisions):

    VALID_JORDANIAN   -> ACCEPT_JORDANIAN
    VALID_GOVERNMENT  -> ACCEPT_JORDANIAN
    FOREIGN_LIKELY    -> REJECT_FOREIGN          (all rules: KSA + IRQ/UAE)
    INVALID_FORMAT    -> FLAG_REVIEW             (may be OCR-mangled)
    UNKNOWN           -> FLAG_REVIEW             (cannot decide)
    anything else     -> FLAG_REVIEW

Speculative-rule transparency: whenever a decision rests on a rule flagged
``speculative`` in :data:`app.validator.rules.RULES`, the ``speculative`` output
field is ``True`` and the ``reason`` preserves the exact ``[SPECULATIVE ...]``
substring from Phase 9 (for rejections) or the short
``[SPECULATIVE - owner insight]`` note (for speculative acceptances). A
non-speculative rule always yields ``speculative=False`` with no such note.

This module is **stdlib only** and does not import the validator at import time
except through the tiny convenience wrapper :func:`decide_text`.
"""

from __future__ import annotations

try:  # package import (``from app.validator.foreign_policy import decide``)
    from .rules import RULES
except ImportError:  # direct execution: ``python app/validator/foreign_policy.py``
    from rules import RULES  # type: ignore

# Exact Phase 9 speculative note (em dash), preserved verbatim in reject reasons.
SPECULATIVE_NOTE = "[SPECULATIVE \u2014 owner insight; confirm before ground truth]"
# Short note appended to speculative *acceptances* (owner wording).
SPECULATIVE_NOTE_SHORT = "[SPECULATIVE \u2014 owner insight]"

# Rule-ID -> speculative flag (source of truth: Phase 9 rules.py, read-only).
_RULE_SPECULATIVE: dict[str, bool] = {
    rule["id"]: bool(rule.get("speculative")) for rule in RULES
}

# The three possible decisions.
ACCEPT_JORDANIAN = "ACCEPT_JORDANIAN"
REJECT_FOREIGN = "REJECT_FOREIGN"
FLAG_REVIEW = "FLAG_REVIEW"
DECISIONS = (ACCEPT_JORDANIAN, REJECT_FOREIGN, FLAG_REVIEW)


def _is_speculative(matched_rule: str | None) -> bool:
    return bool(matched_rule) and _RULE_SPECULATIVE.get(matched_rule, False)


def decide(classification: str, matched_rule: str | None,
           normalized: str) -> dict:
    """Map a Phase 9 classification to a rejection decision.

    Returns exactly four keys:

    ``decision``
        ``ACCEPT_JORDANIAN`` | ``REJECT_FOREIGN`` | ``FLAG_REVIEW``.
    ``reason``
        short human-readable explanation. Speculative rules preserve the exact
        ``[SPECULATIVE ...]`` substring.
    ``matched_rule``
        the Phase 9 rule ID echoed back, or ``None``.
    ``speculative``
        ``True`` iff the decision rests on a rule flagged speculative in Phase 9.

    The policy never rewrites ``normalized`` and never invents a confidence.
    """
    speculative = _is_speculative(matched_rule)
    rule_id = matched_rule or "unknown"

    if classification == "VALID_JORDANIAN":
        reason = f"matched Jordanian private rule {rule_id}"
        if speculative:
            reason = f"{reason} {SPECULATIVE_NOTE_SHORT}"
        return _out(ACCEPT_JORDANIAN, reason, matched_rule, speculative)

    if classification == "VALID_GOVERNMENT":
        reason = f"matched Jordanian government rule {rule_id}"
        if speculative:
            reason = f"{reason} {SPECULATIVE_NOTE_SHORT}"
        return _out(ACCEPT_JORDANIAN, reason, matched_rule, speculative)

    if classification == "FOREIGN_LIKELY":
        reason = f"matched foreign rule {rule_id}; rejected"
        if speculative:
            # Preserve the exact Phase 9 [SPECULATIVE ...] substring.
            reason = f"{reason} {SPECULATIVE_NOTE}"
        return _out(REJECT_FOREIGN, reason, matched_rule, speculative)

    if classification == "INVALID_FORMAT":
        return _out(FLAG_REVIEW, "no known pattern; needs human review",
                    matched_rule, False)

    if classification == "UNKNOWN":
        return _out(FLAG_REVIEW, "insufficient input; needs human review",
                    matched_rule, False)

    # Defensive: any future/unknown classification value.
    return _out(FLAG_REVIEW, "unexpected classification; needs review",
                matched_rule, False)


def _out(decision: str, reason: str, matched_rule: str | None,
         speculative: bool) -> dict:
    return {
        "decision": decision,
        "reason": reason,
        "matched_rule": matched_rule,
        "speculative": speculative,
    }


def decide_text(text: str) -> dict:
    """Convenience: run Phase 9 ``validate()`` then :func:`decide`.

    Adds the Phase 9 ``classification`` and ``normalized`` fields to the
    decision dict so a caller sees the full chain in one object.
    """
    from .jordanian_validator import validate

    v = validate(text)
    d = decide(v["classification"], v["matched_rule"], v["normalized"])
    d["classification"] = v["classification"]
    d["normalized"] = v["normalized"]
    return d


def _main() -> int:
    import json
    import sys

    text = sys.stdin.read().rstrip("\r\n")
    print(json.dumps(decide_text(text), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # tiny self-check
    assert set(decide("VALID_JORDANIAN", "JOR_PRIVATE_7", "1234567").values())
    assert decide("FOREIGN_LIKELY", "KSA_LIKELY", "2111GUD")["decision"] == REJECT_FOREIGN
    assert decide("FOREIGN_LIKELY", "UAE_LIKELY_L5D", "V42779")["speculative"] is True
    assert decide("UNKNOWN", None, "")["decision"] == FLAG_REVIEW
    assert decide("INVALID_FORMAT", None, "ABC1234")["decision"] == FLAG_REVIEW
    print(f"foreign_policy.py OK: {len(RULES)} rules, "
          f"{len(_RULE_SPECULATIVE)} speculative flags indexed; "
          f"{len(DECISIONS)} decisions")
