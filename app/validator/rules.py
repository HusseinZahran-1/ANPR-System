"""Jordanian plate validation ruleset (Phase 9 - Step 2).

Pure, stdlib-only data: a **top-down, first-match-wins** list of anchored
regular-expression rules over the character set ``[A-Z0-9]``. There is no ML,
no third-party dependency, and no image/confidence input at this layer.

Design + owner decisions: ``logs/phase9_validator/DESIGN.md`` (section C) and the
owner-approved Phase 9 Step 2 ruleset. Every speculative rule carries
``speculative=True``; the validator appends an explicit ``[SPECULATIVE ...]``
note to its ``reason`` for those rules.

Rule dict schema::

    {
        "id":             unique rule identifier,
        "pattern":        anchored regex (``^...$``) over ``[A-Z0-9]``,
        "classification": VALID_JORDANIAN | VALID_GOVERNMENT |
                          FOREIGN_LIKELY | INVALID_FORMAT | UNKNOWN,
        "confidence":     "HIGH" | "MEDIUM" | "SPECULATIVE",
        "evidence":       why the rule exists,
        "speculative":    bool (True -> reason gets the [SPECULATIVE ...] note),
        "notes":          free-form caveat,
    }

Ordering matters: the observed-only Jordanian shapes (``JOR_OBS_T`` /
``JOR_OBS_X``) must be tested **before** the broad ``UAE_LIKELY_L5D`` rule,
because e.g. ``X78572`` matches both and is intended to classify as an observed
Jordanian government shape.
"""

from __future__ import annotations

# The 8 authority codes confirmed by the owner (spec section 12, decision I.3).
# Kept as data for reference; the patterns below inline the alternation.
GOV_CODES = ("GV", "PM", "SN", "PR", "JC", "AQ", "CD", "TM")

RULES: list[dict] = [
    # ----------------------------------------------------------------------
    # Jordanian government / authority
    # ----------------------------------------------------------------------
    {
        "id": "JOR_GOV_SPEC_5D",
        "pattern": r"^(?:GV|PM|SN|PR|JC|AQ|CD|TM)\d{5}$",
        "classification": "VALID_GOVERNMENT",
        "confidence": "HIGH",
        "evidence": (
            "Spec section 12 authority code list (GV/PM/SN/PR/JC/AQ/CD/TM) + "
            "5-digit suffix (owner decision I.4: 5 digits primary)."
        ),
        "speculative": False,
        "notes": (
            "8 codes are owner-confirmed; the 5-digit suffix matches the "
            "observed T+5 / X+5 shape but is not independently verified."
        ),
    },
    {
        "id": "JOR_GOV_SPEC_ANY",
        "pattern": r"^(?:GV|PM|SN|PR|JC|AQ|CD|TM)\d{1,6}$",
        "classification": "VALID_GOVERNMENT",
        "confidence": "SPECULATIVE",
        "evidence": (
            "Spec section 12 code list only; 1-6 digit suffix fallback "
            "(owner decision I.4: 1-6 fallback, speculative)."
        ),
        "speculative": True,
        "notes": (
            "Deliberately broad (accepts GV1 .. GV123456); a suffix-length "
            "refinement needs owner confirmation before ground-truth use."
        ),
    },
    {
        "id": "JOR_OBS_T",
        "pattern": r"^T\d{5}$",
        "classification": "VALID_GOVERNMENT",
        "confidence": "SPECULATIVE",
        "evidence": (
            "Phase 3 observed camera candidates T56454 / T80116; owner "
            "decision I.5: real (speculative)."
        ),
        "speculative": True,
        "notes": "Observed-only shape; disable if the owner rules it an artifact.",
    },
    {
        "id": "JOR_OBS_X",
        "pattern": r"^X\d{5}$",
        "classification": "VALID_GOVERNMENT",
        "confidence": "SPECULATIVE",
        "evidence": (
            "Phase 3 observed camera candidate X78572; owner decision I.5: "
            "real (speculative)."
        ),
        "speculative": True,
        "notes": (
            "Must be evaluated before UAE_LIKELY_L5D, which would also match "
            "X78572."
        ),
    },
    {
        "id": "JOR_OBS_IN",
        "pattern": r"^IN\d{2,5}$",
        "classification": "VALID_GOVERNMENT",
        "confidence": "SPECULATIVE",
        "evidence": (
            "Phase 3 observed camera candidate IN39; owner decision I.5: "
            "uncertain / likely not real (speculative)."
        ),
        "speculative": True,
        "notes": "Lowest-confidence rule; owner flagged it as probably not real.",
    },
    # ----------------------------------------------------------------------
    # Jordanian private (numeric)
    # ----------------------------------------------------------------------
    {
        "id": "JOR_PRIVATE_7",
        "pattern": r"^\d{7}$",
        "classification": "VALID_JORDANIAN",
        "confidence": "HIGH",
        "evidence": (
            "Primary private length (owner decision I.1: 5-7 primary); Phase 3: "
            "2,002 crops; Phase 7: 95 verified."
        ),
        "speculative": False,
        "notes": "Dominant observed private format.",
    },
    {
        "id": "JOR_PRIVATE_6",
        "pattern": r"^\d{6}$",
        "classification": "VALID_JORDANIAN",
        "confidence": "HIGH",
        "evidence": (
            "Primary private length (owner decision I.1); Phase 3: 931 crops; "
            "Phase 7: 12 verified."
        ),
        "speculative": False,
        "notes": "",
    },
    {
        "id": "JOR_PRIVATE_5",
        "pattern": r"^\d{5}$",
        "classification": "VALID_JORDANIAN",
        "confidence": "HIGH",
        "evidence": (
            "Primary private length (owner decision I.1); Phase 3: 177 crops; "
            "0 in the verified 126."
        ),
        "speculative": False,
        "notes": "No verified sample in the 126; accepted per owner decision.",
    },
    {
        "id": "JOR_PRIVATE_4",
        "pattern": r"^\d{4}$",
        "classification": "VALID_JORDANIAN",
        "confidence": "HIGH",
        "evidence": (
            "Historical private length (owner decision I.1: 3-4 historical); "
            "Phase 3: 71 crops; Phase 7: 1 verified."
        ),
        "speculative": False,
        "notes": "",
    },
    {
        "id": "JOR_PRIVATE_3",
        "pattern": r"^\d{3}$",
        "classification": "VALID_JORDANIAN",
        "confidence": "HIGH",
        "evidence": (
            "Historical private length (owner decision I.1: 3-4 historical); "
            "Phase 3: 16 crops; Phase 7: 3 verified."
        ),
        "speculative": False,
        "notes": "Owner-confirmed as accepted (historical).",
    },
    # ----------------------------------------------------------------------
    # Foreign-likely (classification only - never rejection)
    # ----------------------------------------------------------------------
    {
        "id": "KSA_LIKELY",
        "pattern": r"^\d{4}[A-Z]{3}$",
        "classification": "FOREIGN_LIKELY",
        "confidence": "HIGH",
        "evidence": (
            "All 16 held-out Saudi candidates match (Phase 3); owner-confirmed "
            "KSA shape."
        ),
        "speculative": False,
        "notes": "Classification only; rejection belongs to Phase 10.",
    },
    {
        "id": "IRQ_LIKELY_2L5D",
        "pattern": r"^\d{2}[A-Z]\d{5}$",
        "classification": "FOREIGN_LIKELY",
        "confidence": "SPECULATIVE",
        "evidence": (
            "Owner domain insight (decision I.2): the alphanumeric verified "
            "strings are likely Iraqi - this 2-digit/1-letter/5-digit shape."
        ),
        "speculative": True,
        "notes": "Flag only; no rejection. Confirm before ground truth.",
    },
    {
        "id": "IRQ_LIKELY_2L4D",
        "pattern": r"^\d{2}[A-Z]\d{4}$",
        "classification": "FOREIGN_LIKELY",
        "confidence": "SPECULATIVE",
        "evidence": (
            "Owner domain insight (decision I.2): likely Iraqi - 2-digit/"
            "1-letter/4-digit shape."
        ),
        "speculative": True,
        "notes": "Flag only; no rejection. Confirm before ground truth.",
    },
    {
        "id": "UAE_LIKELY_L5D",
        "pattern": r"^[A-Z]\d{5}$",
        "classification": "FOREIGN_LIKELY",
        "confidence": "SPECULATIVE",
        "evidence": (
            "Owner domain insight (decision I.2): likely Dubai/UAE - 1-letter/"
            "5-digit shape."
        ),
        "speculative": True,
        "notes": (
            "Tested after JOR_OBS_X so X78572 classifies as an observed Jordanian "
            "shape; flag only, no rejection."
        ),
    },
]

# Fallback (no rule matched, length 3-8, [A-Z0-9]) -> INVALID_FORMAT, handled by
# the validator; it is intentionally NOT a rule entry.
FALLBACK_CLASSIFICATION = "INVALID_FORMAT"

if __name__ == "__main__":  # tiny self-check: IDs must be unique and non-empty
    ids = [r["id"] for r in RULES]
    assert ids and len(ids) == len(set(ids)), f"duplicate/empty rule IDs: {ids}"
    for r in RULES:
        assert set(r) >= {"id", "pattern", "classification", "confidence",
                          "evidence", "speculative", "notes"}, r
    print(f"rules.py OK: {len(RULES)} rules, unique IDs, schema complete")
