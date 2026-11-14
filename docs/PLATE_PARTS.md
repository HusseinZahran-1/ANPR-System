# Separate plate code and number

Each final `ocr_results` entry now includes `plate_parts` in the manual demo,
general prediction and camera ingestion. Existing `text` remains unchanged.
Both the demo and live view display the separate code and number. Archived
older events without these fields continue to display their original reading;
they are not reprocessed or rewritten automatically.

```json
{
  "text": "CD90001",
  "plate_parts": {
    "full_text": "CD90001",
    "plate_code": "CD",
    "plate_number": "90001",
    "split_basis": "recognized_authority_prefix",
    "requires_human_review": true
  }
}
```

All parts are strings, preserving leading zeros. The final OCR reading is split
after any optional visual authority reader has selected its output. A separate
possible-authority suggestion does not replace the reading or provide a code.
There is no automatic C0-to-CD, 5N-to-SN or 5S-to-SN substitution.

The text-only helper uses the project's existing authority-code list, or an
explicit numeric separator such as `12-90001`. Seven-digit numeric readings
use the project's two-digit-code/five-digit-number assumption, exposed as
`numeric_2_plus_5_assumption`. This is inferred formatting metadata, not a
separate visual prediction, calibrated confidence or verification of country.
Unseparated numeric readings with ambiguous boundaries return null code and
number with `split_basis=unresolved`; the UI displays `Unresolved`. Empty and
unsupported readings do likewise. Six digits are not blindly split into 2+4
because 1+5 is also possible. A future visual boundary reader needs reviewed
code/number labels and independent evaluation.

This adds structured output, not an OCR accuracy improvement. Human review
remains required. No original files, camera archives or model weights change.
