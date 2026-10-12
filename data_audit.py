"""Read-only audit of the selected Excel workbook; never reconstructs respondents.

Usage: python data_audit.py [workbook.xlsx]
The default is the repository-local dataset_2.xlsx. Aggregate frequency tables are reported as
such and are rejected as inputs to respondent-level model fitting.
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
from pathlib import Path
from openpyxl import load_workbook

DEFAULT = Path(__file__).resolve().parent / "dataset_2.xlsx"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def audit(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")
    wb = load_workbook(path, data_only=True, read_only=False)
    result = {"file": str(path), "sha256": sha256(path), "sheets": []}
    aggregate = False
    for ws in wb.worksheets:
        item_rows = []
        for r in range(1, ws.max_row + 1):
            for label_col, bins, total_col in ((1, range(2, 7), 7), (9, range(10, 15), 15)):
                label = ws.cell(r, label_col).value
                if isinstance(label, str) and re.fullmatch(r"(?:PM|EU|IU|SM|MD|IH|IR|CA)[1-9]", label.strip()):
                    counts = [ws.cell(r, c).value for c in bins]
                    if all(isinstance(v, (int, float)) for v in counts):
                        item_rows.append({"item": label.strip(), "bin_counts": [int(v) for v in counts],
                                          "sum_of_bins": int(sum(counts)),
                                          "displayed_total": ws.cell(r, total_col).value,
                                          "bins_match_total": sum(counts) == ws.cell(r, total_col).value})
        study = [ws.cell(r, 2).value for r in range(8, 11)]
        age = [ws.cell(r, 2).value for r in range(14, 19)]
        gender = [ws.cell(r, 8).value for r in range(8, 11)]
        confidence = [ws.cell(r, 8).value for r in range(14, 19)]
        site_counts = [ws.cell(r, 4).value for r in range(1, 4)]
        summary = {
            "name": ws.title,
            "shape": [ws.max_row, ws.max_column],
            "item_frequency_rows": len(item_rows),
            "practice_item_rows": sum(not x["item"].startswith("CA") for x in item_rows),
            "CA_item_rows": sum(x["item"].startswith("CA") for x in item_rows),
            "item_totals_match_bins": sum(x["bins_match_total"] for x in item_rows),
            "item_total_mismatches": [x["item"] for x in item_rows if not x["bins_match_total"]],
            "item_total_range": [min((x["sum_of_bins"] for x in item_rows), default=None),
                                 max((x["sum_of_bins"] for x in item_rows), default=None)],
            "site_total_sum": sum(x for x in site_counts if isinstance(x, (int, float))),
            "demographic_category_sums": {
                "study_level": sum(x for x in study if isinstance(x, (int, float))),
                "age_group": sum(x for x in age if isinstance(x, (int, float))),
                "gender": sum(x for x in gender if isinstance(x, (int, float))),
                "computer_confidence": sum(x for x in confidence if isinstance(x, (int, float))),
            },
            "respondent_level_records_present": False,
        }
        summary["item_rows"] = item_rows
        result["sheets"].append(summary)
        aggregate = aggregate or len(item_rows) >= 50
    result["data_shape"] = "aggregate_frequency_summaries" if aggregate else "requires_manual_schema_review"
    return result

def main() -> int:
    path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT
    try:
        report = audit(path.resolve())
    except Exception as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    if report["data_shape"] == "aggregate_frequency_summaries":
        print("\nSTOP: Workbook contains marginal frequency counts, not linked respondent rows.")
        print("No student records, TOTAL scores, predictions, or model metrics were constructed.")
        return 3
    print("\nSchema review required; this audit does not certify respondent provenance or outcome validity.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
