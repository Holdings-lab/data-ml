"""Evaluate claim-ID-aligned predictions against a labeled JSON dataset.

Dataset: {cases: [{id, expected_verdict, ...}]}; predictions: {cases: [{id, verdict}]}.
This reports label agreement, not the reliability of the labels themselves.
"""
import argparse
import json
from pathlib import Path

VERDICTS = {"supported", "refuted", "insufficient_evidence"}


def evaluate(dataset: dict, predictions: dict) -> dict:
    def indexed(data, key):
        result = {}
        for row in data["cases"]:
            if row["id"] in result or row[key] not in VERDICTS:
                raise ValueError("Duplicate IDs or invalid verdict")
            result[row["id"]] = row[key]
        return result
    expected, actual = indexed(dataset, "expected_verdict"), indexed(predictions, "verdict")
    if expected.keys() != actual.keys() or not expected:
        raise ValueError("Predictions must cover every evaluation case exactly once")
    correct = sum(actual[k] == v for k, v in expected.items())
    supported = [k for k, v in actual.items() if v == "supported"]
    false_supported = sum(expected[k] != "supported" for k in supported)
    return {"cases": len(expected), "accuracy": correct / len(expected),
            "supported_rate": len(supported) / len(expected),
            "target_supported_rate": 0.5,
            "target_met": len(supported) / len(expected) >= 0.5,
            "supported_precision": (len(supported) - false_supported) / len(supported) if supported else None,
            "false_supported_count": false_supported,
            "false_supported_rate": false_supported / len(expected),
            "insufficient_evidence_rate": sum(v == "insufficient_evidence" for v in actual.values()) / len(actual),
            "confusion": {truth: {pred: sum(expected[k] == truth and actual[k] == pred for k in expected)
                                    for pred in sorted(VERDICTS)} for truth in sorted(VERDICTS)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(evaluate(json.loads(args.dataset.read_text(encoding="utf-8-sig")),
                                 json.loads(args.predictions.read_text(encoding="utf-8-sig"))), indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
