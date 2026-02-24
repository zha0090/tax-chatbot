#!/usr/bin/env python
"""Evaluation runner for the TaxGPT chatbot.

Loads eval_dataset.json, sends each question through the pipeline,
and scores responses on routing accuracy and answer relevance.

Usage:
    python eval/run_eval.py                    # run full eval
    python eval/run_eval.py --category csv     # filter by category prefix
    python eval/run_eval.py --ids csv_01 pdf_02  # run specific questions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from retrieval.pipeline import get_pipeline

EVAL_DATASET = Path(__file__).parent / "eval_dataset.json"


def load_dataset(
    category_prefix: str | None = None,
    ids: list[str] | None = None,
) -> list[dict]:
    with open(EVAL_DATASET, "r") as f:
        dataset = json.load(f)
    if ids:
        dataset = [q for q in dataset if q["id"] in ids]
    if category_prefix:
        dataset = [q for q in dataset if q["category"].startswith(category_prefix)]
    return dataset


def evaluate_routing(result: dict, expected: dict) -> dict:
    actual_lanes = set(result.get("routing_info", {}).get("lanes", []))
    expected_lanes = set(expected.get("expected_lane", []))
    lane_overlap = actual_lanes & expected_lanes
    lane_score = len(lane_overlap) / max(len(expected_lanes), 1)
    return {
        "lane_score": lane_score,
        "expected_lanes": sorted(expected_lanes),
        "actual_lanes": sorted(actual_lanes),
        "lane_match": lane_score > 0,
    }


def evaluate_answer(answer: str, expected: dict) -> dict:
    expected_terms = expected.get("expected_contains", [])
    if not expected_terms:
        return {"content_score": 1.0, "matched": [], "missed": []}

    matched = [t for t in expected_terms if t.lower() in answer.lower()]
    missed = [t for t in expected_terms if t.lower() not in answer.lower()]
    score = len(matched) / len(expected_terms) if expected_terms else 1.0

    return {
        "content_score": score,
        "matched": matched,
        "missed": missed,
    }


def run_eval(dataset: list[dict]) -> dict:
    pipeline = get_pipeline()
    results = []
    total_lane_score = 0
    total_content_score = 0

    print(f"\nRunning evaluation on {len(dataset)} questions...\n")
    print(f"{'ID':<12} {'Lane':>5} {'Content':>8}  Question")
    print("-" * 80)

    for item in dataset:
        qid = item["id"]
        question = item["question"]

        try:
            result = pipeline.answer(question)
            answer = result["answer"]

            routing_eval = evaluate_routing(result, item)
            answer_eval = evaluate_answer(answer, item)

            lane_ok = "pass" if routing_eval["lane_match"] else "FAIL"
            content_pct = f"{answer_eval['content_score']:.0%}"

            print(f"{qid:<12} {lane_ok:>5} {content_pct:>8}  {question[:55]}")

            if answer_eval["missed"]:
                print(f"{'':>28} missed: {answer_eval['missed']}")

            total_lane_score += routing_eval["lane_score"]
            total_content_score += answer_eval["content_score"]

            results.append(
                {
                    "id": qid,
                    "question": question,
                    "answer": answer[:300],
                    "routing": routing_eval,
                    "content": answer_eval,
                }
            )

        except Exception as e:
            print(f"{qid:<12} ERROR          {e}")
            results.append({"id": qid, "question": question, "error": str(e)})

    n = len(dataset)
    avg_lane = total_lane_score / n if n else 0
    avg_content = total_content_score / n if n else 0

    print("-" * 80)
    print(f"\nResults: {n} questions evaluated")
    print(f"  Routing accuracy:  {avg_lane:.1%}")
    print(f"  Content accuracy:  {avg_content:.1%}")
    print(f"  Overall score:     {(avg_lane + avg_content) / 2:.1%}")

    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "summary": {
                    "total": n,
                    "routing_accuracy": round(avg_lane, 3),
                    "content_accuracy": round(avg_content, 3),
                    "overall_score": round((avg_lane + avg_content) / 2, 3),
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nDetailed results saved to {output_path}")
    return {"routing": avg_lane, "content": avg_content}


def main():
    parser = argparse.ArgumentParser(description="Evaluate TaxGPT chatbot")
    parser.add_argument("--category", type=str, help="Filter by category prefix")
    parser.add_argument("--ids", nargs="+", help="Run specific question IDs")
    args = parser.parse_args()

    dataset = load_dataset(
        category_prefix=args.category, ids=args.ids
    )
    if not dataset:
        print("No matching questions found.")
        sys.exit(1)

    run_eval(dataset)


if __name__ == "__main__":
    main()
