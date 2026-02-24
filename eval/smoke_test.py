#!/usr/bin/env python
"""Smoke test: sends 30 diverse queries to the running chatbot and prints results.

Usage: python eval/smoke_test.py
Requires the server to be running at http://127.0.0.1:8000
"""

import requests
import sys
import time

BASE_URL = "http://127.0.0.1:8000/api/chat/"

TESTS = [
    # -- Greetings & Chitchat --
    ("CHITCHAT", "hi"),
    ("CHITCHAT", "what can you do?"),
    ("CHITCHAT", "thank you"),

    # -- Broad queries (no specific entity) --
    ("BROAD", "What is the highest tax rate?"),
    ("BROAD", "What is the lowest tax rate?"),
    ("BROAD", "Which state has the highest tax rate?"),
    ("BROAD", "Which state has the lowest tax rate?"),
    ("BROAD", "Which taxpayer type pays the most tax overall?"),

    # -- State-specific --
    ("STATE", "What is the average tax rate in California?"),
    ("STATE", "How much total tax was owed in Texas?"),
    ("STATE", "Compare tax rates between New York and Florida"),

    # -- Taxpayer type specific --
    ("TYPE", "What is the average income for corporations?"),
    ("TYPE", "How many partnership records are there?"),
    ("TYPE", "What deduction type is most common for trusts?"),
    ("TYPE", "Compare tax rates between individuals and corporations"),

    # -- Filtered (type + state) --
    ("FILTERED", "What is the average tax rate for corporations in California?"),
    ("FILTERED", "Total tax owed by partnerships in Ohio"),
    ("FILTERED", "Average income for individuals in Florida"),

    # -- Year-based --
    ("YEAR", "What is the average tax rate for 2023?"),
    ("YEAR", "How does the total tax owed compare between 2019 and 2023?"),

    # -- Income source / deduction --
    ("SOURCE", "Which income source has the highest average income?"),
    ("SOURCE", "What is the average deduction for mortgage interest?"),

    # -- PDF / IRS questions --
    ("PDF", "What are the standard deduction amounts for 2023?"),
    ("PDF", "When is the deadline to file Form 1040?"),
    ("PDF", "What is the Taxpayer Advocate Service?"),
    ("PDF", "How do I report self-employment income?"),

    # -- PPT / Economics --
    ("PPT", "What happens when an excise tax is applied?"),
    ("PPT", "What is the effect of an excise tax with inelastic demand?"),

    # -- Edge cases --
    ("EDGE", "asdfghjkl"),
    ("EDGE", "What is the tax rate on Mars?"),
]


def run():
    passed = 0
    failed = 0
    errors = 0

    print(f"\n{'='*90}")
    print(f"  SMOKE TEST: {len(TESTS)} queries against http://127.0.0.1:8000")
    print(f"{'='*90}\n")

    for category, query in TESTS:
        try:
            t0 = time.time()
            resp = requests.post(BASE_URL, json={"query": query}, timeout=30)
            elapsed = time.time() - t0

            if resp.status_code != 200:
                status = "FAIL"
                answer = f"HTTP {resp.status_code}"
                failed += 1
            else:
                data = resp.json()
                answer = data.get("answer", "")
                lanes = data.get("routing", {}).get("lanes", [])
                sources = data.get("sources", [])

                is_empty = not answer or "no specific data" in answer.lower()
                is_error = "error" in answer.lower() and "encountered" in answer.lower()

                if is_error:
                    status = "ERR "
                    errors += 1
                elif is_empty and category not in ("EDGE",):
                    status = "WEAK"
                    failed += 1
                else:
                    status = "OK  "
                    passed += 1

            answer_preview = answer.replace("\n", " ")[:80]
            lane_str = ",".join(lanes) if 'lanes' in dir() else ""
            print(f"  [{status}] {elapsed:5.1f}s  {category:<10} {query[:40]:<42} {answer_preview}")

        except requests.ConnectionError:
            print(f"  [FAIL]        {category:<10} {query[:40]:<42} Server not reachable")
            errors += 1
        except Exception as e:
            print(f"  [ERR ]        {category:<10} {query[:40]:<42} {e}")
            errors += 1

    total = len(TESTS)
    print(f"\n{'='*90}")
    print(f"  Results: {passed}/{total} passed, {failed} weak/failed, {errors} errors")
    print(f"  Pass rate: {passed/total:.0%}")
    print(f"{'='*90}\n")

    return passed, failed, errors


if __name__ == "__main__":
    p, f, e = run()
    sys.exit(1 if e > 0 else 0)
