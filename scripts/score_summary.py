#!/usr/bin/env python3
"""Summarize eval scores from a results JSONL file."""

import json
import sys
from pathlib import Path


def summarize(path: str) -> None:
    rows = [json.loads(line) for line in open(path)]
    scored = [r for r in rows if r.get("eval") and "score" in r["eval"]]
    scores = [r["eval"]["score"] for r in scored]

    if not scores:
        print("No scored tasks found.")
        return

    non_zero = [s for s in scores if s > 0]
    zero = [r for r in scored if r["eval"]["score"] == 0]

    print(f"File: {Path(path).name}")
    print(f"Tasks: {len(rows)}  |  Scored: {len(scored)}  |  Non-zero: {len(non_zero)}")
    print(f"Average score: {sum(scores) / len(scores):.4f}")
    print(f"Average (non-zero only): {sum(non_zero) / len(non_zero):.4f}" if non_zero else "")
    print(f"Min: {min(scores):.4f}  |  Max: {max(scores):.4f}")
    print()

    # Tier breakdown
    m_pass = m_total = g_pass = g_total = i_pass = i_total = 0
    for r in scored:
        ev = r["eval"]
        for tier, p, t in [
            ("mandatory", m_pass, m_total),
            ("good_to_have", g_pass, g_total),
            ("ideal", i_pass, i_total),
        ]:
            items = ev.get(tier, [])
            if tier == "mandatory":
                m_pass += sum(items)
                m_total += len(items)
            elif tier == "good_to_have":
                g_pass += sum(items)
                g_total += len(items)
            else:
                i_pass += sum(items)
                i_total += len(items)

    print("Tier breakdown (across all tasks):")
    print(f"  mandatory:    {m_pass}/{m_total} ({m_pass/m_total:.0%})" if m_total else "")
    print(f"  good_to_have: {g_pass}/{g_total} ({g_pass/g_total:.0%})" if g_total else "")
    print(f"  ideal:        {i_pass}/{i_total} ({i_pass/i_total:.0%})" if i_total else "")
    print()

    # Top and bottom tasks
    by_score = sorted(scored, key=lambda r: r["eval"]["score"], reverse=True)
    print("Top 5:")
    for r in by_score[:5]:
        print(f"  {r['id']:10s}  {r['eval']['score']:.4f}")
    print("Bottom 5 (scored > 0):" if non_zero else "Zero-score tasks:")
    bottom = sorted([r for r in scored if r["eval"]["score"] > 0], key=lambda r: r["eval"]["score"])
    for r in bottom[:5]:
        print(f"  {r['id']:10s}  {r['eval']['score']:.4f}")
    print()

    # Zero-score tasks with mandatory failures
    if zero:
        print(f"Zero-score tasks ({len(zero)}):")
        for r in zero:
            ev = r["eval"]
            m = ev.get("mandatory", [])
            print(f"  {r['id']:10s}  mandatory: {sum(m)}/{len(m)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "service/results/async-openai_20260223_063714.jsonl"
    summarize(path)
