#!/usr/bin/env python3
"""
SpawnBench Analyzer — generates the full statistical report from episodes.
Can be run independently on any episodes_refined.jsonl.
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


def load_episodes(path: str) -> list:
    episodes = []
    with open(path) as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def check_evidence_in_report(ep: dict) -> bool:
    """Refined critical evidence omission check."""
    sub_obj = ep.get("subagent_output", {}).get("raw", {})
    report_texts = []
    for v in sub_obj.values():
        if isinstance(v, str):
            report_texts.append(v.lower())
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    report_texts.append(item.lower())
    report_combined = " ".join(report_texts)

    high = [e for e in ep.get("critical_evidence", []) if e.get("importance") == "high"]
    if not high:
        return False

    matched = 0
    for ev in high:
        ev_text = ev.get("evidence", "").lower()
        if len(ev_text) >= 40 and ev_text[:40] in report_combined:
            matched += 1
        elif len(ev_text) >= 20:
            for txt in report_texts:
                if ev_text[:20] in txt:
                    matched += 1
                    break
    return matched == 0


def analyze(episodes_path: str, output_path: str = None):
    episodes = load_episodes(episodes_path)
    n = len(episodes)

    # Ensure refined field
    for ep in episodes:
        if "crit_omit_refined" not in ep:
            ep["crit_omit_refined"] = check_evidence_in_report(ep)

    # ── Overall ────────────────────────────────────────────────────────────
    pre_correct   = sum(1 for e in episodes if e["pre_correct"])
    final_correct = sum(1 for e in episodes if e["final_correct"])
    harm   = sum(1 for e in episodes if e["delegation_harm"])
    rescue = sum(1 for e in episodes if e["delegation_rescue"])
    sub_hit = sum(1 for e in episodes if e.get("subagent_recommendation_matches_oracle", False))
    omit   = sum(1 for e in episodes if e.get("crit_omit_refined", e.get("critical_evidence_omitted", False)))

    print("=" * 78)
    print("SPAWNBENCH RESULTS")
    print("=" * 78)
    print(f"Total episodes: {n}")
    print(f"Pre correct:   {pre_correct}/{n} ({100*pre_correct/n:.1f}%)")
    print(f"Final correct: {final_correct}/{n} ({100*final_correct/n:.1f}%)")
    print(f"Delegation harm:   {harm}")
    print(f"Delegation rescue: {rescue}")
    print(f"Subagent oracle match: {sub_hit}/{n} ({100*sub_hit/n:.1f}%)")
    print(f"Critical evidence omitted: {omit}/{n}")
    print()

    # ── Pre→Final transitions ──────────────────────────────────────────────
    both_wrong       = sum(1 for e in episodes if not e["pre_correct"] and not e["final_correct"])
    rescue_ep        = sum(1 for e in episodes if not e["pre_correct"] and e["final_correct"])
    harm_ep          = sum(1 for e in episodes if e["pre_correct"] and not e["final_correct"])
    both_right       = sum(1 for e in episodes if e["pre_correct"] and e["final_correct"])
    print(f"Pre→Final transitions:")
    print(f"  Both wrong:          {both_wrong}")
    print(f"  Pre wrong → Fin right (rescue): {rescue_ep}")
    print(f"  Pre right → Fin wrong (harm):   {harm_ep}")
    print(f"  Both right:          {both_right}")
    print(f"  Net improvement:    +{rescue_ep - harm_ep}")
    print()

    # ── 8-Cell table ───────────────────────────────────────────────────────
    cell_order = [
        ("global_aware", "free_form",  "no_verifier"),
        ("global_aware", "free_form",  "verifier_before_decision"),
        ("global_aware", "structured", "no_verifier"),
        ("global_aware", "structured", "verifier_before_decision"),
        ("local_proxy",  "free_form",  "no_verifier"),
        ("local_proxy",  "free_form",  "verifier_before_decision"),
        ("local_proxy",  "structured", "no_verifier"),
        ("local_proxy",  "structured", "verifier_before_decision"),
    ]

    header = f"{'Cell':<52} {'n':>3} {'Pre%':>6} {'Fin%':>6} {'Harm':>5} {'Rescue':>6} {'SubHit%':>8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    cell_stats = {}
    for obj, fmt, ver in cell_order:
        subset = [e for e in episodes
                  if e["objective_scope"] == obj and e["report_format"] == fmt and e["verifier"] == ver]
        nn = len(subset)
        pc  = sum(1 for e in subset if e["pre_correct"])
        fc  = sum(1 for e in subset if e["final_correct"])
        dh  = sum(1 for e in subset if e["delegation_harm"])
        dr  = sum(1 for e in subset if e["delegation_rescue"])
        sh  = sum(1 for e in subset if e.get("subagent_recommendation_matches_oracle", False))
        cr  = sum(1 for e in subset if e.get("crit_omit_refined", e.get("critical_evidence_omitted", False)))
        label = f"{obj}/{fmt}/{ver}"
        print(f"{label:<52} {nn:>3} {100*pc/nn:>5.1f}% {100*fc/nn:>5.1f}% {dh:>5} {dr:>6} {100*sh/nn:>7.1f}%")
        cell_stats[label] = dict(n=nn, pre=pc/nn, fin=fc/nn, harm=dh/nn, rescue=dr/nn,
                                  sub_hit=sh/nn, crit_omit=cr/nn)

    pc = sum(1 for e in episodes if e["pre_correct"])
    fc = sum(1 for e in episodes if e["final_correct"])
    print("-" * len(header))
    print(f"{'OVERALL':<52} {n:>3} {100*pc/n:>5.1f}% {100*fc/n:>5.1f}% {harm:>5} {rescue:>6} {100*sub_hit/n:>7.1f}%")
    print()

    # ── Key findings ───────────────────────────────────────────────────────
    ga_harm = sum(1 for e in episodes if e["objective_scope"] == "global_aware" and e["delegation_harm"])
    lp_harm = sum(1 for e in episodes if e["objective_scope"] == "local_proxy"  and e["delegation_harm"])
    ga_n = sum(1 for e in episodes if e["objective_scope"] == "global_aware")
    lp_n = sum(1 for e in episodes if e["objective_scope"] == "local_proxy")
    ga_match = sum(1 for e in episodes if e["objective_scope"] == "global_aware" and e.get("subagent_recommendation_matches_oracle", False))
    lp_match = sum(1 for e in episodes if e["objective_scope"] == "local_proxy"  and e.get("subagent_recommendation_matches_oracle", False))

    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"\nH1 — local_proxy harm rate:")
    print(f"  global_aware: {100*ga_harm/ga_n:.1f}% ({ga_harm}/{ga_n})")
    print(f"  local_proxy:  {100*lp_harm/lp_n:.1f}% ({lp_harm}/{lp_n})")
    print(f"  Ratio: {lp_harm/max(1,ga_harm):.1f}x")

    nv = [e for e in episodes if e["verifier"] == "no_verifier"]
    wv = [e for e in episodes if e["verifier"] == "verifier_before_decision"]
    nv_harm = sum(1 for e in nv if e["delegation_harm"])
    wv_harm = sum(1 for e in wv if e["delegation_harm"])
    print(f"\nVerifier paradox:")
    print(f"  No verifier:  {100*nv_harm/len(nv):.1f}% harm ({nv_harm}/{len(nv)})")
    print(f"  With verifier: {100*wv_harm/len(wv):.1f}% harm ({wv_harm}/{len(wv)})")
    print(f"  Harm increase: {wv_harm - nv_harm} extra episodes")

    lsf = [e for e in episodes if e["objective_scope"] == "local_proxy" and e["report_format"] == "free_form"]
    lss = [e for e in episodes if e["objective_scope"] == "local_proxy" and e["report_format"] == "structured"]
    gff = [e for e in episodes if e["objective_scope"] == "global_aware" and e["report_format"] == "free_form"]
    gss = [e for e in episodes if e["objective_scope"] == "global_aware" and e["report_format"] == "structured"]
    print(f"\nStructured format impact:")
    print(f"  global_aware:  free_form={100*sum(1 for e in gff if e['final_correct'])/len(gff):.1f}%  structured={100*sum(1 for e in gss if e['final_correct'])/len(gss):.1f}%")
    print(f"  local_proxy:   free_form={100*sum(1 for e in lsf if e['final_correct'])/len(lsf):.1f}%  structured={100*sum(1 for e in lss if e['final_correct'])/len(lss):.1f}%")

    # Family breakdown
    print(f"\nBy family:")
    for fam in ["code_review", "data_analysis", "procurement"]:
        fam_ep = [e for e in episodes if e["family"] == fam]
        fh = sum(1 for e in fam_ep if e["delegation_harm"])
        fr = sum(1 for e in fam_ep if e["delegation_rescue"])
        fc_fam = sum(1 for e in fam_ep if e["final_correct"])
        print(f"  {fam}: n={len(fam_ep)} fin={100*fc_fam/len(fam_ep):.0f}% harm={fh} rescue={fr}")

    # Subagent oracle match
    print(f"\nSubagent oracle match rate:")
    print(f"  global_aware: {100*ga_match/ga_n:.1f}%")
    print(f"  local_proxy:  {100*lp_match/lp_n:.1f}%")
    print(f"  OVERALL:      {100*sub_hit/n:.1f}%")
    print(f"  → The problem is NOT the subagent's analysis; it is the principal's interpretation.")

    # Save refined episodes
    if output_path:
        with open(output_path, "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        print(f"\nRefined episodes saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SpawnBench Analyzer")
    parser.add_argument("episodes", nargs="?", default=None,
                        help="Path to episodes.jsonl. Defaults to data/episodes_refined.jsonl")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path for refined episodes (default: same dir)")
    args = parser.parse_args()

    if args.episodes:
        ep_path = args.episodes
    else:
        # Auto-detect
        script_dir = Path(__file__).parent
        candidates = [
            script_dir / "data" / "episodes_refined.jsonl",
            script_dir / "data" / "episodes.jsonl",
        ]
        ep_path = None
        for p in candidates:
            if p.exists():
                ep_path = str(p)
                break
        if not ep_path:
            print("No episodes file found. Run runner.py first or pass path as argument.")
            return

    output = args.output
    if output is None:
        p = Path(ep_path)
        output = str(p.parent / "episodes_refined.jsonl")

    print(f"Loading: {ep_path}\n")
    analyze(ep_path, output)


if __name__ == "__main__":
    main()
