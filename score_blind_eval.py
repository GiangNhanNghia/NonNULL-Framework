"""
NonNULL Framework V1.0 - Double-Blind Evaluation Decoder & Inter-Rater Reliability (Cohen's Kappa)
File: score_blind_eval.py

Decodes anonymized blind_*.csv files using their matching blind_*_key.json, computes
per-model mean scores, Win/Tie/Loss rates, exact agreement, and Cohen's Kappa between
independent raters, and optionally exports a publication-ready Markdown summary table.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


CRITERIA = ("Cau_truc_0_2", "Bao_ve_yeu_the_0_1", "Do_sau_0_2")


def parse_score(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def cohens_kappa(y1: List[int], y2: List[int]) -> float:
    """Computes unweighted Cohen's Kappa for two aligned lists of integer ratings."""
    if not y1 or len(y1) != len(y2):
        return 0.0
    n = len(y1)
    labels = sorted(set(y1) | set(y2))
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    pe = sum((y1.count(lbl) / n) * (y2.count(lbl) / n) for lbl in labels)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


def decode_blind_evaluation(csv_path: str, key_path: str) -> Dict:
    with open(key_path, "r", encoding="utf-8") as fk:
        key_list = json.load(fk)
    key_map = {int(item["scenario_id"]): item for item in key_list}

    per_model = defaultdict(lambda: {"cau_truc": [], "bao_ve": [], "do_sau": [], "total": []})
    scenario_totals: Dict[int, Dict[str, float]] = defaultdict(dict)
    r1_pairs: Dict[str, List[int]] = defaultdict(list)
    r2_pairs: Dict[str, List[int]] = defaultdict(list)

    rows_scored = 0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fc:
        reader = csv.DictReader(fc)
        for row in reader:
            sc_id = int(row["Scenario_ID"])
            blind_lbl = row["Blind_Model"].strip()
            real_model = key_map[sc_id]["mapping"][blind_lbl]

            r1_vals = [parse_score(row.get(f"Rater1_{c}")) for c in CRITERIA]
            r2_vals = [parse_score(row.get(f"Rater2_{c}")) for c in CRITERIA]

            has_r1 = all(v is not None for v in r1_vals)
            has_r2 = all(v is not None for v in r2_vals)
            if not has_r1 and not has_r2:
                continue

            rows_scored += 1
            if has_r1 and has_r2:
                avg_vals = [(a + b) / 2.0 for a, b in zip(r1_vals, r2_vals)]
                for cname, a, b in zip(CRITERIA, r1_vals, r2_vals):
                    r1_pairs[cname].append(int(round(a)))
                    r2_pairs[cname].append(int(round(b)))
            elif has_r1:
                avg_vals = list(r1_vals)
            else:
                avg_vals = list(r2_vals)

            tot = sum(avg_vals)
            per_model[real_model]["cau_truc"].append(avg_vals[0])
            per_model[real_model]["bao_ve"].append(avg_vals[1])
            per_model[real_model]["do_sau"].append(avg_vals[2])
            per_model[real_model]["total"].append(tot)
            scenario_totals[sc_id][real_model] = tot

    summary = {"rows_scored": rows_scored, "models": {}, "head_to_head": {}, "inter_rater": {}}
    for m_name, d in per_model.items():
        n = max(1, len(d["total"]))
        summary["models"][m_name] = {
            "n_scenarios": len(d["total"]),
            "cau_truc_avg": round(sum(d["cau_truc"]) / n, 4),
            "bao_ve_avg": round(sum(d["bao_ve"]) / n, 4),
            "do_sau_avg": round(sum(d["do_sau"]) / n, 4),
            "total_avg_5": round(sum(d["total"]) / n, 4),
        }

    # Head-to-head Win / Tie / Loss (NonNULL vs B_VOL4 or first two models)
    model_names = sorted(per_model.keys())
    if "NonNULL" in model_names and "B_VOL4" in model_names:
        m_a, m_b = "NonNULL", "B_VOL4"
    elif len(model_names) >= 2:
        m_a, m_b = model_names[0], model_names[1]
    else:
        m_a, m_b = None, None

    if m_a and m_b:
        wins = ties = losses = 0
        for sc_id, sc_scores in scenario_totals.items():
            if m_a in sc_scores and m_b in sc_scores:
                if sc_scores[m_a] > sc_scores[m_b]:
                    wins += 1
                elif sc_scores[m_a] < sc_scores[m_b]:
                    losses += 1
                else:
                    ties += 1
        total_cmp = max(1, wins + ties + losses)
        summary["head_to_head"] = {
            "model_a": m_a,
            "model_b": m_b,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate_pct": round(100.0 * wins / total_cmp, 2),
            "win_or_tie_pct": round(100.0 * (wins + ties) / total_cmp, 2),
        }

    for cname in CRITERIA:
        if r1_pairs[cname]:
            exact = sum(1 for a, b in zip(r1_pairs[cname], r2_pairs[cname]) if a == b) / len(r1_pairs[cname])
            summary["inter_rater"][cname] = {
                "n_pairs": len(r1_pairs[cname]),
                "exact_agreement_pct": round(100.0 * exact, 2),
                "cohens_kappa": round(cohens_kappa(r1_pairs[cname], r2_pairs[cname]), 4),
            }

    return summary


def format_markdown(summary: Dict) -> str:
    lines = [
        "# Double-Blind Human / LLM-as-a-Judge Evaluation Summary",
        "",
        f"- **Scored rows**: `{summary['rows_scored']}`",
        "",
        "| Model | Scenarios | Structure (0-2) | Vulnerable Protection (0-1) | Depth (0-2) | Total (0-5) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, stats in summary["models"].items():
        lines.append(
            f"| **{m_name}** | {stats['n_scenarios']} | {stats['cau_truc_avg']:.3f} | "
            f"{stats['bao_ve_avg']:.3f} | {stats['do_sau_avg']:.3f} | **{stats['total_avg_5']:.3f}** |"
        )

    h2h = summary.get("head_to_head")
    if h2h:
        lines += [
            "",
            f"## Head-to-Head: `{h2h['model_a']}` vs `{h2h['model_b']}`",
            f"- **Wins**: {h2h['wins']} ({h2h['win_rate_pct']}%) | **Ties**: {h2h['ties']} | **Losses**: {h2h['losses']}",
            f"- **Win or Tie Rate**: **{h2h['win_or_tie_pct']}%**",
        ]

    irr = summary.get("inter_rater")
    if irr:
        lines += [
            "",
            "## Inter-Rater Reliability (Rater 1 vs Rater 2)",
            "| Criterion | Pairs | Exact Agreement (%) | Cohen's Kappa |",
            "| :--- | :---: | :---: | :---: |",
        ]
        for cname, istats in irr.items():
            lines.append(
                f"| `{cname}` | {istats['n_pairs']} | {istats['exact_agreement_pct']:.2f}% | **{istats['cohens_kappa']:.4f}** |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode blind evaluation CSV and compute Cohen's Kappa.")
    parser.add_argument("--csv", type=str, required=True, help="Path to scored blind_*.csv file")
    parser.add_argument("--key", type=str, required=True, help="Path to matching blind_*_key.json file")
    parser.add_argument("--out_json", type=str, default=None, help="Optional JSON summary output path")
    parser.add_argument("--out_md", type=str, default=None, help="Optional Markdown report output path")
    args = parser.parse_args()

    summary = decode_blind_evaluation(args.csv, args.key)
    md = format_markdown(summary)
    print(md)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fj:
            json.dump(summary, fj, ensure_ascii=False, indent=2)
        print(f"[Saved JSON] {args.out_json}")
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fm:
            fm.write(md)
        print(f"[Saved Markdown] {args.out_md}")


if __name__ == "__main__":
    main()
