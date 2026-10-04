from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

AXES = (
    "mean_time",
    "tail10_time",
    "continuation_preservation",
    "fleet_option_reserve",
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def _wtl(
    rows: list[dict[str, Any]],
    axis: str,
    *,
    tolerance: float,
) -> dict[str, Any]:
    deltas = np.asarray(
        [
            float(row["adaptive_minus_center"][axis])
            for row in rows
        ],
        dtype=np.float64,
    )
    wins = int(np.sum(deltas > tolerance))
    losses = int(np.sum(deltas < -tolerance))
    ties = int(len(deltas) - wins - losses)
    return {
        "n": int(len(deltas)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "non_tied_win_rate": (
            float(wins / (wins + losses))
            if (wins + losses) > 0
            else None
        ),
    }


def _cap_signature(capabilities: list[str]) -> str:
    return "+".join(sorted(capabilities)) or "none"


def _parent_context(row: dict[str, Any]) -> str:
    a = _cap_signature(row["parent_a_capabilities"])
    b = _cap_signature(row["parent_b_capabilities"])
    return f"{a} || {b}" if a <= b else f"{b} || {a}"


def _fourcap(row: dict[str, Any], side: str) -> bool:
    return len(row[f"{side}_certified_capabilities"]) == len(AXES)


def _transition(adaptive: bool, center: bool) -> str:
    if adaptive and not center:
        return "rescue"
    if center and not adaptive:
        return "loss"
    if adaptive and center:
        return "both"
    return "neither"


def _effect_shape(row: dict[str, Any]) -> dict[str, float]:
    groups = row.get("adaptive_metadata", {}).get("groups", [])
    if not groups:
        return {
            "mean_alpha_deviation": 0.0,
            "mean_eta_deviation": 0.0,
            "max_alpha_deviation": 0.0,
            "max_eta_deviation": 0.0,
        }
    alpha = np.asarray(
        [abs(float(g["alpha"]) - 0.5) for g in groups],
        dtype=np.float64,
    )
    eta = np.asarray(
        [abs(float(g["eta"]) - 1.0) for g in groups],
        dtype=np.float64,
    )
    return {
        "mean_alpha_deviation": float(np.mean(alpha)),
        "mean_eta_deviation": float(np.mean(eta)),
        "max_alpha_deviation": float(np.max(alpha)),
        "max_eta_deviation": float(np.max(eta)),
    }


def _group_summary(
    rows: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    dominance = Counter(str(row["dominance"]) for row in rows)
    fourcap = Counter(
        _transition(_fourcap(row, "adaptive"), _fourcap(row, "center"))
        for row in rows
    )
    gate = Counter(
        _transition(
            bool(row["adaptive_dual_gate_pass"]),
            bool(row["center_dual_gate_pass"]),
        )
        for row in rows
    )
    effects = [_effect_shape(row) for row in rows]
    return {
        "n": len(rows),
        "axis_stats": {
            axis: _wtl(rows, axis, tolerance=tolerance)
            for axis in AXES
        },
        "dominance": dict(dominance),
        "fourcap_transition": dict(fourcap),
        "dual_gate_transition": dict(gate),
        "exact_identical_count": int(
            sum(bool(row["exact_identical_child"]) for row in rows)
        ),
        "mean_alpha_deviation": float(
            np.mean([item["mean_alpha_deviation"] for item in effects])
        ),
        "mean_eta_deviation": float(
            np.mean([item["mean_eta_deviation"] for item in effects])
        ),
    }


def _write_table(
    path: Path,
    groups: dict[str, list[dict[str, Any]]],
    *,
    tolerance: float,
    min_count: int,
) -> None:
    fields = [
        "group",
        "n",
        "fourcap_rescue",
        "fourcap_loss",
        "gate_rescue",
        "gate_loss",
        "adaptive_dominance",
        "center_dominance",
        "identical",
    ]
    for axis in AXES:
        fields += [
            f"{axis}_mean_delta",
            f"{axis}_wins",
            f"{axis}_ties",
            f"{axis}_losses",
        ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for key, rows in sorted(
            groups.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            if len(rows) < min_count:
                continue
            summary = _group_summary(rows, tolerance=tolerance)
            out: dict[str, Any] = {
                "group": key,
                "n": len(rows),
                "fourcap_rescue": summary["fourcap_transition"].get("rescue", 0),
                "fourcap_loss": summary["fourcap_transition"].get("loss", 0),
                "gate_rescue": summary["dual_gate_transition"].get("rescue", 0),
                "gate_loss": summary["dual_gate_transition"].get("loss", 0),
                "adaptive_dominance": summary["dominance"].get("adaptive", 0),
                "center_dominance": summary["dominance"].get("center", 0),
                "identical": summary["exact_identical_count"],
            }
            for axis in AXES:
                stats = summary["axis_stats"][axis]
                out[f"{axis}_mean_delta"] = stats["mean_delta"]
                out[f"{axis}_wins"] = stats["wins"]
                out[f"{axis}_ties"] = stats["ties"]
                out[f"{axis}_losses"] = stats["losses"]
            writer.writerow(out)


def analyze(
    run_dir: Path,
    *,
    tolerance: float,
    min_rule_count: int,
    min_context_count: int,
) -> Path:
    rows = _load_rows(run_dir / "pair_results.jsonl")
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )

    noncenter = [
        row
        for row in rows
        if int(row["adaptive_rule_active_terms"]) > 0
    ]

    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_rule_context: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in noncenter:
        rule_id = str(row["adaptive_rule_id"])
        context = _parent_context(row)
        by_rule[rule_id].append(row)
        by_context[context].append(row)
        by_rule_context[f"{rule_id} :: {context}"].append(row)

    rule_summaries = {}
    for rule_id, group in by_rule.items():
        base = _group_summary(group, tolerance=tolerance)
        base["active_term_count"] = int(
            group[0]["adaptive_rule_active_terms"]
        )
        base["gates"] = group[0]["adaptive_rule"]["gates"]
        base["masked_coefficients"] = group[0]["adaptive_rule"][
            "masked_coefficients"
        ]
        rule_summaries[rule_id] = base

    reliable_rules = []
    for rule_id, item in rule_summaries.items():
        if int(item["n"]) < min_rule_count:
            continue
        cont = item["axis_stats"]["continuation_preservation"]
        reliable_rules.append(
            {
                "rule_id": rule_id,
                "n": item["n"],
                "active_term_count": item["active_term_count"],
                "continuation_mean_delta": cont["mean_delta"],
                "continuation_wtl": [
                    cont["wins"],
                    cont["ties"],
                    cont["losses"],
                ],
                "fourcap_rescue_minus_loss": (
                    item["fourcap_transition"].get("rescue", 0)
                    - item["fourcap_transition"].get("loss", 0)
                ),
                "dominance_margin": (
                    item["dominance"].get("adaptive", 0)
                    - item["dominance"].get("center", 0)
                ),
            }
        )

    reliable_rules.sort(
        key=lambda item: (
            item["fourcap_rescue_minus_loss"],
            item["continuation_mean_delta"],
            item["continuation_wtl"][0] - item["continuation_wtl"][2],
            item["n"],
        ),
        reverse=True,
    )

    candidate_contexts = []
    for key, group in by_rule_context.items():
        if len(group) < min_context_count:
            continue
        item = _group_summary(group, tolerance=tolerance)
        cont = item["axis_stats"]["continuation_preservation"]
        rescue_margin = (
            item["fourcap_transition"].get("rescue", 0)
            - item["fourcap_transition"].get("loss", 0)
        )
        if (
            cont["mean_delta"] > 0.0
            or cont["wins"] > cont["losses"]
            or rescue_margin > 0
        ):
            candidate_contexts.append(
                {
                    "rule_context": key,
                    "fourcap_rescue_margin": rescue_margin,
                    **item,
                }
            )

    candidate_contexts.sort(
        key=lambda item: (
            item["fourcap_rescue_margin"],
            item["axis_stats"]["continuation_preservation"]["mean_delta"],
            item["n"],
        ),
        reverse=True,
    )

    fourcap_rescues = [
        row
        for row in noncenter
        if _fourcap(row, "adaptive") and not _fourcap(row, "center")
    ]
    fourcap_losses = [
        row
        for row in noncenter
        if _fourcap(row, "center") and not _fourcap(row, "adaptive")
    ]

    identical_noncenter = [
        {
            "pair_index": row["pair_index"],
            "rule_id": row["adaptive_rule_id"],
            "parent_context": _parent_context(row),
            "effect_shape": _effect_shape(row),
            "rule": row["adaptive_rule"],
        }
        for row in noncenter
        if bool(row["exact_identical_child"])
    ]

    top_cont = sorted(
        noncenter,
        key=lambda row: float(
            row["adaptive_minus_center"]["continuation_preservation"]
        ),
        reverse=True,
    )[:20]

    output = {
        "experiment": "v1143_offline_rule_context_analysis",
        "source_run": str(run_dir),
        "source_pair_count": len(rows),
        "source_summary": summary,
        "noncenter_pair_count": len(noncenter),
        "overall_noncenter_axis_stats": {
            axis: _wtl(noncenter, axis, tolerance=tolerance)
            for axis in AXES
        },
        "fourcap_rescue_count": len(fourcap_rescues),
        "fourcap_loss_count": len(fourcap_losses),
        "rule_summaries": rule_summaries,
        "parent_context_summaries": {
            key: _group_summary(group, tolerance=tolerance)
            for key, group in by_context.items()
        },
        "reliable_rule_ranking": reliable_rules,
        "candidate_rule_contexts": candidate_contexts,
        "top_continuation_pairs": [
            {
                "pair_index": row["pair_index"],
                "rule_id": row["adaptive_rule_id"],
                "parent_context": _parent_context(row),
                "continuation_delta": row["adaptive_minus_center"][
                    "continuation_preservation"
                ],
                "fourcap_transition": _transition(
                    _fourcap(row, "adaptive"),
                    _fourcap(row, "center"),
                ),
                "dominance": row["dominance"],
            }
            for row in top_cont
        ],
        "fourcap_rescue_pairs": [
            {
                "pair_index": row["pair_index"],
                "rule_id": row["adaptive_rule_id"],
                "parent_context": _parent_context(row),
                "deltas": row["adaptive_minus_center"],
            }
            for row in fourcap_rescues
        ],
        "fourcap_loss_pairs": [
            {
                "pair_index": row["pair_index"],
                "rule_id": row["adaptive_rule_id"],
                "parent_context": _parent_context(row),
                "deltas": row["adaptive_minus_center"],
            }
            for row in fourcap_losses
        ],
        "identical_noncenter_pairs": identical_noncenter,
        "guardrail": (
            "This is descriptive reuse of the same 95M development assay. "
            "It may define a context-selection hypothesis but cannot validate it."
        ),
    }

    out = run_dir / "v1143_rule_context_analysis.json"
    out.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _write_table(
        run_dir / "v1143_rule_table.csv",
        by_rule,
        tolerance=tolerance,
        min_count=min_rule_count,
    )
    _write_table(
        run_dir / "v1143_parent_context_table.csv",
        by_context,
        tolerance=tolerance,
        min_count=min_context_count,
    )
    _write_table(
        run_dir / "v1143_rule_context_table.csv",
        by_rule_context,
        tolerance=tolerance,
        min_count=min_context_count,
    )

    print("V1143_RULE_CONTEXT_ANALYSIS")
    print(
        f"SOURCE_PAIRS={len(rows)} NONCENTER={len(noncenter)} "
        f"FOURCAP_RESCUE={len(fourcap_rescues)} "
        f"FOURCAP_LOSS={len(fourcap_losses)}"
    )
    for axis in AXES:
        stats = output["overall_noncenter_axis_stats"][axis]
        print(
            f"NONCENTER {axis}: delta={stats['mean_delta']:.8f} "
            f"W/T/L={stats['wins']}/{stats['ties']}/{stats['losses']}"
        )
    print(
        "RELIABLE_RULES="
        + json.dumps(reliable_rules[:10], ensure_ascii=False)
    )
    print(f"CANDIDATE_RULE_CONTEXTS={len(candidate_contexts)}")
    print(f"IDENTICAL_NONCENTER={len(identical_noncenter)}")
    print(f"OUTPUT={out}")
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tolerance", type=float, default=1e-12)
    p.add_argument("--min-rule-count", type=int, default=5)
    p.add_argument("--min-context-count", type=int, default=3)
    return p


def main() -> None:
    args = parser().parse_args()
    analyze(
        Path(args.run_dir),
        tolerance=args.tolerance,
        min_rule_count=args.min_rule_count,
        min_context_count=args.min_context_count,
    )


if __name__ == "__main__":
    main()
