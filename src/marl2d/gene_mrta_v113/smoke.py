from __future__ import annotations

import argparse
import json
from pathlib import Path

from marl2d.gene_mrta_v110.scenario_bank import (
    load_scenario_bank,
)
from marl2d.gene_mrta_v110.train import (
    GeneRecord,
    _active_parent_ids,
    _best_by_axis,
    _certified_capabilities,
    _quality_weight,
)
from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
)
from marl2d.gene_mrta_v113.route_tail import (
    rollout_route_tail_gene,
)


def _load_records(
    checkpoint: Path,
) -> dict[str, GeneRecord]:
    data = json.loads(
        checkpoint.read_text(
            encoding="utf-8"
        )
    )
    return {
        str(item["record_id"]): (
            GeneRecord.from_dict(
                item
            )
        )
        for item in data["records"]
    }


def _select_source_gene(
    records: dict[str, GeneRecord],
    archive_size: int,
    hybrid_limit: int,
    threshold: float,
) -> tuple[
    GeneRecord,
    tuple[str, ...],
]:
    active_ids, _ = (
        _active_parent_ids(
            records,
            archive_size,
            hybrid_limit,
        )
    )
    ceiling = _best_by_axis(
        records,
        active_ids,
    )

    ranked = []
    for record_id in active_ids:
        record = records[
            record_id
        ]
        certified = (
            _certified_capabilities(
                record,
                ceiling,
                threshold=threshold,
            )
        )
        ranked.append(
            (
                len(certified),
                _quality_weight(
                    record,
                    ceiling,
                ),
                record_id,
                certified,
            )
        )

    ranked.sort(
        reverse=True
    )
    _, _, record_id, certified = (
        ranked[0]
    )
    return (
        records[record_id],
        certified,
    )


def run(
    args: argparse.Namespace,
) -> None:
    from marl2d.gene_mrta_v16t.env import (
        EnvConfig,
    )

    config = EnvConfig()
    worlds, _stars, seeds = (
        load_scenario_bank(
            Path(
                args.scenario_bank
            ),
            config,
        )
    )
    records = _load_records(
        Path(
            args.v110_checkpoint
        )
    )
    source, certified = (
        _select_source_gene(
            records,
            args.archive_size_per_axis,
            args.hybrid_bank_limit,
            args.inheritance_threshold,
        )
    )
    gene = RouteTailDirectGene.from_v18(
        source.gene
    )

    print(
        f"SOURCE_GENE={source.record_id}"
    )
    print(
        "SOURCE_CERTIFIED="
        + ",".join(certified)
    )
    print(
        "PARAMETERS="
        f"{gene.parameter_count(gene.hidden_dim)}"
    )

    multi_worlds = 0
    total_assigned = 0
    for index in range(
        min(
            args.worlds,
            len(worlds),
        )
    ):
        rollout = (
            rollout_route_tail_gene(
                gene,
                worlds[index],
                config,
            )
        )
        route_lengths = [
            len(route)
            for route
            in rollout.plan.routes
        ]
        if max(
            route_lengths,
            default=0,
        ) >= 2:
            multi_worlds += 1
        total_assigned += int(
            rollout.evaluation.completed_tasks
        )

        print(
            f"WORLD {index:02d} "
            f"seed={seeds[index]} "
            f"routes={rollout.plan.routes} "
            f"lengths={route_lengths} "
            f"completed="
            f"{rollout.evaluation.completed_tasks:.0f} "
            f"T="
            f"{rollout.evaluation.time_optimality:.6f} "
            f"STOP="
            f"{rollout.plan.stopped_by_policy}"
        )

    print(
        "ROUTE_TAIL_SMOKE="
        + json.dumps(
            {
                "worlds": min(
                    args.worlds,
                    len(worlds),
                ),
                "worlds_with_multi_task_robot": (
                    multi_worlds
                ),
                "total_assigned_tasks": (
                    total_assigned
                ),
                "source_gene": (
                    source.record_id
                ),
                "source_certified": list(
                    certified
                ),
                "architecture": (
                    "task-only masking + "
                    "virtual route-tail update"
                ),
            },
            ensure_ascii=False,
        )
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scenario-bank",
        required=True,
    )
    p.add_argument(
        "--v110-checkpoint",
        required=True,
    )
    p.add_argument(
        "--worlds",
        type=int,
        default=5,
    )
    p.add_argument(
        "--archive-size-per-axis",
        type=int,
        default=16,
    )
    p.add_argument(
        "--hybrid-bank-limit",
        type=int,
        default=128,
    )
    p.add_argument(
        "--inheritance-threshold",
        type=float,
        default=0.95,
    )
    return p


def main() -> None:
    run(
        parser().parse_args()
    )


if __name__ == "__main__":
    main()
