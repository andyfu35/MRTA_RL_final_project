import json

import numpy as np

from marl2d.gene_mrta_v119.bank import (
    BankRecord,
    rebuild_bank,
)
from marl2d.gene_mrta_v119.benchmark import (
    MANIFEST_VERSION,
    MTRPDInstance,
    default_split_for_replicate,
    load_manifest,
    pairwise_tsplib_euc_2d,
)
from marl2d.gene_mrta_v119.capabilities import (
    evaluate_gene,
    paired_delta,
)
from marl2d.gene_mrta_v119.rollout import (
    rollout_gene,
)


class _NearestGene:
    def action_logits(
        self,
        observations,
        eligible,
        row_open,
        col_open,
        step,
    ):
        logits = -np.asarray(
            observations[
                ...,
                1,
            ],
            dtype=np.float64,
        )
        logits = np.where(
            eligible,
            logits,
            -np.inf,
        )
        return (
            logits,
            -1e9,
        )


class _FarthestGene:
    def action_logits(
        self,
        observations,
        eligible,
        row_open,
        col_open,
        step,
    ):
        logits = np.asarray(
            observations[
                ...,
                1,
            ],
            dtype=np.float64,
        )
        logits = np.where(
            eligible,
            logits,
            -np.inf,
        )
        return (
            logits,
            -1e9,
        )


def _instance(
    *,
    route_limit=4.0,
    optimum=3.0,
):
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )
    return MTRPDInstance(
        instance_id="toy-30-0",
        family="toy",
        replicate_index=0,
        robot_count=1,
        route_limit=(
            route_limit
        ),
        optimum_total_latency=(
            optimum
        ),
        proven_optimal=True,
        split="evolution",
        coordinates=coords,
        distance_matrix=(
            pairwise_tsplib_euc_2d(
                coords
            )
        ),
    )


def test_v119_tsplib_euc_2d_rounding():
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.6, 0.0],
        ],
        dtype=np.float64,
    )
    matrix = (
        pairwise_tsplib_euc_2d(
            coords
        )
    )
    assert matrix[
        0,
        1,
    ] == 2.0


def test_v119_public_split_is_balanced_4_3_3():
    values = [
        default_split_for_replicate(
            index
        )
        for index in range(
            10
        )
    ]
    assert values.count(
        "evolution"
    ) == 4
    assert values.count(
        "validation"
    ) == 3
    assert values.count(
        "protected_test"
    ) == 3


def test_v119_rollout_matches_toy_optimum():
    result = rollout_gene(
        _NearestGene(),
        _instance(),
    )
    assert result.success
    assert result.completed_tasks == 2
    assert (
        result.total_latency
        == 3.0
    )
    assert (
        result.route_lengths
        == (4.0,)
    )


def test_v119_route_limit_reserves_return_to_depot():
    result = rollout_gene(
        _NearestGene(),
        _instance(
            route_limit=3.0,
        ),
    )
    assert not result.success
    assert (
        result.completion
        == 0.5
    )


def test_v119_optimum_retention_is_one_at_optimum():
    assessment = evaluate_gene(
        _NearestGene(),
        [
            _instance(),
        ],
    )
    assert assessment.success
    assert (
        assessment.overall_optimum_retention
        == 1.0
    )
    assert (
        assessment.exact_optimum_matches
        == 1
    )


def test_v119_paired_delta_detects_inheritance_regression():
    parent = evaluate_gene(
        _NearestGene(),
        [
            _instance(),
        ],
    )
    child = evaluate_gene(
        _FarthestGene(),
        [
            _instance(),
        ],
    )
    result = paired_delta(
        parent,
        child,
    )
    assert result[
        "overall_delta"
    ] < 0.0
    assert result[
        "losses"
    ] == 1


def test_v119_generic_pareto_bank_preserves_scale_tradeoff():
    axes = (
        "opt_retention_v30",
        "opt_retention_v40",
        "opt_retention_v50",
    )
    records = {
        "a": BankRecord(
            record_id="a",
            vector_data=(0.0,),
            hidden_dim=1,
            scores={
                "opt_retention_v30": 1.0,
                "opt_retention_v40": 0.8,
                "opt_retention_v50": 0.8,
            },
            overall_retention=0.866,
            worst_retention=0.8,
            generation=0,
            origin="test",
        ),
        "b": BankRecord(
            record_id="b",
            vector_data=(0.0,),
            hidden_dim=1,
            scores={
                "opt_retention_v30": 0.8,
                "opt_retention_v40": 1.0,
                "opt_retention_v50": 0.8,
            },
            overall_retention=0.866,
            worst_retention=0.8,
            generation=0,
            origin="test",
        ),
        "c": BankRecord(
            record_id="c",
            vector_data=(0.0,),
            hidden_dim=1,
            scores={
                "opt_retention_v30": 0.7,
                "opt_retention_v40": 0.7,
                "opt_retention_v50": 0.7,
            },
            overall_retention=0.7,
            worst_retention=0.7,
            generation=0,
            origin="test",
        ),
    }
    result = rebuild_bank(
        records,
        axes,
        epsilon=0.0,
        max_size=16,
    )
    assert set(
        result.records
    ) == {
        "a",
        "b",
    }


def test_v119_manifest_loader_supports_variable_robot_counts(
    tmp_path,
):
    manifest = {
        "version": (
            MANIFEST_VERSION
        ),
        "instances": [
            {
                "instance_id": "toy-a",
                "family": "toy",
                "replicate_index": 0,
                "robot_count": 2,
                "route_limit": 10.0,
                "optimum_total_latency": 3.0,
                "proven_optimal": True,
                "coordinates": [
                    [0, 0],
                    [1, 0],
                    [2, 0],
                ],
            },
            {
                "instance_id": "toy-b",
                "family": "toy",
                "replicate_index": 4,
                "robot_count": 5,
                "route_limit": 20.0,
                "optimum_total_latency": 3.0,
                "proven_optimal": True,
                "coordinates": [
                    [0, 0],
                    [1, 0],
                    [2, 0],
                ],
            },
        ],
    }
    path = (
        tmp_path
        / "manifest.json"
    )
    path.write_text(
        json.dumps(
            manifest
        ),
        encoding="utf-8",
    )
    rows = load_manifest(
        path
    )
    assert [
        row.robot_count
        for row in rows
    ] == [
        2,
        5,
    ]
    assert rows[
        0
    ].split == "evolution"
    assert rows[
        1
    ].split == "validation"
