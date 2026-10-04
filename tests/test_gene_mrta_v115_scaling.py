import json

import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    generate_world,
)
from marl2d.gene_mrta_v18.direct_gene import (
    ConsequenceAwareDirectGene,
)
from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
)
from marl2d.gene_mrta_v113.evolve import (
    AXES,
    GeneRecord,
)
from marl2d.gene_mrta_v113.robust_metrics import (
    rollout_route_tail_robust_metrics,
)
from marl2d.gene_mrta_v113.route_tail import (
    plan_route_tails,
)
from marl2d.gene_mrta_v115.scaling import (
    _load_frozen_gene,
    _pair_slot_count,
    _parse_cases,
    _robust_from_plan,
    scale_config,
)


def test_scale_config_preserves_area_density():
    c4 = scale_config(
        4,
        20,
    )
    c16 = scale_config(
        16,
        80,
    )
    c64 = scale_config(
        64,
        320,
    )

    assert c4.world_size == 100.0
    assert c4.obstacle_count == 10
    assert c16.world_size == 200.0
    assert c16.obstacle_count == 40
    assert c64.world_size == 400.0
    assert c64.obstacle_count == 160


def test_parse_cases_and_pair_slots():
    assert _parse_cases(
        "4x20,8x40"
    ) == [
        (4, 20),
        (8, 40),
    ]
    assert _pair_slot_count(
        4,
        20,
        0,
    ) == 0
    assert _pair_slot_count(
        4,
        20,
        2,
    ) == (
        4
        * (
            20
            + 19
        )
    )


def test_robust_metrics_reuse_existing_plan():
    config = EnvConfig(
        world_size=40.0,
        num_robots=2,
        num_tasks=4,
        obstacle_count=0,
    )
    world = generate_world(
        config,
        seed=123,
    )
    base = (
        ConsequenceAwareDirectGene.random(
            np.random.default_rng(
                9
            )
        )
    )
    gene = RouteTailDirectGene.from_v18(
        base
    )
    plan = plan_route_tails(
        gene,
        world,
        config,
    )
    continuation, reserve = (
        _robust_from_plan(
            plan,
            world,
            config,
        )
    )
    reference = (
        rollout_route_tail_robust_metrics(
            gene,
            world,
            config,
        )
    )

    assert np.isclose(
        continuation,
        reference.continuation_preservation,
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.isclose(
        reserve,
        reference.fleet_option_reserve,
        atol=1e-12,
        rtol=1e-12,
    )


def test_frozen_gene_selection_uses_max_min_retention(tmp_path):
    rng = np.random.default_rng(
        5
    )
    scores = [
        {
            axis: value
            for axis
            in AXES
        }
        for value in (
            0.96,
            0.98,
            1.0,
        )
    ]
    records = []
    for idx, item in enumerate(
        scores
    ):
        record = GeneRecord(
            record_id=f"g{idx}",
            gene=(
                ConsequenceAwareDirectGene.random(
                    rng
                )
            ),
            capabilities=tuple(
                AXES
            ),
            scores=item,
            origin="mating",
            generation=10,
        )
        records.append(
            record.to_dict()
        )

    checkpoint = (
        tmp_path
        / "checkpoint.json"
    )
    checkpoint.write_text(
        json.dumps(
            {
                "records": records
            }
        ),
        encoding="utf-8",
    )

    gene_id, gene, selected, retention = (
        _load_frozen_gene(
            checkpoint,
            archive_size=16,
            hybrid_limit=128,
            threshold=0.95,
        )
    )

    assert gene_id == "g2"
    assert gene.vector_data.size == 148
    assert all(
        np.isclose(
            retention[axis],
            1.0,
        )
        for axis in AXES
    )
    assert all(
        np.isclose(
            selected[axis],
            1.0,
        )
        for axis in AXES
    )
