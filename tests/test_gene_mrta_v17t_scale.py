import json
from pathlib import Path

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, generate_world
from marl2d.gene_mrta_v17.direct_gene import DirectAssignmentGene
from marl2d.gene_mrta_v17.direct_time_scale import (
    _load_checkpoint,
    _save_checkpoint,
    direct_time_scores,
)


def test_direct_time_scores_shape_and_bounds(tmp_path: Path):
    config = EnvConfig()
    worlds = [
        generate_world(config, 230_000_000),
        generate_world(config, 230_000_001),
    ]
    rng = np.random.default_rng(7)
    genes = [
        DirectAssignmentGene.random(rng, hidden_dim=4),
        DirectAssignmentGene.random(rng, hidden_dim=4),
    ]

    # A deliberately loose upper reference is enough to test normalization.
    stars = np.array([1.0, 1.0], dtype=np.float64)
    ratios, means = direct_time_scores(
        genes,
        worlds,
        stars,
        config,
    )

    assert ratios.shape == (2, 2)
    assert means.shape == (2,)
    assert np.all(ratios >= 0.0)
    assert np.all(ratios <= 1.0)
    assert np.allclose(means, ratios.mean(axis=1))


def test_checkpoint_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(11)
    genes = [
        DirectAssignmentGene.random(rng, hidden_dim=4)
        for _ in range(3)
    ]
    path = tmp_path / "checkpoint.json"

    _save_checkpoint(
        path,
        next_generation=25,
        population=genes,
        archive=genes[:2],
        hof=genes[1:],
        rng=rng,
    )

    restored_rng = np.random.default_rng(0)
    generation, population, archive, hof = _load_checkpoint(
        path,
        restored_rng,
    )

    assert generation == 25
    assert [g.key() for g in population] == [g.key() for g in genes]
    assert [g.key() for g in archive] == [g.key() for g in genes[:2]]
    assert [g.key() for g in hof] == [g.key() for g in genes[1:]]


def test_checkpoint_finalize_excludes_unevaluated_population(tmp_path: Path):
    from marl2d.gene_mrta_v17.checkpoint_finalize import (
        _genes_from_checkpoint,
    )

    rng = np.random.default_rng(19)
    hof_gene = DirectAssignmentGene.random(rng, hidden_dim=4)
    archive_gene = DirectAssignmentGene.random(rng, hidden_dim=4)
    population_gene = DirectAssignmentGene.random(rng, hidden_dim=4)

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "next_generation": 1000,
                "hof": [hof_gene.to_dict()],
                "archive": [archive_gene.to_dict()],
                "population": [population_gene.to_dict()],
                "rng_state": rng.bit_generator.state,
            }
        ),
        encoding="utf-8",
    )

    generation, candidates = _genes_from_checkpoint(checkpoint)

    assert generation == 1000
    assert {g.key() for g in candidates} == {
        hof_gene.key(),
        archive_gene.key(),
    }
    assert population_gene.key() not in {g.key() for g in candidates}



def test_failure_trace_matches_direct_rollout():
    from marl2d.gene_mrta_v17.failure_trace import _direct_trace
    from marl2d.gene_mrta_v17.rollout import rollout_direct_gene

    config = EnvConfig()
    world = generate_world(config, 230_000_010)
    rng = np.random.default_rng(23)
    gene = DirectAssignmentGene.random(rng, hidden_dim=4)

    traced = _direct_trace(gene, world, config)
    expected = rollout_direct_gene(gene, world, config).evaluation

    assert np.isclose(
        traced["evaluation"]["time_optimality"],
        expected.time_optimality,
    )
    assert np.isclose(
        traced["evaluation"]["completion"],
        expected.completion,
    )
    assert isinstance(traced["events"], list)
    assert all("assignments" in event for event in traced["events"])
