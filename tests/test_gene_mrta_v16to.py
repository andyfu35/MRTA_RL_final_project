import json

import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    evaluate_gene,
    evaluate_genes_time_optimality_matrix,
    generate_world,
)
from marl2d.gene_mrta_v16t.gene import Gene
from marl2d.gene_mrta_v16t.oracle_train import (
    _oracle_scores,
    _parse_schedule,
    _schedule_value,
)


def test_time_optimality_matrix_matches_scalar_evaluation():
    config = EnvConfig()
    worlds = [
        generate_world(config, 140_000_000),
        generate_world(config, 140_000_001),
    ]
    genes = [
        Gene(np.array([-1.0, -1.2, -0.8, 0.1, 0.1, 0.0, -0.3, -0.2])),
        Gene(np.array([-0.8, -1.0, -1.1, 0.0, 0.2, 0.1, -0.6, -0.1])),
    ]

    matrix = evaluate_genes_time_optimality_matrix(
        genes,
        worlds,
        config,
    )

    assert matrix.shape == (2, 2)
    for i, gene in enumerate(genes):
        for j, world in enumerate(worlds):
            scalar = evaluate_gene(gene, world, config).time_optimality
            assert np.isclose(matrix[i, j], scalar, atol=1e-12)


def test_oracle_scores_are_mean_per_world_retention():
    config = EnvConfig()
    worlds = [
        generate_world(config, 141_000_000),
        generate_world(config, 141_000_001),
    ]
    genes = [
        Gene(np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -0.2, -0.1])),
        Gene(np.array([-0.5, -1.5, -0.7, 0.2, 0.0, -0.1, -0.4, -0.2])),
    ]
    matrix = evaluate_genes_time_optimality_matrix(
        genes,
        worlds,
        config,
    )
    stars = np.array([0.25, 0.20], dtype=np.float64)

    scores = _oracle_scores(genes, worlds, stars, config)

    expected = np.mean(matrix / stars[None, :], axis=1)
    assert np.allclose(scores, expected, atol=1e-12)


def test_progressive_schedule():
    schedule = _parse_schedule("0:16,200:32,500:64,1000:128")
    assert _schedule_value(schedule, 0) == 16
    assert _schedule_value(schedule, 199) == 16
    assert _schedule_value(schedule, 200) == 32
    assert _schedule_value(schedule, 999) == 64
    assert _schedule_value(schedule, 1000) == 128
    assert _schedule_value(schedule, 5000) == 128
