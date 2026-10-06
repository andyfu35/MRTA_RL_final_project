import numpy as np
import torch

from marl2d.gene_mrta_v120.benchmark import load_benchmark
from marl2d.gene_mrta_v120.gene import ScalableRouteTailGene
from marl2d.gene_mrta_v120.rollout import rollout_gene
from marl2d.gene_mrta_v122.mps_batch import (
    _distance_matrix,
    _neighbor_order,
    prepare_instance,
    rollout_gene_batch_mps,
)


def _rows():
    return {
        item.instance_id: item
        for item in load_benchmark()
    }


def test_v122_precomputed_distance_matrix_matches_public_edge_distance():
    instance = _rows()["mtsp100_10"]
    matrix = _distance_matrix(instance)
    assert matrix.shape == (
        instance.vertex_count,
        instance.vertex_count,
    )
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)


def test_v122_neighbor_order_contains_every_task_exactly_once():
    instance = _rows()["mtsp51_3"]
    order = _neighbor_order(instance)
    expected = set(range(1, instance.vertex_count))
    assert order.shape == (
        instance.vertex_count,
        instance.task_count,
    )
    for row in order:
        assert set(row.tolist()) == expected


def test_v122_batch_backend_matches_serial_on_cpu_for_small_case():
    instance = _rows()["mtsp51_3"]
    rng = np.random.default_rng(122)
    genes = [
        ScalableRouteTailGene.from_gene(
            ScalableRouteTailGene.random(
                rng,
                hidden_dim=8,
                scale=0.35,
            )
        )
        for _ in range(2)
    ]

    serial = [
        rollout_gene(
            gene,
            instance,
            candidate_k=16,
        )
        for gene in genes
    ]
    cache = prepare_instance(
        instance,
        device=torch.device("cpu"),
    )
    batch = rollout_gene_batch_mps(
        genes,
        cache,
        candidate_k=16,
    )

    for index, expected in enumerate(serial):
        assert batch.routes_for_world(
            index,
            instance.robot_count,
        ) == expected.routes
        assert np.isclose(
            batch.objectives[index],
            expected.objective,
            rtol=0.0,
            atol=1e-8,
        )
