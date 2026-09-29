import numpy as np

from marl2d.gene_mrta_v1.env import EnvConfig, evaluate_gene, generate_world, jain_fairness
from marl2d.gene_mrta_v1.gene import Gene


def test_jain_fairness_extremes():
    assert np.isclose(jain_fairness(np.array([5, 5, 5, 5])), 1.0)
    assert np.isclose(jain_fairness(np.array([20, 0, 0, 0])), 0.25)


def test_environment_is_deterministic():
    config = EnvConfig()
    world = generate_world(config, seed=123)
    gene = Gene(np.array([-1.0, -0.5, -0.2, 0.1]), 0.0)
    a = evaluate_gene(gene, world, config)
    b = evaluate_gene(gene, world, config)
    assert a == b


def test_axis_scores_are_bounded():
    config = EnvConfig()
    world = generate_world(config, seed=321)
    gene = Gene(np.zeros(4), 0.0)
    result = evaluate_gene(gene, world, config)
    assert 0.0 <= result.completion <= 1.0
    assert 0.0 <= result.efficiency <= 1.0
    assert 0.0 <= result.balance <= 1.0
