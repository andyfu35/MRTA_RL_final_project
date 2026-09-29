import numpy as np

from marl2d.gene_mrta_v11.env import EnvConfig, evaluate_gene, generate_world, jain_fairness
from marl2d.gene_mrta_v11.gene import Gene


def test_jain_fairness_extremes():
    assert np.isclose(jain_fairness(np.array([5, 5, 5, 5])), 1.0)
    assert np.isclose(jain_fairness(np.array([20, 0, 0, 0])), 0.25)


def test_gene_has_only_four_effective_parameters():
    gene = Gene(np.array([-1.0, 0.2, -0.5, 0.1]))
    assert gene.vector().shape == (4,)
    assert "bias" not in gene.to_dict()


def test_environment_is_deterministic():
    config = EnvConfig()
    world = generate_world(config, seed=123)
    gene = Gene(np.array([-1.0, -0.5, -0.2, 0.1]))
    a = evaluate_gene(gene, world, config)
    b = evaluate_gene(gene, world, config)
    assert a == b


def test_axis_scores_are_bounded_and_efficiency_cannot_exceed_completion():
    config = EnvConfig()
    world = generate_world(config, seed=321)
    gene = Gene(np.zeros(4))
    result = evaluate_gene(gene, world, config)
    assert 0.0 <= result.completion <= 1.0
    assert 0.0 <= result.efficiency <= 1.0
    assert 0.0 <= result.balance <= 1.0
    assert 0.0 <= result.route_efficiency <= 1.0
    assert result.efficiency <= result.completion + 1e-12


def test_reported_mean_loads_match_mean_completed_tasks():
    config = EnvConfig()
    world = generate_world(config, seed=456)
    gene = Gene(np.array([-1.0, 0.0, 0.0, 0.0]))
    result = evaluate_gene(gene, world, config)
    assert np.isclose(sum(result.robot_task_counts), result.completed_tasks)
