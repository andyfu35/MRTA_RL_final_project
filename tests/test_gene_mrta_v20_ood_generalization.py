import numpy as np

from marl2d.gene_mrta_v20.ood_generalization import (
    DEFAULT_CELLS,
    _parse_cells,
    _regime,
    _unseen_seeds,
    _worlds_for_cell,
)
from marl2d.gene_mrta_v20.suite import FIXED_WORLD_SEEDS


def test_ood_seed_stream_is_deterministic_unique_and_unseen():
    a = _unseen_seeds(20070001, 100)
    b = _unseen_seeds(20070001, 100)
    assert a == b
    assert len(a) == 100
    assert len(set(a)) == 100
    assert not (set(a) & set(FIXED_WORLD_SEEDS))


def test_default_ood_grid_contains_id_boundary_and_size_extrapolation():
    cells = list(DEFAULT_CELLS)
    assert (10, 50) in cells
    assert (20, 100) in cells
    assert (25, 100) in cells
    assert (20, 125) in cells
    assert (40, 200) in cells
    assert (60, 300) in cells
    assert _regime(10, 50) == "in_distribution"
    assert _regime(20, 100) == "train_boundary"
    assert _regime(25, 100) == "robot_ood"
    assert _regime(20, 125) == "task_ood"
    assert _regime(40, 200) == "both_ood"


def test_parse_cells_deduplicates_and_preserves_order():
    assert _parse_cells("20x100,25x100,20x100") == [
        (20, 100, 100.0),
        (25, 100, 100.0),
    ]


def test_exact_size_ood_world_generation_keeps_deadlines_feasible():
    seeds = _unseen_seeds(20070001, 2)
    config, worlds = _worlds_for_cell(25, 125, seeds)
    assert config.robot_min == config.robot_max == 25
    assert config.task_min == config.task_max == 125
    for world in worlds:
        assert world.robot_count == 25
        assert world.task_count == 125
        assert np.all(
            world.task_deadlines > world.baseline_completion_times
        )



def test_parse_cells_accepts_density_controlled_world_size():
    cells = _parse_cells(
        "20x300@100,40x600@141.421356,60x900@173.205081"
    )
    assert cells[0] == (20, 300, 100.0)
    assert cells[1][:2] == (40, 600)
    assert np.isclose(cells[1][2], 141.421356)
    assert cells[2][:2] == (60, 900)
    assert np.isclose(cells[2][2], 173.205081)


def test_density_controlled_generation_uses_requested_world_size():
    seeds = _unseen_seeds(20110001, 1)
    config, worlds = _worlds_for_cell(
        40,
        200,
        seeds,
        world_size=141.421356,
    )
    world = worlds[0]
    assert np.isclose(config.world_size, 141.421356)
    assert world.robot_count == 40
    assert world.task_count == 200
    assert np.all(world.robot_positions >= 0.0)
    assert np.all(world.robot_positions <= config.world_size)
    assert np.all(world.task_positions >= 0.0)
    assert np.all(world.task_positions <= config.world_size)
