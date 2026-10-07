import numpy as np

from marl2d.gene_mrta_v20.public_benchmark import (
    MTSPLIB_REFERENCE,
    _best_feasible_reference,
    _parse_tsplib,
    _parse_twpc_csv,
    _parse_twpc_solution,
    _twpc_precedence_state,
)


def test_parse_tsplib_coordinates():
    text = """NAME : tiny
TYPE : TSP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 0 0
2 3 0
3 0 4
EOF
"""
    name, coords = _parse_tsplib(text)
    assert name == "tiny"
    assert coords.shape == (3, 2)
    assert np.allclose(coords[1], [3.0, 0.0])


def test_published_mtsplib_reference_marks_only_supported_true_optima():
    assert MTSPLIB_REFERENCE[("eil51", 2)]["cplex_proven_optimal"] is True
    assert MTSPLIB_REFERENCE[("eil76", 2)]["cplex_proven_optimal"] is True
    assert MTSPLIB_REFERENCE[("eil51", 5)]["cplex_proven_optimal"] is False
    assert np.isclose(MTSPLIB_REFERENCE[("eil51", 5)]["cplex"], 110.43)
    method, value = _best_feasible_reference(MTSPLIB_REFERENCE[("eil51", 5)])
    assert method == "schedulenet"
    assert np.isclose(value, 118.94)


def test_parse_twpc_csv_and_solution():
    # n = 1 robot + 2 tasks = 3 nodes
    text = """XCOORD.,YCOORD.,EST,TWL,DUR
0,0,0,100,0
1,0,0,20,2
2,0,5,20,3
0,0,0
0,0,1
0,0,0
0,1,2
1,0,1
2,1,0
"""
    info, precedence, distances = _parse_twpc_csv(text, 1, 2)
    assert info.shape == (3, 5)
    assert precedence.shape == (3, 3)
    assert distances.shape == (3, 3)
    assert precedence[1, 2] == 1
    assert np.isclose(distances[0, 2], 2.0)

    sol = _parse_twpc_solution(
        """# Solution for model tiny
# Finished Task: 2 / 2
# Makespan: 12.0, Total Distance: 3.0, Total Time: 0.1250
# R O S F
"""
    )
    assert sol["finished"] == 2
    assert sol["tasks"] == 2
    assert np.isclose(sol["makespan"], 12.0)
    assert np.isclose(sol["runtime_s"], 0.125)


def test_twpc_full_completion_is_trivial_primary_upper_bound():
    # Public TWPC primary objective is completed-task count, so N/N cannot be beaten.
    sol = _parse_twpc_solution(
        """# Solution for model full
# Finished Task: 18 / 18
# Makespan: 425.0, Total Distance: 347.0, Total Time: 15.5877
"""
    )
    assert sol["finished"] == sol["tasks"] == 18



def test_twpc_precedence_releases_successor_at_predecessor_finish():
    completed = np.asarray([True, False], dtype=bool)
    task_finish = np.asarray([12.5, np.nan], dtype=np.float64)
    predecessors = [[], [0]]
    available, release = _twpc_precedence_state(
        completed,
        task_finish,
        predecessors,
    )
    assert available.tolist() == [False, True]
    assert np.isclose(release[0], 0.0)
    assert np.isclose(release[1], 12.5)
