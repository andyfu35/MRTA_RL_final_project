import numpy as np

from marl2d.gene_mrta_v120.benchmark import (
    CERTIFICATE_ZIP,
    EXACT_OPTIMUM_IDS,
    INSTANCE_ZIP,
    PAPER_ONLY_REFERENCES,
    describe_benchmark,
    edge_distance,
    load_benchmark,
    load_certificates,
)
from marl2d.gene_mrta_v120.gene import (
    ScalableRouteTailGene,
)
from marl2d.gene_mrta_v120.rollout import (
    rollout_gene,
)


def _by_id():
    return {
        item.instance_id: item
        for item in load_benchmark()
    }


def _certificate_objective(instance, certificate):
    values = []
    for route in certificate.routes:
        total = 0.0
        for a, b in zip(route, route[1:]):
            total += float(
                edge_distance(
                    instance,
                    instance.coordinates[a],
                    instance.coordinates[b],
                )
            )
        values.append(total)
    return max(values)


def test_v120_mirrored_public_assets_exist():
    assert INSTANCE_ZIP.is_file()
    assert CERTIFICATE_ZIP.is_file()


def test_v120_public_benchmark_structure():
    rows = load_benchmark()
    summary = describe_benchmark(rows)

    assert len(rows) == 77
    assert summary["instance_count"] == 77
    assert summary["by_paper_set"] == {
        "S": 41,
        "L": 36,
    }
    assert summary["known_exact_optimum_count"] == 22
    assert set(summary["by_robot_count"]) == {
        "3",
        "5",
        "10",
        "20",
        "30",
    }
    assert summary["min_vertices"] == 51
    assert summary["max_vertices"] == 5915


def test_v120_exact_optimum_catalog_has_22_instances():
    assert len(EXACT_OPTIMUM_IDS) == 22


def test_v120_certificate_archive_has_72_solutions():
    certificates = load_certificates()
    assert len(certificates) == 72


def test_v120_five_paper_only_references_are_present():
    rows = _by_id()
    assert set(PAPER_ONLY_REFERENCES) == {
        "mtsp51_3",
        "mtsp51_5",
        "mtsp51_10",
        "mtsp150_30",
        "gtsp150_30",
    }
    for instance_id, expected in PAPER_ONLY_REFERENCES.items():
        assert np.isclose(
            rows[instance_id].reference_value,
            expected,
        )


def test_v120_malformed_legacy_headers_are_parsed_from_filename():
    rows = _by_id()
    assert rows["mtsp51_5"].robot_count == 5
    assert rows["mtsp51_10"].robot_count == 10
    assert rows["mtsp150_30"].robot_count == 30


def test_v120_certificate_reconstructs_kroa200_3_objective():
    rows = _by_id()
    certificates = load_certificates()
    instance = rows["kroa200_3"]
    cert = certificates["kroa200_3"]
    value = _certificate_objective(instance, cert)
    assert np.isclose(
        value,
        cert.objective,
        rtol=0.0,
        atol=0.02,
    )


def test_v120_certificate_reconstructs_mtsp100_3_objective():
    rows = _by_id()
    certificates = load_certificates()
    instance = rows["mtsp100_3"]
    cert = certificates["mtsp100_3"]
    value = _certificate_objective(instance, cert)
    assert np.isclose(
        value,
        cert.objective,
        rtol=0.0,
        atol=0.02,
    )


def test_v120_one_gene_handles_variable_robot_and_task_counts():
    rows = _by_id()
    rng = np.random.default_rng(120)
    base = ScalableRouteTailGene.random(
        rng,
        hidden_dim=8,
        scale=0.1,
    )
    gene = ScalableRouteTailGene.from_gene(base)

    a = rollout_gene(
        gene,
        rows["mtsp51_3"],
        candidate_k=16,
    )
    b = rollout_gene(
        gene,
        rows["mtsp150_10"],
        candidate_k=16,
    )

    assert a.success
    assert b.success
    assert len(a.routes) == 3
    assert len(b.routes) == 10
    assert a.task_count != b.task_count
    assert all(len(route) >= 1 for route in a.routes)
    assert all(len(route) >= 1 for route in b.routes)
