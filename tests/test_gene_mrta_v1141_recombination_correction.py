import json

import numpy as np

from marl2d.gene_mrta_v1141.evolve import (
    _rule_specialists,
)
from marl2d.gene_mrta_v1141.recombination_gene import (
    RecombinationGene,
    RecombinationRecord,
    initial_recombination_bank,
    pareto_front_ids,
    prune_recombination_bank,
    spawn_recombination_mutants,
)


def _gene(
    coefficients,
    gates,
    sigma=0.2,
    generation=0,
):
    return RecombinationGene(
        coefficients=np.asarray(
            coefficients,
            dtype=np.float64,
        ),
        gates=np.asarray(
            gates,
            dtype=bool,
        ),
        mutation_sigma=sigma,
        generation=generation,
    )


def test_disabled_terms_are_canonicalized_to_zero():
    gene = _gene(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
        ],
        [
            True,
            False,
            True,
            False,
            False,
            True,
            False,
        ],
    )
    assert np.array_equal(
        gene.coefficients,
        np.asarray(
            [
                1.0,
                0.0,
                3.0,
                0.0,
                0.0,
                6.0,
                0.0,
            ]
        ),
    )


def test_identical_formula_has_identical_phenotype_id_even_with_different_sigma():
    a = _gene(
        np.arange(
            7,
            dtype=np.float64,
        ),
        np.zeros(
            7,
            dtype=bool,
        ),
        sigma=0.1,
    )
    b = _gene(
        -np.arange(
            7,
            dtype=np.float64,
        ),
        np.zeros(
            7,
            dtype=bool,
        ),
        sigma=0.7,
    )

    assert (
        a.phenotype_id
        == b.phenotype_id
    )
    assert (
        a.gene_id
        == b.gene_id
    )
    assert (
        a.genotype_fingerprint
        != b.genotype_fingerprint
    )


def test_capability_difference_equation_reports_normalization():
    gene = _gene(
        np.ones(7),
        np.ones(
            7,
            dtype=bool,
        ),
    )
    assert "(H_A-H_B)/4" in (
        gene.equation()[
            "alpha"
        ]
    )


def test_mature_rule_lcb_beats_lucky_two_trial_rule():
    mature = RecombinationRecord(
        gene=_gene(
            np.ones(7),
            np.ones(
                7,
                dtype=bool,
            ),
        ),
        generated=100,
        screen_selected=50,
        accepted=40,
        four_capability_accepted=30,
        screen_retention_sum=96.0,
    )
    lucky = RecombinationRecord(
        gene=_gene(
            np.ones(7) * 2.0,
            np.ones(
                7,
                dtype=bool,
            ),
        ),
        generated=2,
        screen_selected=2,
        accepted=2,
        four_capability_accepted=2,
        screen_retention_sum=2.0,
    )

    assert (
        mature.evidence_score(
            "acceptance_yield",
            quantile=0.10,
        )
        > lucky.evidence_score(
            "acceptance_yield",
            quantile=0.10,
        )
    )


def test_specialist_requires_mature_evidence_when_available():
    mature = RecombinationRecord(
        gene=_gene(
            np.ones(7),
            np.ones(
                7,
                dtype=bool,
            ),
        ),
        generated=40,
        screen_selected=15,
        accepted=12,
        four_capability_accepted=10,
        screen_retention_sum=38.0,
    )
    newborn = RecombinationRecord(
        gene=_gene(
            np.ones(7) * 2.0,
            np.ones(
                7,
                dtype=bool,
            ),
        ),
        generated=2,
        screen_selected=2,
        accepted=2,
        four_capability_accepted=2,
        screen_retention_sum=2.0,
    )
    records = {
        mature.gene_id: mature,
        newborn.gene_id: newborn,
    }

    specialists, are_mature = (
        _rule_specialists(
            records,
            min_evidence=32,
            evidence_quantile=0.10,
        )
    )

    assert are_mature
    assert set(
        specialists.values()
    ) == {
        mature.gene_id
    }


def test_center_phenotype_cannot_occupy_multiple_bank_slots():
    center_a = _gene(
        np.zeros(7),
        np.zeros(
            7,
            dtype=bool,
        ),
        sigma=0.1,
    )
    center_b = _gene(
        np.ones(7) * 3.0,
        np.zeros(
            7,
            dtype=bool,
        ),
        sigma=0.7,
    )
    bank = {}
    bank[
        center_a.gene_id
    ] = RecombinationRecord(
        gene=center_a
    )
    bank[
        center_b.gene_id
    ] = RecombinationRecord(
        gene=center_b
    )

    assert len(bank) == 1


def test_spawned_mutants_have_unique_phenotypes():
    bank = initial_recombination_bank(
        count=8,
        seed=5,
        initial_sigma=0.35,
    )
    children = (
        spawn_recombination_mutants(
            bank,
            np.random.default_rng(
                33
            ),
            generation=1,
            count=6,
            selection_power=2.0,
            uniform_fraction=0.25,
            gate_flip_rate=0.25,
            sigma_tau=0.15,
            sigma_min=0.03,
            sigma_max=0.80,
            evidence_quantile=0.10,
        )
    )
    ids = [
        child.gene_id
        for child in children
    ]
    assert len(ids) == len(
        set(ids)
    )
    assert not (
        set(ids)
        & set(bank)
    )


def test_pareto_front_excludes_immature_rules_when_threshold_is_set():
    mature = RecombinationRecord(
        gene=_gene(
            np.ones(7),
            np.ones(
                7,
                dtype=bool,
            ),
        ),
        generated=40,
        screen_selected=15,
        accepted=10,
        four_capability_accepted=8,
        screen_retention_sum=38.0,
    )
    newborn = RecombinationRecord(
        gene=_gene(
            np.ones(7) * 2.0,
            np.ones(
                7,
                dtype=bool,
            ),
        ),
        generated=2,
        screen_selected=2,
        accepted=2,
        four_capability_accepted=2,
        screen_retention_sum=2.0,
    )
    records = {
        mature.gene_id: mature,
        newborn.gene_id: newborn,
    }

    front = pareto_front_ids(
        records,
        min_evidence=32,
        evidence_quantile=0.10,
    )

    assert front == [
        mature.gene_id
    ]


def test_pruning_keeps_capacity_and_exploration_slots():
    bank = initial_recombination_bank(
        count=20,
        seed=14,
        initial_sigma=0.35,
    )
    for idx, record in enumerate(
        bank.values()
    ):
        record.generated = (
            40
            if idx < 8
            else idx % 4
        )
        record.screen_selected = (
            min(
                record.generated,
                8 + idx % 5,
            )
        )
        record.accepted = min(
            record.generated,
            4 + idx % 4,
        )
        record.four_capability_accepted = min(
            record.generated,
            2 + idx % 3,
        )
        record.screen_retention_sum = (
            0.9
            * record.generated
        )

    pruned = prune_recombination_bank(
        bank,
        specialist_size=2,
        pareto_limit=4,
        total_limit=10,
        min_evidence=32,
        evidence_quantile=0.10,
        exploration_slots=3,
    )

    assert len(pruned) == 10
    assert any(
        record.generated < 32
        for record
        in pruned.values()
    )
