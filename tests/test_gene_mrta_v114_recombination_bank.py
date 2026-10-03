import numpy as np

from marl2d.gene_mrta_v18.direct_gene import (
    ConsequenceAwareDirectGene,
)
from marl2d.gene_mrta_v114.recombination_gene import (
    COEFFICIENT_NAMES,
    RECOMBINATION_AXES,
    RecombinationGene,
    RecombinationRecord,
    initial_recombination_bank,
    pareto_front_ids,
    prune_recombination_bank,
    recombine_with_gene,
    sample_recombination_gene_id,
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


def test_recombination_gene_has_seven_coefficients_and_gates():
    gene = _gene(
        np.arange(
            7,
            dtype=np.float64,
        ),
        [
            True,
            False,
            True,
            False,
            True,
            False,
            True,
        ],
    )

    assert len(
        COEFFICIENT_NAMES
    ) == 7
    assert (
        gene.coefficients.shape
        == (7,)
    )
    assert (
        gene.gates.shape
        == (7,)
    )
    assert (
        gene.active_term_count
        == 4
    )
    assert np.array_equal(
        gene.masked_coefficients(),
        np.asarray(
            [
                0.0,
                0.0,
                2.0,
                0.0,
                4.0,
                0.0,
                6.0,
            ]
        ),
    )


def test_all_gates_off_is_exact_center_law():
    gene = _gene(
        np.ones(7),
        np.zeros(
            7,
            dtype=bool,
        ),
    )
    law = gene.to_law()

    assert law.beta_norm_ratio == 0.0
    assert law.beta_quality_diff == 0.0
    assert law.beta_capability_diff == 0.0
    assert law.gamma_bias == 0.0
    assert law.gamma_cosine == 0.0
    assert law.gamma_sign_agreement == 0.0
    assert law.gamma_magnitude == 0.0


def test_rule_child_remains_parent_swap_symmetric():
    rng = np.random.default_rng(
        17
    )
    anchor = (
        ConsequenceAwareDirectGene.random(
            rng
        )
    )
    parent_a = anchor.mutated(
        rng,
        sigma=0.2,
        mutation_rate=0.8,
    )
    parent_b = anchor.mutated(
        rng,
        sigma=0.18,
        mutation_rate=0.7,
    )
    rule = _gene(
        [
            1.2,
            -0.7,
            0.5,
            -0.2,
            1.1,
            0.8,
            -0.4,
        ],
        [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    )

    child_ab, _ = (
        recombine_with_gene(
            rule,
            parent_a,
            parent_b,
            anchor,
            quality_a=0.98,
            quality_b=0.95,
            capability_count_a=3,
            capability_count_b=2,
        )
    )
    child_ba, _ = (
        recombine_with_gene(
            rule,
            parent_b,
            parent_a,
            anchor,
            quality_a=0.95,
            quality_b=0.98,
            capability_count_a=2,
            capability_count_b=3,
        )
    )

    assert np.allclose(
        child_ab.vector_data,
        child_ba.vector_data,
        atol=1e-12,
        rtol=1e-12,
    )


def test_mutation_is_deterministic_for_seed_and_self_adapts_sigma():
    parent = _gene(
        np.zeros(7),
        np.ones(
            7,
            dtype=bool,
        ),
        sigma=0.3,
    )
    a = parent.mutate(
        np.random.default_rng(9),
        generation=1,
        gate_flip_rate=0.2,
        sigma_tau=0.15,
        sigma_min=0.03,
        sigma_max=0.8,
    )
    b = parent.mutate(
        np.random.default_rng(9),
        generation=1,
        gate_flip_rate=0.2,
        sigma_tau=0.15,
        sigma_min=0.03,
        sigma_max=0.8,
    )

    assert np.array_equal(
        a.coefficients,
        b.coefficients,
    )
    assert np.array_equal(
        a.gates,
        b.gates,
    )
    assert (
        a.mutation_sigma
        == b.mutation_sigma
    )
    assert 0.03 <= (
        a.mutation_sigma
    ) <= 0.8
    assert a.parents == (
        parent.gene_id,
    )


def test_newborn_rule_priors_are_conservative():
    record = RecombinationRecord(
        gene=_gene(
            np.zeros(7),
            np.zeros(
                7,
                dtype=bool,
            ),
        )
    )
    scores = record.axis_scores()

    assert np.isclose(
        scores[
            "screen_yield"
        ],
        0.125,
    )
    assert np.isclose(
        scores[
            "acceptance_yield"
        ],
        0.10,
    )
    assert np.isclose(
        scores[
            "four_capability_yield"
        ],
        0.05,
    )
    assert np.isclose(
        scores[
            "retention_quality"
        ],
        0.95,
    )


def test_initial_bank_contains_center_and_is_deterministic():
    first = initial_recombination_bank(
        count=8,
        seed=3,
        initial_sigma=0.35,
    )
    second = initial_recombination_bank(
        count=8,
        seed=3,
        initial_sigma=0.35,
    )

    assert list(
        first
    ) == list(
        second
    )
    assert len(first) == 8
    assert any(
        record.gene.active_term_count
        == 0
        for record
        in first.values()
    )


def test_axis_sampling_is_deterministic_for_seed():
    bank = initial_recombination_bank(
        count=6,
        seed=4,
        initial_sigma=0.35,
    )
    first = (
        sample_recombination_gene_id(
            bank,
            np.random.default_rng(
                11
            ),
            selection_power=2.0,
            uniform_fraction=0.25,
        )
    )
    second = (
        sample_recombination_gene_id(
            bank,
            np.random.default_rng(
                11
            ),
            selection_power=2.0,
            uniform_fraction=0.25,
        )
    )
    assert first == second
    assert first[1] in (
        RECOMBINATION_AXES
    )


def test_pareto_front_keeps_conflicting_rule_specialists():
    a = RecombinationRecord(
        gene=_gene(
            np.zeros(7),
            np.zeros(
                7,
                dtype=bool,
            )
        ),
        generated=10,
        screen_selected=9,
        accepted=1,
        four_capability_accepted=0,
        screen_retention_sum=9.5,
    )
    b = RecombinationRecord(
        gene=_gene(
            np.ones(7),
            np.ones(
                7,
                dtype=bool,
            )
        ),
        generated=10,
        screen_selected=2,
        accepted=8,
        four_capability_accepted=7,
        screen_retention_sum=9.0,
    )
    records = {
        a.gene_id: a,
        b.gene_id: b,
    }

    front = pareto_front_ids(
        records
    )

    assert set(front) == set(
        records
    )


def test_pruning_respects_bank_capacity():
    bank = initial_recombination_bank(
        count=20,
        seed=14,
        initial_sigma=0.35,
    )
    for idx, record in enumerate(
        bank.values()
    ):
        record.generated = 10 + idx
        record.screen_selected = (
            idx % 8
        )
        record.accepted = (
            idx % 5
        )
        record.four_capability_accepted = (
            idx % 3
        )
        record.screen_retention_sum = (
            0.9
            * record.generated
        )

    pruned = prune_recombination_bank(
        bank,
        specialist_size=3,
        pareto_limit=5,
        total_limit=10,
    )

    assert len(pruned) == 10
