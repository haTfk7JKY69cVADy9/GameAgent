import math

import pytest

from mixed_equilibria import (
    INFO_SETS,
    MixedClassification,
    MixedStrategyProfile,
    _classification,
    _snap_to_boundary,
    evaluate_profile,
    find_mixed_equilibria,
)
from model import (
    AgentUtilityParams,
    GameParams,
    Signal,
    VerifierAction,
    VerifierUtilityParams,
)
from pure_equilibria import OffPathMode, find_pure_equilibria


def _deterministic_policy_from_pure(eq):
    return {
        info_set: {
            VerifierAction.GRANT: 1.0 if action == VerifierAction.GRANT else 0.0,
            VerifierAction.CHALLENGE: 1.0 if action == VerifierAction.CHALLENGE else 0.0,
            VerifierAction.DENY: 1.0 if action == VerifierAction.DENY else 0.0,
        }
        for info_set, action in eq.verifier_actions.items()
    }


_CONSTRUCTED_UTILITY = AgentUtilityParams(
    compromised_reinforced_cost=0.35,
    residual_gain_reinforced=0.05,
)


def test_known_pure_equilibrium_embedded_as_behavioral_profile_has_zero_regret():
    params = GameParams(agent_utility=_CONSTRUCTED_UTILITY)
    pure = find_pure_equilibria(params, off_path_mode=OffPathMode.CONSERVATIVE)
    desired = next(
        eq
        for eq in pure
        if eq.intact_signal == Signal.REINFORCED
        and eq.compromised_signal == Signal.BASIC
    )
    profile = MixedStrategyProfile(
        x=1.0,
        y=0.0,
        verifier_policy=_deterministic_policy_from_pure(desired),
    )
    report = evaluate_profile(profile, params, off_path_belief=1.0)
    assert report.max_regret <= 1e-10


def test_mixed_profile_regrets_are_finite_and_nonnegative():
    policy = {
        info_set: {
            VerifierAction.GRANT: 0.2,
            VerifierAction.CHALLENGE: 0.5,
            VerifierAction.DENY: 0.3,
        }
        for info_set in INFO_SETS
    }
    report = evaluate_profile(
        MixedStrategyProfile(x=0.4, y=0.6, verifier_policy=policy),
        GameParams(),
    )
    values = [
        report.intact_regret,
        report.compromised_regret,
        *report.verifier_regrets.values(),
    ]
    assert all(math.isfinite(value) and value >= 0.0 for value in values)
    assert report.max_regret == pytest.approx(max(values))


def test_numerical_search_returns_only_valid_approximate_equilibria():
    params = GameParams(agent_utility=_CONSTRUCTED_UTILITY)
    equilibria = find_mixed_equilibria(
        params,
        tolerance=1e-4,
        n_random_starts=4,
        seed=11,
        use_global_search=False,
        maxiter=400,
    )
    assert equilibria
    for eq in equilibria:
        assert eq.report.max_regret <= 1e-4
        assert 0.0 <= eq.profile.x <= 1.0
        assert 0.0 <= eq.profile.y <= 1.0
        assert eq.classification in set(MixedClassification)
        for probabilities in eq.profile.verifier_policy.values():
            assert sum(probabilities.values()) == pytest.approx(1.0)
            assert all(0.0 <= value <= 1.0 for value in probabilities.values())


def test_base_case_has_one_nonduplicated_pooling_equilibrium_near_zero():
    eqs = find_mixed_equilibria(
        GameParams(),
        n_random_starts=16,
        maxiter=500,
    )
    pooling = [
        eq for eq in eqs
        if abs(eq.profile.x) <= 1e-6 and abs(eq.profile.y) <= 1e-6
    ]
    assert len(pooling) == 1
    assert pooling[0].report.max_regret == pytest.approx(0.0, abs=1e-9)


def test_constructed_case_has_no_phantom_boundary_duplicates():
    params = GameParams(agent_utility=_CONSTRUCTED_UTILITY)
    eqs = find_mixed_equilibria(
        params,
        n_random_starts=16,
    )
    assert len(eqs) == 3


def test_constructed_case_contains_desired_separating_equilibrium():
    params = GameParams(agent_utility=_CONSTRUCTED_UTILITY)
    eqs = find_mixed_equilibria(
        params,
        n_random_starts=16,
    )
    desired = [
        eq for eq in eqs
        if eq.profile.x == pytest.approx(1.0, abs=1e-6)
        and eq.profile.y == pytest.approx(0.0, abs=1e-6)
    ]
    assert len(desired) == 1
    assert desired[0].report.max_regret == pytest.approx(0.0, abs=1e-9)


def test_constructed_case_contains_semi_mixed_equilibrium():
    params = GameParams(agent_utility=_CONSTRUCTED_UTILITY)
    eqs = find_mixed_equilibria(
        params,
        n_random_starts=16,
    )
    semi = [eq for eq in eqs if eq.classification == MixedClassification.SEMI_MIXED]
    assert len(semi) == 1
    assert semi[0].profile.x == pytest.approx(0.1682, abs=1e-3)
    assert semi[0].profile.y == pytest.approx(0.0, abs=1e-6)


def test_edge_case_without_challenge_region_does_not_crash():
    vu = VerifierUtilityParams(compromised_challenge_loss=5.0)
    eqs = find_mixed_equilibria(
        GameParams(verifier_utility=vu),
        n_random_starts=16,
        maxiter=500,
    )
    assert len(eqs) == 1
    assert eqs[0].profile.x == pytest.approx(0.0, abs=1e-6)
    assert eqs[0].profile.y == pytest.approx(0.0, abs=1e-6)


def test_snap_and_classification_tolerances_have_distinct_semantics():
    value = 5e-5
    assert _snap_to_boundary(value, 1e-5) == pytest.approx(value)
    assert _classification(value, 0.0, 1e-4) == MixedClassification.BOUNDARY


def test_snap_removes_solver_scale_residue():
    assert _snap_to_boundary(2e-6, 1e-5) == 0.0
    assert _snap_to_boundary(1.0 - 2e-6, 1e-5) == 1.0


def test_invalid_tolerance_relationship_is_rejected():
    with pytest.raises(ValueError, match="snap_tolerance"):
        find_mixed_equilibria(
            GameParams(),
            n_random_starts=0,
            use_global_search=False,
            snap_tolerance=1e-3,
            boundary_tolerance=1e-4,
        )
