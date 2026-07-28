import pytest
from model import GameParams
from verifier import compute_thresholds


def test_base_thresholds():
    t = compute_thresholds(GameParams())
    assert t.tau_gh == pytest.approx(0.35 / 3.65)
    assert t.tau_hd == pytest.approx(1.65 / 2.35)
    assert t.tau_gd == pytest.approx(2.0 / 6.0)
    assert t.challenge_region_exists


def test_threshold_ordering():
    t = compute_thresholds(GameParams())
    assert 0 < t.tau_gh < t.tau_hd < 1


from model import VerifierUtilityParams


def test_no_challenge_region_when_gh_denominator_is_nonpositive():
    params = GameParams(
        verifier_utility=VerifierUtilityParams(
            benefit_legitimate=1.0,
            loss_compromised_grant=4.0,
            loss_legitimate_deny=1.0,
            challenge_cost=0.20,
            legitimate_challenge_loss=0.15,
            compromised_challenge_loss=5.0,
        )
    )

    thresholds = compute_thresholds(params)

    assert thresholds.tau_gh is None
    assert thresholds.tau_hd == pytest.approx(1.65 / 6.85)
    assert thresholds.tau_gd == pytest.approx(2.0 / 6.0)
    assert not thresholds.challenge_region_exists
