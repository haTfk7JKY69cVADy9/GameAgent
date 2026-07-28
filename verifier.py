from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Mapping

from beliefs import posterior_compromised
from model import Evidence, GameParams, Signal, VerifierAction
from utilities import verifier_expected_utility


@dataclass(frozen=True)
class Thresholds:
    tau_gh: float | None
    tau_hd: float | None
    tau_gd: float
    challenge_region_exists: bool


def compute_thresholds(params: GameParams) -> Thresholds:
    u = params.verifier_utility
    denom_gh = u.loss_compromised_grant + u.legitimate_challenge_loss - u.compromised_challenge_loss
    tau_gh = None if denom_gh <= 0 else (u.challenge_cost + u.legitimate_challenge_loss) / denom_gh
    denom_hd = u.benefit_legitimate + u.loss_legitimate_deny - u.legitimate_challenge_loss + u.compromised_challenge_loss
    tau_hd = None if denom_hd <= 0 else (
        u.benefit_legitimate + u.loss_legitimate_deny - u.challenge_cost - u.legitimate_challenge_loss
    ) / denom_hd
    tau_gd = (u.benefit_legitimate + u.loss_legitimate_deny) / (
        u.benefit_legitimate + u.loss_legitimate_deny + u.loss_compromised_grant
    )
    challenge_region_exists = tau_gh is not None and tau_hd is not None and 0.0 < tau_gh < tau_hd < 1.0
    return Thresholds(tau_gh, tau_hd, tau_gd, challenge_region_exists)


@dataclass(frozen=True)
class VerifierDecision:
    posterior_c: float
    best_actions: tuple[VerifierAction, ...]
    expected_utilities: Mapping[VerifierAction, float]


def best_responses(posterior_c: float, params: GameParams, atol: float = 1e-12) -> tuple[VerifierAction, ...]:
    utilities = {action: verifier_expected_utility(action, posterior_c, params) for action in VerifierAction}
    max_value = max(utilities.values())
    return tuple(action for action, value in utilities.items() if isclose(value, max_value, abs_tol=atol, rel_tol=0.0))


def verifier_policy(
    signal: Signal,
    evidence: Evidence,
    sigma_intact: Mapping[Signal, float],
    sigma_compromised: Mapping[Signal, float],
    params: GameParams,
    off_path_belief: float | None = None,
) -> VerifierDecision:
    mu = posterior_compromised(signal, evidence, sigma_intact, sigma_compromised, params, off_path_belief)
    utilities = {action: verifier_expected_utility(action, mu, params) for action in VerifierAction}
    return VerifierDecision(mu, best_responses(mu, params), utilities)
