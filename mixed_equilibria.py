from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from beliefs import evidence_likelihood
from model import AgentType, Evidence, GameParams, Signal, VerifierAction
from utilities import agent_utility, verifier_expected_utility


INFO_SETS: tuple[tuple[Signal, Evidence], ...] = tuple(
    (signal, evidence) for signal in Signal for evidence in Evidence
)
ACTIONS: tuple[VerifierAction, ...] = tuple(VerifierAction)


class MixedClassification(str, Enum):
    FULLY_MIXED = "fully_mixed"
    SEMI_MIXED = "semi_mixed"
    BOUNDARY = "boundary"


@dataclass(frozen=True)
class MixedStrategyProfile:
    """Behavioral-strategy representation.

    x is Pr(s_R | I), y is Pr(s_R | C). The verifier policy maps each
    information set (signal, evidence) to probabilities over G, H and D.
    """

    x: float
    y: float
    verifier_policy: Mapping[
        tuple[Signal, Evidence], Mapping[VerifierAction, float]
    ]


@dataclass(frozen=True)
class RegretReport:
    intact_regret: float
    compromised_regret: float
    verifier_regrets: Mapping[tuple[Signal, Evidence], float]
    max_regret: float
    intact_signal_utilities: Mapping[Signal, float]
    compromised_signal_utilities: Mapping[Signal, float]
    posteriors: Mapping[tuple[Signal, Evidence], float]
    information_set_probabilities: Mapping[tuple[Signal, Evidence], float]


@dataclass(frozen=True)
class MixedEquilibrium:
    profile: MixedStrategyProfile
    report: RegretReport
    classification: MixedClassification
    objective_value: float
    solver_success: bool


def _signal_probability(prob_reinforced: float, signal: Signal) -> float:
    return prob_reinforced if signal == Signal.REINFORCED else 1.0 - prob_reinforced


def _information_set_probability(
    signal: Signal,
    evidence: Evidence,
    x: float,
    y: float,
    params: GameParams,
) -> float:
    p = params.prior_compromised
    sigma_i = _signal_probability(x, signal)
    sigma_c = _signal_probability(y, signal)
    return (
        (1.0 - p)
        * sigma_i
        * evidence_likelihood(AgentType.INTACT, signal, evidence, params)
        + p
        * sigma_c
        * evidence_likelihood(AgentType.COMPROMISED, signal, evidence, params)
    )


def _posterior(
    signal: Signal,
    evidence: Evidence,
    x: float,
    y: float,
    params: GameParams,
    off_path_belief: float,
    probability_tolerance: float,
) -> tuple[float, float]:
    probability = _information_set_probability(signal, evidence, x, y, params)
    if probability <= probability_tolerance:
        return off_path_belief, probability

    p = params.prior_compromised
    sigma_c = _signal_probability(y, signal)
    numerator = (
        p
        * sigma_c
        * evidence_likelihood(AgentType.COMPROMISED, signal, evidence, params)
    )
    return numerator / probability, probability


def _normalize_policy_pair(g: float, h: float) -> dict[VerifierAction, float]:
    """Convert two bounded coordinates into a valid simplex point.

    The optimizer enforces g+h<=1. Clipping here is defensive against tiny
    floating-point violations returned by numerical solvers.
    """
    g = float(np.clip(g, 0.0, 1.0))
    h = float(np.clip(h, 0.0, 1.0))
    total = g + h
    if total > 1.0:
        g /= total
        h /= total
    return {
        VerifierAction.GRANT: g,
        VerifierAction.CHALLENGE: h,
        VerifierAction.DENY: 1.0 - g - h,
    }


def vector_to_profile(z: Sequence[float]) -> MixedStrategyProfile:
    if len(z) != 10:
        raise ValueError("A mixed-strategy vector must have length 10.")
    x = float(np.clip(z[0], 0.0, 1.0))
    y = float(np.clip(z[1], 0.0, 1.0))
    policy = {}
    cursor = 2
    for info_set in INFO_SETS:
        policy[info_set] = _normalize_policy_pair(z[cursor], z[cursor + 1])
        cursor += 2
    return MixedStrategyProfile(x=x, y=y, verifier_policy=policy)


def profile_to_vector(profile: MixedStrategyProfile) -> np.ndarray:
    values = [profile.x, profile.y]
    for info_set in INFO_SETS:
        probs = profile.verifier_policy[info_set]
        values.extend(
            [probs[VerifierAction.GRANT], probs[VerifierAction.CHALLENGE]]
        )
    return np.asarray(values, dtype=float)


def _expected_type_utility_for_signal(
    agent_type: AgentType,
    signal: Signal,
    profile: MixedStrategyProfile,
    params: GameParams,
) -> float:
    total = 0.0
    for evidence in Evidence:
        evidence_probability = evidence_likelihood(agent_type, signal, evidence, params)
        action_mix = profile.verifier_policy[(signal, evidence)]
        conditional_utility = sum(
            action_probability * agent_utility(agent_type, signal, action, params)
            for action, action_probability in action_mix.items()
        )
        total += evidence_probability * conditional_utility
    return total


def evaluate_profile(
    profile: MixedStrategyProfile,
    params: GameParams,
    *,
    off_path_belief: float = 1.0,
    probability_tolerance: float = 1e-12,
) -> RegretReport:
    """Compute sequential regrets for a behavioral-strategy profile.

    For zero-probability information sets, the supplied off-path belief is used.
    The default 1.0 corresponds to the conservative operational rule from
    Stage 1. On-path beliefs always follow Bayes' rule.
    """
    params.validate()
    if not 0.0 <= off_path_belief <= 1.0:
        raise ValueError("off_path_belief must be in [0, 1].")

    intact_utilities = {
        signal: _expected_type_utility_for_signal(
            AgentType.INTACT, signal, profile, params
        )
        for signal in Signal
    }
    compromised_utilities = {
        signal: _expected_type_utility_for_signal(
            AgentType.COMPROMISED, signal, profile, params
        )
        for signal in Signal
    }

    intact_realized = (
        profile.x * intact_utilities[Signal.REINFORCED]
        + (1.0 - profile.x) * intact_utilities[Signal.BASIC]
    )
    compromised_realized = (
        profile.y * compromised_utilities[Signal.REINFORCED]
        + (1.0 - profile.y) * compromised_utilities[Signal.BASIC]
    )
    intact_regret = max(intact_utilities.values()) - intact_realized
    compromised_regret = max(compromised_utilities.values()) - compromised_realized

    posteriors: dict[tuple[Signal, Evidence], float] = {}
    info_probabilities: dict[tuple[Signal, Evidence], float] = {}
    verifier_regrets: dict[tuple[Signal, Evidence], float] = {}

    for info_set in INFO_SETS:
        signal, evidence = info_set
        mu, probability = _posterior(
            signal,
            evidence,
            profile.x,
            profile.y,
            params,
            off_path_belief,
            probability_tolerance,
        )
        posteriors[info_set] = mu
        info_probabilities[info_set] = probability

        action_utilities = {
            action: verifier_expected_utility(action, mu, params)
            for action in ACTIONS
        }
        mixed_utility = sum(
            profile.verifier_policy[info_set][action] * action_utilities[action]
            for action in ACTIONS
        )
        verifier_regrets[info_set] = max(action_utilities.values()) - mixed_utility

    all_regrets = [intact_regret, compromised_regret, *verifier_regrets.values()]
    max_regret = max(max(0.0, value) for value in all_regrets)

    return RegretReport(
        intact_regret=max(0.0, intact_regret),
        compromised_regret=max(0.0, compromised_regret),
        verifier_regrets={k: max(0.0, v) for k, v in verifier_regrets.items()},
        max_regret=max_regret,
        intact_signal_utilities=intact_utilities,
        compromised_signal_utilities=compromised_utilities,
        posteriors=posteriors,
        information_set_probabilities=info_probabilities,
    )


def _snap_to_boundary(value: float, tolerance: float) -> float:
    """Snap a sender probability to 0 or 1 when within ``tolerance`` of it.

    SLSQP frequently returns values like 2e-6 instead of exactly 0 for a type
    that should never send a given signal. Left unsnapped, such residues push
    a genuinely off-path information set (probability ~1e-6) just above
    ``probability_tolerance`` in ``evaluate_profile``, which then computes an
    on-path Bayes posterior from noise instead of falling back to
    ``off_path_belief``. The resulting policy at that information set is
    calibrated to the noisy posterior, not to the true boundary belief, which
    both misrepresents the equilibrium and produces near-duplicate boundary
    solutions that ``_deduplicate`` fails to merge (their verifier policy at
    the affected information set differs enough from the true boundary
    solution to exceed ``deduplication_tolerance``, even though x and y do
    not). Snapping before evaluation makes the on/off-path classification
    match what the sender strategy actually is, so spurious candidates are
    rejected by the regret check itself rather than needing to be merged
    after the fact.
    """
    if value <= tolerance:
        return 0.0
    if value >= 1.0 - tolerance:
        return 1.0
    return value


def _classification(x: float, y: float, boundary_tolerance: float) -> MixedClassification:
    interior_x = boundary_tolerance < x < 1.0 - boundary_tolerance
    interior_y = boundary_tolerance < y < 1.0 - boundary_tolerance
    if interior_x and interior_y:
        return MixedClassification.FULLY_MIXED
    if interior_x or interior_y:
        return MixedClassification.SEMI_MIXED
    return MixedClassification.BOUNDARY


def _objective(
    z: np.ndarray,
    params: GameParams,
    off_path_belief: float,
    probability_tolerance: float,
) -> float:
    profile = vector_to_profile(z)
    report = evaluate_profile(
        profile,
        params,
        off_path_belief=off_path_belief,
        probability_tolerance=probability_tolerance,
    )
    # Squared component regrets provide a smoother objective than max regret,
    # while acceptance is still based on the independently recomputed max regret.
    components = [
        report.intact_regret,
        report.compromised_regret,
        *report.verifier_regrets.values(),
    ]
    return float(sum(value * value for value in components))


def _constraints() -> list[dict]:
    constraints = []
    for offset in range(2, 10, 2):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda z, i=offset: 1.0 - z[i] - z[i + 1],
            }
        )
    return constraints


def _deduplicate(
    candidates: Iterable[MixedEquilibrium],
    distance_tolerance: float,
) -> list[MixedEquilibrium]:
    unique: list[MixedEquilibrium] = []
    for candidate in sorted(candidates, key=lambda item: item.report.max_regret):
        vector = profile_to_vector(candidate.profile)
        if any(
            np.linalg.norm(vector - profile_to_vector(existing.profile), ord=np.inf)
            <= distance_tolerance
            for existing in unique
        ):
            continue
        unique.append(candidate)
    return unique


def find_mixed_equilibria(
    params: GameParams,
    *,
    tolerance: float = 1e-5,
    n_random_starts: int = 32,
    seed: int = 7,
    off_path_belief: float = 1.0,
    probability_tolerance: float = 1e-12,
    boundary_tolerance: float = 1e-4,
    snap_tolerance: float = 1e-5,
    deduplication_tolerance: float = 1e-3,
    use_global_search: bool = True,
    maxiter: int = 1_000,
) -> list[MixedEquilibrium]:
    """Search for approximate mixed, semi-mixed and boundary equilibria.

    ``snap_tolerance`` controls only numerical normalization of sender
    probabilities near 0 or 1. ``boundary_tolerance`` controls only the
    reported equilibrium classification. Keeping these roles separate avoids
    collapsing economically meaningful small mixing probabilities merely
    because a broader tolerance is useful for classification.

    The routine combines optional differential evolution with multi-start SLSQP.
    Every returned profile is revalidated independently and must have maximum
    unilateral regret no greater than ``tolerance``.
    """
    params.validate()
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if n_random_starts < 0:
        raise ValueError("n_random_starts must be nonnegative.")
    if probability_tolerance < 0:
        raise ValueError("probability_tolerance must be nonnegative.")
    if not 0.0 <= snap_tolerance < 0.5:
        raise ValueError("snap_tolerance must be in [0, 0.5).")
    if not 0.0 <= boundary_tolerance < 0.5:
        raise ValueError("boundary_tolerance must be in [0, 0.5).")
    if snap_tolerance > boundary_tolerance:
        raise ValueError("snap_tolerance must not exceed boundary_tolerance.")
    if deduplication_tolerance < 0:
        raise ValueError("deduplication_tolerance must be nonnegative.")

    rng = np.random.default_rng(seed)
    bounds = [(0.0, 1.0)] * 10
    constraints = _constraints()
    seeds: list[np.ndarray] = []

    # Exact pure-PBE seeds make boundary equilibria reproducible and prevent
    # the local solver from having to cross off-path belief discontinuities.
    from pure_equilibria import OffPathMode, find_pure_equilibria

    for pure_equilibrium in find_pure_equilibria(
        params, off_path_mode=OffPathMode.CONSERVATIVE
    ):
        pure_policy = {}
        for info_set, action in pure_equilibrium.verifier_actions.items():
            pure_policy[info_set] = {
                candidate: 1.0 if candidate == action else 0.0
                for candidate in ACTIONS
            }
        pure_profile = MixedStrategyProfile(
            x=1.0 if pure_equilibrium.intact_signal == Signal.REINFORCED else 0.0,
            y=1.0 if pure_equilibrium.compromised_signal == Signal.REINFORCED else 0.0,
            verifier_policy=pure_policy,
        )
        seeds.append(profile_to_vector(pure_profile))

    # Strategic corner and midpoint seeds improve coverage and reproducibility.
    for x in (0.0, 0.5, 1.0):
        for y in (0.0, 0.5, 1.0):
            z = np.full(10, 1.0 / 3.0)
            z[0], z[1] = x, y
            seeds.append(z)

    for _ in range(n_random_starts):
        z = rng.uniform(0.0, 1.0, size=10)
        for offset in range(2, 10, 2):
            pair = rng.dirichlet(np.ones(3))
            z[offset], z[offset + 1] = pair[0], pair[1]
        seeds.append(z)

    if use_global_search:
        global_result = differential_evolution(
            lambda z: _objective(
                z, params, off_path_belief, probability_tolerance
            ),
            bounds=bounds,
            constraints=(),
            seed=seed,
            maxiter=max(50, maxiter // 5),
            popsize=10,
            polish=False,
            updating="immediate",
        )
        global_seed = np.asarray(global_result.x, dtype=float)
        # Project receiver pairs to the simplex before local refinement.
        for offset in range(2, 10, 2):
            pair = _normalize_policy_pair(global_seed[offset], global_seed[offset + 1])
            global_seed[offset] = pair[VerifierAction.GRANT]
            global_seed[offset + 1] = pair[VerifierAction.CHALLENGE]
        seeds.append(global_seed)

    accepted: list[MixedEquilibrium] = []
    for start in seeds:
        result = minimize(
            lambda z: _objective(
                z, params, off_path_belief, probability_tolerance
            ),
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": 1e-14, "disp": False},
        )
        if not np.all(np.isfinite(result.x)):
            continue
        z = np.array(result.x, dtype=float)
        z[0] = _snap_to_boundary(z[0], snap_tolerance)
        z[1] = _snap_to_boundary(z[1], snap_tolerance)
        profile = vector_to_profile(z)
        report = evaluate_profile(
            profile,
            params,
            off_path_belief=off_path_belief,
            probability_tolerance=probability_tolerance,
        )
        if report.max_regret > tolerance:
            continue
        # Recomputed from the post-snap report, not result.fun: result.fun was
        # evaluated at the pre-snap z, so reusing it here could report an
        # objective value inconsistent with the regrets actually stored above.
        objective_value = float(
            report.intact_regret ** 2
            + report.compromised_regret ** 2
            + sum(value * value for value in report.verifier_regrets.values())
        )
        accepted.append(
            MixedEquilibrium(
                profile=profile,
                report=report,
                classification=_classification(
                    profile.x, profile.y, boundary_tolerance
                ),
                objective_value=objective_value,
                solver_success=bool(result.success),
            )
        )

    return _deduplicate(accepted, deduplication_tolerance)
