from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from beliefs import evidence_likelihood
from model import AgentType, Evidence, GameParams, Signal, VerifierAction
from utilities import agent_utility, verifier_expected_utility


DEFAULT_ACTIONS: tuple[VerifierAction, ...] = (
    VerifierAction.GRANT,
    VerifierAction.CHALLENGE,
    VerifierAction.DENY,
)


@dataclass(frozen=True)
class D1TypeComparison:
    eliminated_type: AgentType
    dominating_type: AgentType
    weak_gain_set_feasible: bool
    minimum_dominating_gain: float | None
    eliminated: bool


@dataclass(frozen=True)
class D1PruningResult:
    deviation_signal: Signal
    retained_types: tuple[AgentType, ...]
    eliminated_types: tuple[AgentType, ...]
    comparisons: tuple[D1TypeComparison, ...]


@dataclass(frozen=True)
class D1EquilibriumAssessment:
    passes_d1: bool
    status: str
    retained_types: tuple[AgentType, ...]
    eliminated_types: tuple[AgentType, ...]
    supporting_message_belief_c: float | None
    supporting_evidence_posteriors: Mapping[Evidence, float] | None
    supporting_actions: Mapping[Evidence, VerifierAction] | None


def _other_signal(signal: Signal) -> Signal:
    return Signal.REINFORCED if signal == Signal.BASIC else Signal.BASIC


def _response_variables(
    action_set: Sequence[VerifierAction],
) -> tuple[tuple[Evidence, VerifierAction], ...]:
    return tuple((evidence, action) for evidence in Evidence for action in action_set)


def deviation_gain_coefficients(
    agent_type: AgentType,
    deviation_signal: Signal,
    equilibrium_payoff: float,
    params: GameParams,
    action_set: Sequence[VerifierAction] = DEFAULT_ACTIONS,
) -> tuple[np.ndarray, float]:
    """Return coefficients c and constant b for gain(q)=c·q-b.

    q is a contingent mixed response with one simplex for each evidence outcome.
    """
    variables = _response_variables(action_set)
    coefficients = np.array(
        [
            evidence_likelihood(agent_type, deviation_signal, evidence, params)
            * agent_utility(agent_type, deviation_signal, action, params)
            for evidence, action in variables
        ],
        dtype=float,
    )
    return coefficients, float(equilibrium_payoff)


def _simplex_constraints(
    action_set: Sequence[VerifierAction],
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    n_actions = len(action_set)
    n_vars = 2 * n_actions
    a_eq = np.zeros((2, n_vars), dtype=float)
    a_eq[0, :n_actions] = 1.0
    a_eq[1, n_actions:] = 1.0
    b_eq = np.ones(2, dtype=float)
    bounds = [(0.0, 1.0)] * n_vars
    return a_eq, b_eq, bounds


def d1_comparison(
    candidate_type: AgentType,
    dominating_type: AgentType,
    deviation_signal: Signal,
    equilibrium_payoffs: Mapping[AgentType, float],
    params: GameParams,
    action_set: Sequence[VerifierAction] = DEFAULT_ACTIONS,
    tolerance: float = 1e-9,
) -> D1TypeComparison:
    """Test the formal D1 set inclusion W_candidate ⊂ S_dominating.

    W_candidate is the set of contingent receiver mixed responses for which the
    candidate type weakly benefits from deviating. S_dominating is the set for
    which the dominating type strictly benefits.

    The candidate is eliminated when every response in W_candidate gives the
    dominating type a strictly positive gain. This is checked by minimizing the
    dominating type's gain subject to candidate_gain >= 0.
    """
    c_candidate, b_candidate = deviation_gain_coefficients(
        candidate_type,
        deviation_signal,
        equilibrium_payoffs[candidate_type],
        params,
        action_set,
    )
    c_dom, b_dom = deviation_gain_coefficients(
        dominating_type,
        deviation_signal,
        equilibrium_payoffs[dominating_type],
        params,
        action_set,
    )
    a_eq, b_eq, bounds = _simplex_constraints(action_set)

    # candidate_gain = c_candidate·q - b_candidate >= 0
    # -> -c_candidate·q <= -b_candidate
    result = linprog(
        c=c_dom,
        A_ub=np.array([-c_candidate]),
        b_ub=np.array([-b_candidate]),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if result.status == 2:  # infeasible: deviation is never weakly profitable
        return D1TypeComparison(
            eliminated_type=candidate_type,
            dominating_type=dominating_type,
            weak_gain_set_feasible=False,
            minimum_dominating_gain=None,
            eliminated=True,
        )
    if not result.success:
        raise RuntimeError(f"D1 linear program failed: {result.message}")

    minimum_gain = float(result.fun - b_dom)
    return D1TypeComparison(
        eliminated_type=candidate_type,
        dominating_type=dominating_type,
        weak_gain_set_feasible=True,
        minimum_dominating_gain=minimum_gain,
        eliminated=minimum_gain > tolerance,
    )


def prune_types_d1(
    pooling_signal: Signal,
    equilibrium_payoffs: Mapping[AgentType, float],
    params: GameParams,
    action_set: Sequence[VerifierAction] = DEFAULT_ACTIONS,
    tolerance: float = 1e-9,
) -> D1PruningResult:
    """Apply the D1 pairwise set-inclusion rule for the two sender types."""
    deviation_signal = _other_signal(pooling_signal)
    comparisons = (
        d1_comparison(
            AgentType.INTACT,
            AgentType.COMPROMISED,
            deviation_signal,
            equilibrium_payoffs,
            params,
            action_set,
            tolerance,
        ),
        d1_comparison(
            AgentType.COMPROMISED,
            AgentType.INTACT,
            deviation_signal,
            equilibrium_payoffs,
            params,
            action_set,
            tolerance,
        ),
    )

    eliminated = {
        comparison.eliminated_type
        for comparison in comparisons
        if comparison.eliminated
    }
    retained = tuple(t for t in AgentType if t not in eliminated)
    return D1PruningResult(
        deviation_signal=deviation_signal,
        retained_types=retained,
        eliminated_types=tuple(t for t in AgentType if t in eliminated),
        comparisons=comparisons,
    )


def evidence_posterior_from_message_belief(
    message_belief_c: float,
    signal: Signal,
    evidence: Evidence,
    params: GameParams,
) -> float:
    """Update a D1-compatible belief after observing the detector evidence."""
    q = float(message_belief_c)
    likelihood_c = evidence_likelihood(
        AgentType.COMPROMISED, signal, evidence, params
    )
    likelihood_i = evidence_likelihood(
        AgentType.INTACT, signal, evidence, params
    )
    numerator = q * likelihood_c
    denominator = numerator + (1.0 - q) * likelihood_i
    if denominator <= 0.0:
        raise ValueError("Evidence has zero probability under all retained types.")
    return numerator / denominator


def receiver_best_actions(
    posterior_c: float,
    params: GameParams,
    action_set: Sequence[VerifierAction] = DEFAULT_ACTIONS,
    tolerance: float = 1e-10,
) -> tuple[VerifierAction, ...]:
    utilities = {
        action: verifier_expected_utility(action, posterior_c, params)
        for action in action_set
    }
    best_value = max(utilities.values())
    return tuple(
        action
        for action, value in utilities.items()
        if abs(value - best_value) <= tolerance
    )


def _receiver_indifference_posteriors(
    params: GameParams,
    action_set: Sequence[VerifierAction],
    tolerance: float = 1e-12,
) -> set[float]:
    """Return posterior values at which two receiver utility lines intersect."""
    points: set[float] = {0.0, 1.0}
    for a, b in product(action_set, repeat=2):
        if a.value >= b.value:
            continue
        ua0 = verifier_expected_utility(a, 0.0, params)
        ua1 = verifier_expected_utility(a, 1.0, params)
        ub0 = verifier_expected_utility(b, 0.0, params)
        ub1 = verifier_expected_utility(b, 1.0, params)
        intercept = ua0 - ub0
        slope = (ua1 - ua0) - (ub1 - ub0)
        if abs(slope) <= tolerance:
            continue
        mu = -intercept / slope
        if tolerance < mu < 1.0 - tolerance:
            points.add(float(mu))
    return points


def _message_belief_for_posterior(
    posterior_c: float,
    signal: Signal,
    evidence: Evidence,
    params: GameParams,
) -> float | None:
    mu = float(posterior_c)
    likelihood_c = evidence_likelihood(
        AgentType.COMPROMISED, signal, evidence, params
    )
    likelihood_i = evidence_likelihood(
        AgentType.INTACT, signal, evidence, params
    )
    denominator = likelihood_c * (1.0 - mu) + mu * likelihood_i
    if denominator <= 0.0:
        return None
    q = mu * likelihood_i / denominator
    return min(1.0, max(0.0, float(q)))


def candidate_message_beliefs(
    deviation_signal: Signal,
    params: GameParams,
    retained_types: Sequence[AgentType],
    action_set: Sequence[VerifierAction] = DEFAULT_ACTIONS,
) -> tuple[float, ...]:
    """Generate all belief regions where receiver best responses can change."""
    if retained_types == (AgentType.INTACT,) or set(retained_types) == {AgentType.INTACT}:
        return (0.0,)
    if retained_types == (AgentType.COMPROMISED,) or set(retained_types) == {AgentType.COMPROMISED}:
        return (1.0,)
    if not retained_types:
        return tuple()

    points = {0.0, 1.0}
    for mu in _receiver_indifference_posteriors(params, action_set):
        for evidence in Evidence:
            q = _message_belief_for_posterior(
                mu, deviation_signal, evidence, params
            )
            if q is not None:
                points.add(q)

    ordered = sorted(points)
    candidates = set(ordered)
    for left, right in zip(ordered, ordered[1:]):
        if right - left > 1e-12:
            candidates.add((left + right) / 2.0)
    return tuple(sorted(candidates))


def assess_pooling_equilibrium_d1(
    pooling_signal: Signal,
    equilibrium_payoffs: Mapping[AgentType, float],
    params: GameParams,
    action_set: Sequence[VerifierAction] = DEFAULT_ACTIONS,
    tolerance: float = 1e-9,
) -> D1EquilibriumAssessment:
    """Assess whether a pooling outcome can be supported after formal D1 pruning."""
    pruning = prune_types_d1(
        pooling_signal,
        equilibrium_payoffs,
        params,
        action_set,
        tolerance,
    )
    deviation_signal = pruning.deviation_signal

    if not pruning.retained_types:
        return D1EquilibriumAssessment(
            passes_d1=False,
            status="all_types_eliminated",
            retained_types=pruning.retained_types,
            eliminated_types=pruning.eliminated_types,
            supporting_message_belief_c=None,
            supporting_evidence_posteriors=None,
            supporting_actions=None,
        )

    for belief_c in candidate_message_beliefs(
        deviation_signal,
        params,
        pruning.retained_types,
        action_set,
    ):
        posteriors = {}
        action_sets = {}
        feasible = True
        for evidence in Evidence:
            try:
                posterior = evidence_posterior_from_message_belief(
                    belief_c, deviation_signal, evidence, params
                )
            except ValueError:
                feasible = False
                break
            posteriors[evidence] = posterior
            action_sets[evidence] = receiver_best_actions(
                posterior, params, action_set
            )
        if not feasible:
            continue

        for selected_actions in product(
            action_sets[Evidence.NO_ALERT],
            action_sets[Evidence.ALERT],
        ):
            action_map = {
                Evidence.NO_ALERT: selected_actions[0],
                Evidence.ALERT: selected_actions[1],
            }
            deviations_profitable = False
            for agent_type in AgentType:
                deviation_payoff = sum(
                    evidence_likelihood(
                        agent_type, deviation_signal, evidence, params
                    )
                    * agent_utility(
                        agent_type,
                        deviation_signal,
                        action_map[evidence],
                        params,
                    )
                    for evidence in Evidence
                )
                if (
                    deviation_payoff
                    > equilibrium_payoffs[agent_type] + tolerance
                ):
                    deviations_profitable = True
                    break
            if not deviations_profitable:
                return D1EquilibriumAssessment(
                    passes_d1=True,
                    status="supported",
                    retained_types=pruning.retained_types,
                    eliminated_types=pruning.eliminated_types,
                    supporting_message_belief_c=float(belief_c),
                    supporting_evidence_posteriors=posteriors,
                    supporting_actions=action_map,
                )

    return D1EquilibriumAssessment(
        passes_d1=False,
        status="no_supporting_d1_belief",
        retained_types=pruning.retained_types,
        eliminated_types=pruning.eliminated_types,
        supporting_message_belief_c=None,
        supporting_evidence_posteriors=None,
        supporting_actions=None,
    )
