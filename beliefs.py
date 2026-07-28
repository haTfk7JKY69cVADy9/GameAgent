from __future__ import annotations

from typing import Mapping

from model import AgentType, Evidence, GameParams, Signal


def evidence_likelihood(agent_type: AgentType, signal: Signal, evidence: Evidence, params: GameParams) -> float:
    alpha = params.detector.alpha(signal)
    beta = params.detector.beta(signal)
    if agent_type == AgentType.INTACT:
        return alpha if evidence == Evidence.ALERT else 1.0 - alpha
    return beta if evidence == Evidence.ALERT else 1.0 - beta


def posterior_compromised(
    signal: Signal,
    evidence: Evidence,
    sigma_intact: Mapping[Signal, float],
    sigma_compromised: Mapping[Signal, float],
    params: GameParams,
    off_path_belief: float | None = None,
) -> float:
    p = params.prior_compromised
    likelihood_c = evidence_likelihood(AgentType.COMPROMISED, signal, evidence, params)
    likelihood_i = evidence_likelihood(AgentType.INTACT, signal, evidence, params)
    numerator = p * sigma_compromised[signal] * likelihood_c
    denominator = numerator + (1.0 - p) * sigma_intact[signal] * likelihood_i
    if denominator > 0:
        return numerator / denominator
    if off_path_belief is None:
        raise ValueError("Off-path information set encountered without an off_path_belief.")
    if not 0.0 <= off_path_belief <= 1.0:
        raise ValueError("off_path_belief must be in [0, 1].")
    return off_path_belief
