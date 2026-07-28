from __future__ import annotations

from model import AgentType, GameParams, Signal, VerifierAction


def verifier_expected_utility(action: VerifierAction, posterior_c: float, params: GameParams) -> float:
    u = params.verifier_utility
    mu = posterior_c
    if action == VerifierAction.GRANT:
        return (1.0 - mu) * u.benefit_legitimate - mu * u.loss_compromised_grant
    if action == VerifierAction.CHALLENGE:
        return ((1.0 - mu) * (u.benefit_legitimate - u.challenge_cost - u.legitimate_challenge_loss)
                + mu * (-u.challenge_cost - u.compromised_challenge_loss))
    return -(1.0 - mu) * u.loss_legitimate_deny


def intact_utility(signal: Signal, action: VerifierAction, params: GameParams) -> float:
    u = params.agent_utility
    signal_cost = u.intact_reinforced_cost if signal == Signal.REINFORCED else 0.0
    if action == VerifierAction.GRANT:
        return u.intact_benefit - signal_cost
    if action == VerifierAction.CHALLENGE:
        return u.intact_benefit - u.intact_challenge_cost - signal_cost
    return -u.intact_deny_loss - signal_cost


def compromised_utility(signal: Signal, action: VerifierAction, params: GameParams) -> float:
    u = params.agent_utility
    signal_cost = u.compromised_reinforced_cost if signal == Signal.REINFORCED else 0.0
    if action == VerifierAction.GRANT:
        residual = u.residual_gain_reinforced if signal == Signal.REINFORCED else 1.0
        return residual * u.compromised_gain - signal_cost
    if action == VerifierAction.CHALLENGE:
        residual = (u.residual_gain_challenge_reinforced if signal == Signal.REINFORCED
                    else u.residual_gain_challenge_basic)
        return (residual * u.compromised_gain - u.compromised_challenge_cost
                - u.compromised_detection_penalty - signal_cost)
    return -u.compromised_detection_penalty - signal_cost


def agent_utility(agent_type: AgentType, signal: Signal, action: VerifierAction, params: GameParams) -> float:
    if agent_type == AgentType.INTACT:
        return intact_utility(signal, action, params)
    return compromised_utility(signal, action, params)
