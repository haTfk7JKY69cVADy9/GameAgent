from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Dict, Mapping

from beliefs import evidence_likelihood
from model import AgentType, Evidence, GameParams, Signal, VerifierAction
from utilities import agent_utility
from verifier import VerifierDecision, verifier_policy


class OffPathMode(str, Enum):
    CONSERVATIVE = "conservative"
    GRID = "grid"


@dataclass(frozen=True)
class PureEquilibrium:
    intact_signal: Signal
    compromised_signal: Signal
    verifier_actions: Mapping[tuple[Signal, Evidence], VerifierAction]
    posteriors: Mapping[tuple[Signal, Evidence], float]
    intact_follow_utility: float
    intact_deviation_utility: float
    compromised_follow_utility: float
    compromised_deviation_utility: float
    off_path_beliefs: Mapping[tuple[Signal, Evidence], float]

    @property
    def classification(self) -> str:
        if self.intact_signal != self.compromised_signal:
            return "separating"
        return "pooling_basic" if self.intact_signal == Signal.BASIC else "pooling_reinforced"


def _pure_sigma(chosen: Signal) -> Dict[Signal, float]:
    return {
        Signal.BASIC: 1.0 if chosen == Signal.BASIC else 0.0,
        Signal.REINFORCED: 1.0 if chosen == Signal.REINFORCED else 0.0,
    }


def _off_path_information_sets(
    intact_signal: Signal,
    compromised_signal: Signal,
    params: GameParams,
) -> list[tuple[Signal, Evidence]]:
    """Identify zero-probability information sets at the (signal, evidence) level."""
    sigma_i = _pure_sigma(intact_signal)
    sigma_c = _pure_sigma(compromised_signal)
    p = params.prior_compromised

    off_path: list[tuple[Signal, Evidence]] = []
    for signal in Signal:
        for evidence in Evidence:
            likelihood_i = evidence_likelihood(
                AgentType.INTACT, signal, evidence, params
            )
            likelihood_c = evidence_likelihood(
                AgentType.COMPROMISED, signal, evidence, params
            )
            probability = (
                (1.0 - p) * sigma_i[signal] * likelihood_i
                + p * sigma_c[signal] * likelihood_c
            )
            if probability <= 0.0:
                off_path.append((signal, evidence))
    return off_path


def _off_path_belief_candidates(
    intact_signal: Signal,
    compromised_signal: Signal,
    params: GameParams,
    mode: str | OffPathMode,
    grid_step: float,
) -> list[dict[tuple[Signal, Evidence], float]]:
    off_path_sets = _off_path_information_sets(
        intact_signal, compromised_signal, params
    )
    if not off_path_sets:
        return [{}]

    if mode in (OffPathMode.CONSERVATIVE, OffPathMode.CONSERVATIVE.value):
        return [{info_set: 1.0 for info_set in off_path_sets}]

    if mode not in (OffPathMode.GRID, OffPathMode.GRID.value):
        raise ValueError(f"Unsupported off-path mode: {mode}")

    values = []
    x = 0.0
    while x < 1.0 + 1e-12:
        values.append(round(min(x, 1.0), 12))
        x += grid_step

    candidates = []
    for assignment in product(values, repeat=len(off_path_sets)):
        candidates.append(dict(zip(off_path_sets, assignment)))
    return candidates


def _expected_agent_utility(agent_type: AgentType, signal: Signal, action_map, params: GameParams) -> float:
    total = 0.0
    for evidence in Evidence:
        probability = evidence_likelihood(agent_type, signal, evidence, params)
        total += probability * agent_utility(agent_type, signal, action_map[(signal, evidence)], params)
    return total


def find_pure_equilibria(params: GameParams, off_path_mode: str = OffPathMode.CONSERVATIVE,
                          grid_step: float = 0.25, tol: float = 1e-10) -> list[PureEquilibrium]:
    params.validate()
    equilibria = []
    for intact_signal, compromised_signal in product(Signal, repeat=2):
        sigma_i, sigma_c = _pure_sigma(intact_signal), _pure_sigma(compromised_signal)
        for off_path_beliefs in _off_path_belief_candidates(intact_signal, compromised_signal, params, off_path_mode, grid_step):
            decisions = {}
            feasible = True
            for signal in Signal:
                for evidence in Evidence:
                    info_set = (signal, evidence)
                    try:
                        decisions[info_set] = verifier_policy(
                            signal, evidence, sigma_i, sigma_c, params, off_path_beliefs.get(info_set)
                        )
                    except ValueError:
                        feasible = False
                        break
                if not feasible:
                    break
            if not feasible:
                continue
            info_sets = [(s, e) for s in Signal for e in Evidence]
            best_action_sets = [decisions[i].best_actions for i in info_sets]
            for action_selection in product(*best_action_sets):
                action_map = dict(zip(info_sets, action_selection))
                alt_i = Signal.REINFORCED if intact_signal == Signal.BASIC else Signal.BASIC
                alt_c = Signal.REINFORCED if compromised_signal == Signal.BASIC else Signal.BASIC
                i_follow = _expected_agent_utility(AgentType.INTACT, intact_signal, action_map, params)
                i_deviate = _expected_agent_utility(AgentType.INTACT, alt_i, action_map, params)
                c_follow = _expected_agent_utility(AgentType.COMPROMISED, compromised_signal, action_map, params)
                c_deviate = _expected_agent_utility(AgentType.COMPROMISED, alt_c, action_map, params)
                if i_follow + tol < i_deviate or c_follow + tol < c_deviate:
                    continue
                eq = PureEquilibrium(
                    intact_signal, compromised_signal, action_map,
                    {i: decisions[i].posterior_c for i in info_sets},
                    i_follow, i_deviate, c_follow, c_deviate, off_path_beliefs
                )
                if eq not in equilibria:
                    equilibria.append(eq)
    return equilibria
