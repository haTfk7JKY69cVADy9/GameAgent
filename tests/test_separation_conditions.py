from model import AgentUtilityParams, GameParams, Signal
from pure_equilibria import find_pure_equilibria, OffPathMode


def has_desired(eqs):
    return any(eq.intact_signal == Signal.REINFORCED and eq.compromised_signal == Signal.BASIC for eq in eqs)


def test_base_case_does_not_support_desired_separation():
    assert not has_desired(find_pure_equilibria(GameParams(), off_path_mode=OffPathMode.CONSERVATIVE))


def test_constructed_case_supports_desired_separation():
    u = AgentUtilityParams(compromised_reinforced_cost=0.35, residual_gain_reinforced=0.05)
    params = GameParams(agent_utility=u)
    assert has_desired(find_pure_equilibria(params, off_path_mode=OffPathMode.CONSERVATIVE))
