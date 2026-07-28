import pytest

from d1_refinement import (
    DEFAULT_ACTIONS,
    assess_pooling_equilibrium_d1,
    candidate_message_beliefs,
    d1_comparison,
    evidence_posterior_from_message_belief,
    prune_types_d1,
)
from model import AgentType, AgentUtilityParams, Evidence, GameParams, Signal
from pure_equilibria import OffPathMode, find_pure_equilibria


def _pooling_equilibrium(params, signal):
    equilibria = find_pure_equilibria(
        params, off_path_mode=OffPathMode.CONSERVATIVE
    )
    return next(
        eq for eq in equilibria
        if eq.intact_signal == signal and eq.compromised_signal == signal
    )


def _equilibrium_payoffs(eq):
    return {
        AgentType.INTACT: eq.intact_follow_utility,
        AgentType.COMPROMISED: eq.compromised_follow_utility,
    }


def test_evidence_update_from_message_belief():
    params = GameParams()
    mu_alert = evidence_posterior_from_message_belief(
        0.2, Signal.BASIC, Evidence.ALERT, params
    )
    mu_no_alert = evidence_posterior_from_message_belief(
        0.2, Signal.BASIC, Evidence.NO_ALERT, params
    )
    assert mu_alert > 0.2 > mu_no_alert


def test_candidate_beliefs_include_boundaries():
    beliefs = candidate_message_beliefs(
        Signal.REINFORCED,
        GameParams(),
        (AgentType.INTACT, AgentType.COMPROMISED),
    )
    assert beliefs[0] == 0.0
    assert beliefs[-1] == 1.0
    assert len(beliefs) > 2


def test_d1_comparison_returns_well_formed_lp_result():
    params = GameParams()
    eq = _pooling_equilibrium(params, Signal.BASIC)
    payoffs = _equilibrium_payoffs(eq)
    result = d1_comparison(
        AgentType.INTACT,
        AgentType.COMPROMISED,
        Signal.REINFORCED,
        payoffs,
        params,
    )
    assert isinstance(result.eliminated, bool)
    if result.weak_gain_set_feasible:
        assert result.minimum_dominating_gain is not None


def test_formal_d1_assessment_is_deterministic():
    params = GameParams()
    eq = _pooling_equilibrium(params, Signal.BASIC)
    payoffs = _equilibrium_payoffs(eq)
    first = assess_pooling_equilibrium_d1(
        Signal.BASIC, payoffs, params
    )
    second = assess_pooling_equilibrium_d1(
        Signal.BASIC, payoffs, params
    )
    assert first == second


def test_constructed_pooling_reinforced_can_be_assessed():
    # rho_R=0.25 (padrao) e c_C=0.03 satisfazem c_C <= 0.1*rho_R + 0.02 = 0.045,
    # a condicao de existencia do pooling reforcado sob a politica G/D on-path da
    # configuracao-base (ver analytic_propositions.md, secao 3). Verificado
    # empiricamente: find_pure_equilibria retorna (s_B,s_B) e (s_R,s_R) aqui.
    #
    # O cenario "construido" usado no resto da suite (c_C=0.35, rho_R=0.05) foi
    # escolhido para produzir o SEPARADOR desejado, e por isso nao tem pooling
    # reforcado (find_pure_equilibria so retorna (s_B,s_B) e (s_R,s_B) la) --
    # usa-lo aqui fazia o "if pooling:" abaixo nunca executar, e o teste passava
    # sem checar nada.
    params = GameParams(
        agent_utility=AgentUtilityParams(
            residual_gain_reinforced=0.25,
            compromised_reinforced_cost=0.03,
        )
    )
    equilibria = find_pure_equilibria(
        params, off_path_mode=OffPathMode.CONSERVATIVE
    )
    pooling = [
        eq for eq in equilibria
        if eq.intact_signal == Signal.REINFORCED
        and eq.compromised_signal == Signal.REINFORCED
    ]
    # Falha ruidosamente em vez de pular silenciosamente o corpo do teste caso
    # os parametros deixem de produzir pooling reforcado no futuro.
    assert pooling, "cenario deveria ter pooling reforcado como PBE; parametros mudaram?"

    assessment = assess_pooling_equilibrium_d1(
        Signal.REINFORCED,
        _equilibrium_payoffs(pooling[0]),
        params,
    )
    # Previsao teorica (Corolario 1 / analytic_propositions.md): sempre que o
    # pooling reforcado existe nessa regiao de parametros, ele sobrevive ao D1.
    assert assessment.status == "supported"
    assert assessment.passes_d1


def test_pruning_never_retains_eliminated_type():
    params = GameParams()
    eq = _pooling_equilibrium(params, Signal.BASIC)
    pruning = prune_types_d1(
        Signal.BASIC, _equilibrium_payoffs(eq), params
    )
    assert set(pruning.retained_types).isdisjoint(
        set(pruning.eliminated_types)
    )
