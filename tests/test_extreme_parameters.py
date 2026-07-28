from model import DetectorParams, GameParams
from pure_equilibria import OffPathMode, find_pure_equilibria


def _profiles(equilibria):
    return frozenset(
        (eq.intact_signal.value, eq.compromised_signal.value) for eq in equilibria
    )


def test_extreme_detector_rates_do_not_break_pure_enumeration():
    # Detector perfeito (alpha=0, beta=1) nos dois sinais: evidencia sozinha ja
    # revela o tipo, independente do sinal escolhido. Pooling em s_B e o
    # separador desejado sobrevivem; pooling reforcado nao (o integro nao tem
    # motivo para pagar c_I quando a evidencia ja o distingue de graca).
    params = GameParams(
        detector=DetectorParams(
            alpha_basic=0.0,
            beta_basic=1.0,
            alpha_reinforced=0.0,
            beta_reinforced=1.0,
        )
    )
    equilibria = find_pure_equilibria(
        params,
        off_path_mode=OffPathMode.CONSERVATIVE,
    )
    assert _profiles(equilibria) == frozenset({("s_B", "s_B"), ("s_R", "s_B")})


def test_zero_prior_does_not_break_zero_probability_information_sets():
    # p=0: o tipo comprometido nunca ocorre, entao qualquer pooling e sustentado
    # sob a crenca conservadora fora do caminho (mu=1 para o sinal nao emitido).
    # Os dois poolings sobrevivem porque nao ha comprometido para imitar nada.
    params = GameParams(prior_compromised=0.0)
    equilibria = find_pure_equilibria(
        params,
        off_path_mode=OffPathMode.CONSERVATIVE,
    )
    assert _profiles(equilibria) == frozenset({("s_B", "s_B"), ("s_R", "s_R")})


def test_unit_prior_does_not_break_zero_probability_information_sets():
    # p=1: todo agente e comprometido, entao a resposta on-path de qualquer
    # pooling e D em toda evidencia (mu=1 sempre acima de tau_HD). Pooling
    # reforcado se desfaz: o comprometido prefere mandar o sinal barato s_B e
    # ser negado do mesmo jeito a pagar c_C por s_R so para tambem ser negado.
    # So sobra pooling em s_B.
    params = GameParams(prior_compromised=1.0)
    equilibria = find_pure_equilibria(
        params,
        off_path_mode=OffPathMode.CONSERVATIVE,
    )
    assert _profiles(equilibria) == frozenset({("s_B", "s_B")})
