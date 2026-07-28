from model import DetectorParams, GameParams
from pure_equilibria import find_pure_equilibria, OffPathMode


def _profiles(equilibria):
    return frozenset(
        (eq.intact_signal.value, eq.compromised_signal.value) for eq in equilibria
    )


def test_zero_probability_evidence_branch_is_handled_per_information_set():
    # alpha_reinforced=0 e beta_reinforced=1: ambos os tipos que mandam s_R tem
    # um dos dois resultados de evidencia com probabilidade exatamente zero.
    # Antes da correcao de _off_path_information_sets (que classifica info-sets
    # fora do caminho por (signal, evidence), nao por sinal inteiro), o perfil
    # separador I->s_R, C->s_B levava a um ValueError nao tratado em (s_R, e_1)
    # -- o unico emissor de s_R (o integro) nunca gera e_1 --, e o perfil inteiro
    # era descartado silenciosamente por find_pure_equilibria.
    detector = DetectorParams(
        alpha_basic=0.15,
        beta_basic=0.70,
        alpha_reinforced=0.0,
        beta_reinforced=1.0,
    )
    params = GameParams(detector=detector)

    equilibria = find_pure_equilibria(
        params,
        off_path_mode=OffPathMode.CONSERVATIVE,
    )

    # Verificado empiricamente contra a versao corrigida: pooling em s_B e o
    # separador desejado (I->s_R, C->s_B) sobrevivem; nenhum equilibrio deveria
    # ser silenciosamente descartado por causa do info-set de probabilidade zero.
    assert _profiles(equilibria) == frozenset({("s_B", "s_B"), ("s_R", "s_B")})
