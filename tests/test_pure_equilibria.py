from model import GameParams
from pure_equilibria import find_pure_equilibria, OffPathMode


def test_all_returned_profiles_satisfy_no_profitable_deviation():
    eqs = find_pure_equilibria(GameParams(), off_path_mode=OffPathMode.GRID, grid_step=0.5)
    for eq in eqs:
        assert eq.intact_follow_utility >= eq.intact_deviation_utility - 1e-10
        assert eq.compromised_follow_utility >= eq.compromised_deviation_utility - 1e-10


def test_equilibrium_classification_is_valid():
    eqs = find_pure_equilibria(GameParams(), off_path_mode=OffPathMode.CONSERVATIVE)
    assert all(eq.classification in {"separating", "pooling_basic", "pooling_reinforced"} for eq in eqs)
