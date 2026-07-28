import pytest
from beliefs import posterior_compromised
from model import Evidence, GameParams, Signal


def test_alert_increases_posterior_when_detector_is_informative():
    params = GameParams()
    sigma_i = {Signal.BASIC: 1.0, Signal.REINFORCED: 0.0}
    sigma_c = {Signal.BASIC: 1.0, Signal.REINFORCED: 0.0}
    mu_alert = posterior_compromised(Signal.BASIC, Evidence.ALERT, sigma_i, sigma_c, params)
    mu_no_alert = posterior_compromised(Signal.BASIC, Evidence.NO_ALERT, sigma_i, sigma_c, params)
    assert mu_alert > params.prior_compromised > mu_no_alert


def test_off_path_requires_explicit_belief():
    params = GameParams()
    sigma_i = {Signal.BASIC: 1.0, Signal.REINFORCED: 0.0}
    sigma_c = {Signal.BASIC: 1.0, Signal.REINFORCED: 0.0}
    with pytest.raises(ValueError):
        posterior_compromised(Signal.REINFORCED, Evidence.ALERT, sigma_i, sigma_c, params)
    mu = posterior_compromised(Signal.REINFORCED, Evidence.ALERT, sigma_i, sigma_c, params, off_path_belief=0.8)
    assert mu == pytest.approx(0.8)
