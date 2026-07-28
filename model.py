from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class AgentType(str, Enum):
    INTACT = "I"
    COMPROMISED = "C"


class Signal(str, Enum):
    BASIC = "s_B"
    REINFORCED = "s_R"


class Evidence(str, Enum):
    NO_ALERT = "e_0"
    ALERT = "e_1"


class VerifierAction(str, Enum):
    GRANT = "G"
    CHALLENGE = "H"
    DENY = "D"


@dataclass(frozen=True)
class DetectorParams:
    alpha_basic: float = 0.15
    beta_basic: float = 0.70
    alpha_reinforced: float = 0.05
    beta_reinforced: float = 0.90

    def alpha(self, signal: Signal) -> float:
        return self.alpha_basic if signal == Signal.BASIC else self.alpha_reinforced

    def beta(self, signal: Signal) -> float:
        return self.beta_basic if signal == Signal.BASIC else self.beta_reinforced

    def has_full_evidence_support(self, signal: Signal) -> bool:
        """Return True when both evidence outcomes have positive probability for both types."""
        alpha = self.alpha(signal)
        beta = self.beta(signal)
        return 0.0 < alpha < 1.0 and 0.0 < beta < 1.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.beta_basic <= self.alpha_basic:
            raise ValueError("beta_basic must be greater than alpha_basic.")
        if self.beta_reinforced <= self.alpha_reinforced:
            raise ValueError("beta_reinforced must be greater than alpha_reinforced.")


@dataclass(frozen=True)
class VerifierUtilityParams:
    benefit_legitimate: float = 1.0
    loss_compromised_grant: float = 4.0
    loss_legitimate_deny: float = 1.0
    challenge_cost: float = 0.20
    legitimate_challenge_loss: float = 0.15
    compromised_challenge_loss: float = 0.50

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")


@dataclass(frozen=True)
class AgentUtilityParams:
    intact_benefit: float = 1.0
    compromised_gain: float = 1.0
    intact_reinforced_cost: float = 0.10
    compromised_reinforced_cost: float = 0.25
    intact_challenge_cost: float = 0.15
    compromised_challenge_cost: float = 0.10
    intact_deny_loss: float = 0.75
    compromised_detection_penalty: float = 0.20
    residual_gain_reinforced: float = 0.25
    residual_gain_challenge_basic: float = 0.30
    residual_gain_challenge_reinforced: float = 0.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        for name in (
            "residual_gain_reinforced",
            "residual_gain_challenge_basic",
            "residual_gain_challenge_reinforced",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")


@dataclass(frozen=True)
class GameParams:
    prior_compromised: float = 0.20
    detector: DetectorParams = DetectorParams()
    verifier_utility: VerifierUtilityParams = VerifierUtilityParams()
    agent_utility: AgentUtilityParams = AgentUtilityParams()

    def validate(self) -> None:
        if not 0.0 <= self.prior_compromised <= 1.0:
            raise ValueError("prior_compromised must be in [0, 1].")
        self.detector.validate()
        self.verifier_utility.validate()
        self.agent_utility.validate()


PureStrategy = Dict[AgentType, Signal]
