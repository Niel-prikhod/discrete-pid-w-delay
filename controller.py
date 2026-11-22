"""Discrete P, I, PI and PID control laws, independent of plant simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Gains:
    """Continuous-style proportional, integral and derivative tuning gains."""

    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0


class DiscreteController:
    """Stateful discrete controller with coefficients derived from its gains.

    I uses rectangular integration; PI and PID use trapezoidal integration.
    PID uses an unfiltered backward difference on the error.
    """

    def __init__(self, kind: str, gains: Gains, sample_period: float) -> None:
        """Validate configuration and initialize the controller at rest."""
        if kind not in ("P", "I", "PI", "PID"):
            raise ValueError(f"Unknown controller structure: {kind}")
        if not math.isfinite(sample_period) or sample_period <= 0:
            raise ValueError("Sample period must be positive and finite")
        if any(not math.isfinite(value) or value < 0 for value in (gains.kp, gains.ki, gains.kd)):
            raise ValueError("Controller gains must be nonnegative and finite")
        self._kind = kind
        self._gains = gains
        self._sample_period = sample_period
        self._coefficients = self._calculate_coefficients()
        self.reset()

    def _calculate_coefficients(self) -> tuple[float, float, float]:
        """Calculate incremental coefficients using the selected discretization."""
        kp, ki, kd = self._gains.kp, self._gains.ki, self._gains.kd
        ts = self._sample_period
        if self._kind == "P":
            return kp, -kp, 0.0
        if self._kind == "I":
            return ki * ts, 0.0, 0.0
        derivative = kd / ts if self._kind == "PID" else 0.0
        return (
            kp + ki * ts / 2.0 + derivative,
            -kp + ki * ts / 2.0 - 2.0 * derivative,
            derivative,
        )

    @property
    def coefficients(self) -> tuple[float, float, float]:
        """Return the calculated incremental coefficients q0, q1 and q2."""
        return self._coefficients

    def reset(self) -> None:
        """Reset the previous control and error samples to zero."""
        self._previous_input = 0.0
        self._error_1 = 0.0
        self._error_2 = 0.0

    def update(self, error: float) -> float:
        """Calculate the current control command and advance controller state."""
        if self._kind == "P":
            command = self._gains.kp * error
        else:
            q0, q1, q2 = self._coefficients
            command = self._previous_input + q0 * error + q1 * self._error_1 + q2 * self._error_2
        self._previous_input = command
        self._error_2, self._error_1 = self._error_1, error
        return command
