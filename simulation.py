#!/usr/bin/env python3
"""Discrete delayed-plant controller design, stability checks, and figures."""

from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import control
import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RESULTS_PATH = ROOT / "results.json"
TS = 0.1
DELAYS = tuple(range(1, 6))
PLOT_HORIZON_SECONDS = 40.0
SIMULATION_STEPS = int(PLOT_HORIZON_SECONDS / TS)


@dataclass(frozen=True)
class Gains:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0


@dataclass
class Plant:
    ad: np.ndarray
    bd: np.ndarray
    cd: np.ndarray
    dd: np.ndarray
    numerator_z_inverse: np.ndarray
    denominator_z_inverse: np.ndarray


GAIN_SCHEDULE: dict[str, dict[int, Gains]] = {
    "P": {
        1: Gains(kp=0.8),
        2: Gains(kp=0.7),
        3: Gains(kp=0.6),
        4: Gains(kp=0.5),
        5: Gains(kp=0.4),
    },
    "I": {
        1: Gains(ki=0.50),
        2: Gains(ki=0.42),
        3: Gains(ki=0.35),
        4: Gains(ki=0.28),
        5: Gains(ki=0.22),
    },
    "PI": {
        1: Gains(kp=0.80, ki=0.50),
        2: Gains(kp=0.75, ki=0.48),
        3: Gains(kp=0.70, ki=0.46),
        4: Gains(kp=0.65, ki=0.43),
        5: Gains(kp=0.60, ki=0.40),
    },
    "PID": {
        1: Gains(kp=1.10, ki=0.65, kd=0.12),
        2: Gains(kp=1.00, ki=0.60, kd=0.11),
        3: Gains(kp=0.90, ki=0.55, kd=0.10),
        4: Gains(kp=0.80, ki=0.50, kd=0.09),
        5: Gains(kp=0.70, ki=0.45, kd=0.08),
    },
}


def define_plant() -> Plant:
    """ZOH-discretize Gc(s)=1/(s+1)^2 and retain its state-space realization."""
    ac = np.array([[0.0, 1.0], [-1.0, -2.0]])
    bc = np.array([[0.0], [1.0]])
    cc = np.array([[1.0, 0.0]])
    dc = np.zeros((1, 1))
    ad, bd, cd, dd, _ = signal.cont2discrete(
        (ac, bc, cc, dc), TS, method="zoh")
    discrete_ss = control.ss(ad, bd, cd, dd, dt=TS)
    discrete_tf = control.ss2tf(discrete_ss)
    numerator, denominator = control.tfdata(discrete_tf)
    numerator = np.asarray(numerator, dtype=float).reshape(-1)
    denominator = np.asarray(denominator, dtype=float).reshape(-1)
    denominator_scale = denominator[0]
    denominator /= denominator_scale
    numerator /= denominator_scale
    numerator = np.pad(numerator, (len(denominator) - len(numerator), 0))
    return Plant(
        ad=np.asarray(ad, dtype=float),
        bd=np.asarray(bd, dtype=float),
        cd=np.asarray(cd, dtype=float),
        dd=np.asarray(dd, dtype=float),
        numerator_z_inverse=numerator,
        denominator_z_inverse=denominator,
    )


def controller_coefficients(kind: str, gains: Gains) -> tuple[float, float, float]:
    """Return q0,q1,q2 for the incremental-controller representation."""
    if kind == "P":
        return gains.kp, -gains.kp, 0.0
    if kind == "I":
        return gains.ki * TS, 0.0, 0.0
    if kind == "PI":
        return (
            gains.kp + gains.ki * TS / 2.0,
            -gains.kp + gains.ki * TS / 2.0,
            0.0,
        )
    if kind == "PID":
        return (
            gains.kp + gains.ki * TS / 2.0 + gains.kd / TS,
            -gains.kp + gains.ki * TS / 2.0 - 2.0 * gains.kd / TS,
            gains.kd / TS,
        )
    raise ValueError(f"Unknown controller structure: {kind}")


def characteristic_polynomial(
    plant: Plant, kind: str, gains: Gains, delay: int
) -> np.ndarray:
    """Return the coefficients to pass directly to ``np.roots``.

    The polynomial routines here store coefficients in ascending powers of
    z^-1. After multiplying by the highest power of z, this same sequence is
    descending powers of z, which is the ordering expected by ``np.roots``.
    """
    if delay < 0:
        raise ValueError("Transport delay must be nonnegative")
    denominator = plant.denominator_z_inverse
    numerator = plant.numerator_z_inverse
    if kind == "P":
        open_loop_numerator = np.r_[np.zeros(delay), numerator * gains.kp]
        return np.polynomial.polynomial.polyadd(denominator, open_loop_numerator)

    q = np.asarray(controller_coefficients(kind, gains))
    controller_denominator = np.array([1.0, -1.0])
    closed_loop = np.polynomial.polynomial.polymul(
        denominator, controller_denominator
    )
    open_loop_numerator = np.r_[
        np.zeros(delay), np.polynomial.polynomial.polymul(numerator, q)
    ]
    return np.polynomial.polynomial.polyadd(closed_loop, open_loop_numerator)


def schur_cohn_test(coefficients: np.ndarray) -> dict[str, Any]:
    """Apply the real Schur-Cohn recursion and return its reflection ratios."""
    reduced = np.asarray(coefficients, dtype=float).copy()
    if reduced.ndim != 1 or len(reduced) < 2 or reduced[0] == 0.0:
        raise ValueError("Schur-Cohn test requires a nonconstant polynomial")
    reduced /= reduced[0]
    steps: list[dict[str, float | int | bool]] = []
    while len(reduced) > 1:
        degree = len(reduced) - 1
        reflection = float(reduced[-1])
        step_stable = abs(reflection) < 1.0
        steps.append(
            {
                "degree": degree,
                "reflection_coefficient": reflection,
                "absolute_reflection_coefficient": abs(reflection),
                "stable": step_stable,
            }
        )
        if not step_stable:
            break
        next_polynomial = reduced - reflection * reduced[::-1]
        reduced = next_polynomial[:-1]
        reduced /= reduced[0]
    return {
        "stable": all(bool(step["stable"]) for step in steps),
        "steps": steps,
    }


def closed_loop_poles(
    plant: Plant, kind: str, gains: Gains, delay: int
) -> np.ndarray:
    coefficients = characteristic_polynomial(plant, kind, gains, delay)
    return np.roots(coefficients)


def augmented_closed_loop_matrix(
    plant: Plant, gains: Gains, delay: int
) -> np.ndarray:
    """Build the zero-reference closed-loop matrix including the input delay line."""
    if delay < 1:
        raise ValueError("The augmented delay model requires delay >= 1")
    q0, q1, q2 = controller_coefficients("PID", gains)
    state_size = plant.ad.shape[0] + delay + 2
    queue_start = plant.ad.shape[0]
    queue_end = queue_start + delay
    error_1_index = queue_end
    error_2_index = queue_end + 1

    def transition(state: np.ndarray) -> np.ndarray:
        x = state[: plant.ad.shape[0]]
        delayed_inputs = state[queue_start:queue_end]
        error_1 = state[error_1_index]
        error_2 = state[error_2_index]
        error = float((-plant.cd @ x).item())
        input_now = delayed_inputs[0] + q0 * \
            error + q1 * error_1 + q2 * error_2
        x_next = plant.ad @ x + plant.bd[:, 0] * delayed_inputs[-1]
        queue_next = np.r_[input_now, delayed_inputs[:-1]]
        return np.r_[x_next, queue_next, error, error_1]

    basis = np.eye(state_size)
    return np.column_stack([transition(basis[:, index]) for index in range(state_size)])


def solve_lyapunov_lmi(matrix: np.ndarray) -> dict[str, Any]:
    """Solve A'PA-P<0, P>0 and return numerical feasibility evidence."""
    dimension = matrix.shape[0]
    p = cp.Variable((dimension, dimension), symmetric=True)
    margin = cp.Variable(nonneg=True)
    identity = np.eye(dimension)
    problem = cp.Problem(
        cp.Maximize(margin),
        [
            cp.trace(p) == 1.0,
            p >> margin * identity,
            matrix.T @ p @ matrix - p << -margin * identity,
        ],
    )
    problem.solve(
        solver=cp.CLARABEL,
        verbose=False,
        tol_gap_abs=1e-10,
        tol_feas=1e-10,
        tol_gap_rel=1e-10,
        max_iter=500,
    )
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or p.value is None:
        raise RuntimeError(f"Lyapunov LMI was not feasible: {problem.status}")

    p_value = (p.value + p.value.T) / 2.0
    p_eigenvalues = np.linalg.eigvalsh(p_value)
    lmi_eigenvalues = np.linalg.eigvalsh(matrix.T @ p_value @ matrix - p_value)
    feasible = bool(p_eigenvalues.min() > 0.0 and lmi_eigenvalues.max() < 0.0)
    if not feasible:
        raise RuntimeError(
            "Solver returned a numerically non-strict Lyapunov certificate")
    return {
        "status": problem.status,
        "solver": "CLARABEL",
        "dimension": dimension,
        "min_eigenvalue_P": float(p_eigenvalues.min()),
        "max_eigenvalue_A_transpose_P_A_minus_P": float(lmi_eigenvalues.max()),
        "feasible": feasible,
    }


def simulate_step(
    plant: Plant,
    kind: str,
    gains: Gains,
    delay: int,
    steps: int = SIMULATION_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a unit reference step from a zero state, with no saturation."""
    x = np.zeros(plant.ad.shape[0])
    delayed_inputs = np.zeros(delay)
    previous_input = 0.0
    error_1 = 0.0
    error_2 = 0.0
    q0, q1, q2 = controller_coefficients(kind, gains)
    output = np.zeros(steps)
    for index in range(steps):
        error = 1.0 - float((plant.cd @ x + plant.dd @ np.zeros((1,))).item())
        if kind == "P":
            input_now = gains.kp * error
        else:
            previous_controller_input = delayed_inputs[0] if delay else previous_input
            input_now = (
                previous_controller_input
                + q0 * error
                + q1 * error_1
                + q2 * error_2
            )
        output[index] = float((plant.cd @ x).item())

        if delay == 0:
            x = plant.ad @ x + plant.bd[:, 0] * input_now
        else:
            x = plant.ad @ x + plant.bd[:, 0] * delayed_inputs[-1]
            delayed_inputs = np.r_[input_now, delayed_inputs[:-1]]
        previous_input = input_now
        error_2, error_1 = error_1, error
    return np.arange(steps) * TS, output


def response_metrics(time: np.ndarray, output: np.ndarray) -> dict[str, float | None]:
    final_error = float(abs(1.0 - output[-1]))
    within_band = np.abs(output - 1.0) <= 0.02
    settling_index: int | None = None
    for index in range(len(output)):
        if np.all(within_band[index:]):
            settling_index = index
            break
    settling_time = None if settling_index is None else float(
        time[settling_index])
    return {
        "overshoot_percent": float(max(0.0, (float(np.max(output)) - 1.0) * 100.0)),
        "settling_time_s": settling_time,
        "steady_state_error": final_error,
    }


def plot_step_comparison(plant: Plant) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for kind in ("P", "I", "PI", "PID"):
        gains = GAIN_SCHEDULE[kind][5]
        time, output = simulate_step(plant, kind, gains, delay=5)
        axis.plot(time, output, label=kind, linewidth=1.8)
    axis.axhline(1.0, color="black", linewidth=0.8,
                 linestyle="--", label="reference")
    axis.set(title="Unit-step response at delay N=5",
             xlabel="Time (s)", ylabel="Output")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_step_comparison.png", dpi=300)
    plt.close(fig)


def plot_delay_sweep(plant: Plant) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(DELAYS)))
    for delay, color in zip(DELAYS, colors):
        time, output = simulate_step(
            plant, "PID", GAIN_SCHEDULE["PID"][delay], delay=delay
        )
        axis.plot(time, output, color=color, label=f"scheduled PID, N={delay}")
    fixed_gains = GAIN_SCHEDULE["PID"][1]
    time, output = simulate_step(plant, "PID", fixed_gains, delay=5)
    axis.plot(
        time,
        output,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="fixed PID, N=5",
    )
    axis.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    axis.set(title="Discrete PID response under transport delay",
             xlabel="Time (s)", ylabel="Output")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_delay_sweep.png", dpi=300)
    plt.close(fig)


def plot_pole_migration(pole_data: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 6.0))
    circle = plt.Circle((0.0, 0.0), 1.0, color="black",
                        fill=False, linestyle="--")
    axis.add_patch(circle)
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(DELAYS)))
    for delay, color in zip(DELAYS, colors):
        poles = pole_data[str(delay)]["PID"]
        axis.scatter(poles.real, poles.imag, s=24,
                     color=color, label=f"N={delay}")
    axis.axhline(0.0, color="gray", linewidth=0.6)
    axis.axvline(0.0, color="gray", linewidth=0.6)
    axis.set(
        title="Scheduled PID closed-loop poles",
        xlabel="Real(z)",
        ylabel="Imaginary(z)",
        xlim=(-1.1, 1.1),
        ylim=(-1.1, 1.1),
        aspect="equal",
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_zplane_poles.png", dpi=300)
    plt.close(fig)


def complex_list(values: np.ndarray) -> list[dict[str, float]]:
    return [
        {"real": float(value.real), "imag": float(value.imag)}
        for value in values
    ]


def run() -> dict[str, Any]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    plant = define_plant()
    summary: dict[str, Any] = {
        "plant": {
            "continuous_transfer_function": "1/(s+1)^2",
            "sample_period_s": TS,
            "discrete_numerator_z_inverse": plant.numerator_z_inverse.tolist(),
            "discrete_denominator_z_inverse": plant.denominator_z_inverse.tolist(),
            "delay_operator": "z^(-N)",
        },
        "controllers": {},
        "lmi_n5_pid": {},
        "lmi_pid_by_delay": {},
        "pole_realization_checks": {},
    }
    all_poles: dict[str, dict[str, np.ndarray]] = {}
    for delay in DELAYS:
        delay_entries: dict[str, Any] = {}
        all_poles[str(delay)] = {}
        for kind in ("P", "I", "PI", "PID"):
            gains = GAIN_SCHEDULE[kind][delay]
            poles = closed_loop_poles(plant, kind, gains, delay)
            maximum_radius = float(np.max(np.abs(poles)))
            if not maximum_radius < 1.0:
                raise RuntimeError(
                    f"{kind} controller is unstable at N={delay}: "
                    f"maximum pole radius={maximum_radius:.8f}"
                )
            polynomial_coefficients = characteristic_polynomial(
                plant, kind, gains, delay
            )
            schur_result = schur_cohn_test(polynomial_coefficients)
            if not schur_result["stable"]:
                raise RuntimeError(
                    f"{kind} controller failed the Schur-Cohn test at N={delay}"
                )
            time, output = simulate_step(plant, kind, gains, delay)
            metrics = response_metrics(time, output)
            all_poles[str(delay)][kind] = poles
            delay_entries[kind] = {
                "gains": asdict(gains),
                "q": list(controller_coefficients(kind, gains)),
                "poles": complex_list(poles),
                "maximum_pole_radius": maximum_radius,
                "pole_radius_margin": 1.0 - maximum_radius,
                "schur_cohn": schur_result,
                **metrics,
            }
        summary["controllers"][str(delay)] = delay_entries

    fixed_pid = GAIN_SCHEDULE["PID"][1]
    fixed_poles = {
        str(delay): closed_loop_poles(plant, "PID", fixed_pid, delay)
        for delay in DELAYS
    }
    summary["fixed_pid_reference_gains"] = asdict(fixed_pid)
    summary["fixed_pid_maximum_pole_radius"] = {
        delay: float(np.max(np.abs(poles))) for delay, poles in fixed_poles.items()
    }
    if any(radius >= 1.0 for radius in summary["fixed_pid_maximum_pole_radius"].values()):
        raise RuntimeError(
            "The fixed-gain PID reference is unstable for one or more delays")

    for delay in DELAYS:
        augmented = augmented_closed_loop_matrix(
            plant, GAIN_SCHEDULE["PID"][delay], delay
        )
        augmented_eigenvalues = np.linalg.eigvals(augmented)
        polynomial_poles = all_poles[str(delay)]["PID"]
        realization_matches = len(augmented_eigenvalues) == len(polynomial_poles) and np.allclose(
            np.sort_complex(augmented_eigenvalues),
            np.sort_complex(polynomial_poles),
            atol=1e-7,
        )
        if not realization_matches:
            raise RuntimeError(
                f"Augmented-matrix eigenvalues do not match PID polynomial poles at N={delay}"
            )
        summary["pole_realization_checks"][str(delay)] = True
        lmi_result = solve_lyapunov_lmi(augmented)
        lmi_result["augmented_matrix_eigenvalues"] = complex_list(
            augmented_eigenvalues
        )
        lmi_result["max_augmented_eigenvalue_magnitude"] = float(
            np.max(np.abs(augmented_eigenvalues))
        )
        summary["lmi_pid_by_delay"][str(delay)] = lmi_result
    summary["lmi_n5_pid"] = summary["lmi_pid_by_delay"]["5"]

    fixed_pid_n5_poles = fixed_poles["5"]
    fixed_time, fixed_output = simulate_step(plant, "PID", fixed_pid, delay=5)
    summary["fixed_pid_at_n5"] = {
        "gains": asdict(fixed_pid),
        "poles": complex_list(fixed_pid_n5_poles),
        "maximum_pole_radius": float(np.max(np.abs(fixed_pid_n5_poles))),
        "pole_radius_margin": float(1.0 - np.max(np.abs(fixed_pid_n5_poles))),
        **response_metrics(fixed_time, fixed_output),
    }

    plot_step_comparison(plant)
    plot_delay_sweep(plant)
    plot_pole_migration(all_poles)
    RESULTS_PATH.write_text(json.dumps(
        summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"LMI N=5 PID: {summary['lmi_n5_pid']['status']} "
        f"(min eig(P)={summary['lmi_n5_pid']['min_eigenvalue_P']:.3e}, "
        "max eig(A'PA-P)="
        f"{summary['lmi_n5_pid']
            ['max_eigenvalue_A_transpose_P_A_minus_P']:.3e})"
    )
    print(f"All scheduled controller poles are strictly inside the unit circle for N=1..5.")
    print(f"Wrote {RESULTS_PATH} and three figures under {ASSETS}/")
    return summary


if __name__ == "__main__":
    run()
