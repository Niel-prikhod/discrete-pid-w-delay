---
title: "Discrete P, I, PI, and PID Control with Transport Delay"
author: "Control Systems Engineering Semester Project"
geometry: margin=25mm
---

# Discrete-Time Control of a Plant with Integer Transport Delay

## 1. Problem Statement and Plant Model

The objective is to design and verify discrete P, I, PI, and PID controllers for a stable second-order plant subject to an integer transport delay. The continuous-time plant is the normalized first-order thermal model cascaded twice,

$$
G_c(s)=\frac{1}{(s+1)^2}, \qquad T_s=0.1\ \mathrm{s}.
$$

Zero-order-hold discretization gives

$$
G(z)=\frac{0.00467884z^{-1}+0.00437708z^{-2}}
{1-1.80967484z^{-1}+0.81873075z^{-2}},
\qquad G_N(z)=G(z)z^{-N}.
$$

The one- and two-sample numerator terms are the sampled plant dynamics; the separately modeled transport delay is $z^{-N}$, with $N\in\{0,1,2,3,4,5\}$. Stability and simulations below focus on $N=1,\ldots,5$. The feedback is unity and negative, the reference is a unit step, and the plant/controller start from rest. The delay is assumed constant during each run and known exactly, or estimated correctly as an integer, by the gain scheduler; if the estimate changes, the corresponding table row is selected. External disturbances are zero, feedback is noise-free, and actuator saturation is not modeled.

## 2. Discrete Controller Synthesis

The controller is represented in incremental form:

$$
u(k)=u(k-1)+q_0e(k)+q_1e(k-1)+q_2e(k-2).
$$

Its transfer function is $G_R(z)=(q_0+q_1z^{-1}+q_2z^{-2})/(1-z^{-1})$. With $T_s=0.1\ \mathrm{s}$, the selected coefficient mappings are:

| Structure | Gains | $(q_0,q_1,q_2)$ |
|---|---|---|
| P | $K_p$ | $(K_p,-K_p,0)$; algebraically reduced to the direct proportional law |
| I | $K_i$ | $(K_iT_s,0,0)$ |
| PI | $K_p,K_i$ | $(K_p+K_iT_s/2,\,-K_p+K_iT_s/2,\,0)$ |
| PID | $K_p,K_i,K_d$ | $(K_p+K_iT_s/2+K_d/T_s,\,-K_p+K_iT_s/2-2K_d/T_s,\,K_d/T_s)$ |

These are respectively direct proportional action, rectangular-sum integration, trapezoidal PI integration, and trapezoidal PI plus backward-difference derivative action. The simulation applies the P law directly as $u(k)=K_pe(k)$; its listed $(q_0,q_1,q_2)$ only express the equivalent incremental relation when the previous command and error states are consistent. The characteristic polynomial is reduced to the direct $K_p$ form, avoiding a canceled pole at $z=1$ and a redundant integrator state.

### Numerical tuning and gain scheduling

The gains are calculated offline, separately for each controller structure and integer delay, by `scipy.optimize.differential_evolution` with seed 42, population multiplier 10, tolerance $10^{-6}$ and a limit of 150 iterations, followed by local polishing. The optimizer varies only the gains applicable to the structure, within $0.05\leq K_p\leq4$, $0.01\leq K_i\leq2$, and $0.01\leq K_d\leq2$. It minimizes the reference-tracking objective

$$
J=T_s\sum_{k=0}^{199}|1-y(k)|+
20\left[\max\left(0,\max_{0\leq k<200}y(k)-1\right)\right]^2,
$$

computed on a 20-second unit-step response. Candidate gains must have maximum closed-loop pole radius below $0.99$; infeasible candidates receive a large penalty. Because the $N=1$ PID also serves as the fixed-gain baseline, its pole constraint is checked at every $N=1,\ldots,5$, while its objective still uses the $N=1$ response. All other candidates are checked at their own delay. The optimization is numerical and bounded; it does not establish a global optimum. It does not penalize control effort, actuator limits, disturbances, or noise, and the unfiltered derivative produces a potentially large initial control command.

The resulting delay-regime gain schedule is a lookup table, selected using the known/estimated integer delay. The delay estimate is assumed available before tuning the loop; there is no interpolation or online gain adaptation between rows. Entries below are rounded $(K_p,K_i,K_d)$; the simulation uses the unrounded optimized values and calculates $(q_0,q_1,q_2)$ from them. Some gains reach the chosen bounds.

| $N$ | P | I | PI | PID |
|---:|---:|---:|---:|---:|
| 1 | (4.0000, 0, 0) | (0, 0.2949, 0) | (1.7179, 0.7390, 0) | (3.9861, 2.0000, 1.8792) |
| 2 | (3.5313, 0, 0) | (0, 0.2797, 0) | (1.4614, 0.6428, 0) | (3.2933, 1.6691, 1.8016) |
| 3 | (3.0137, 0, 0) | (0, 0.2660, 0) | (1.2695, 0.5721, 0) | (2.5806, 1.2867, 1.4630) |
| 4 | (2.7042, 0, 0) | (0, 0.2536, 0) | (1.1283, 0.5149, 0) | (2.1528, 1.0661, 1.2817) |
| 5 | (2.4468, 0, 0) | (0, 0.2427, 0) | (1.0595, 0.4710, 0) | (1.8498, 0.9094, 1.1410) |

`controller.py` implements the stateful control law independently of the plant; `simulation.py` performs tuning, analysis and comparison. Create the locked environment with `uv sync`, then reproduce the results and figures with `uv run python simulation.py`.

## 3. Theoretical Stability Proof

For negative unity feedback, each characteristic polynomial is obtained from

$$
1+G_R(z)G(z)z^{-N}=0.
$$

For I, PI, and PID, write $Q(z^{-1})=q_0+q_1z^{-1}+q_2z^{-2}$. The closed-loop polynomial in $z^{-1}$ is

$$
\Delta_N(z^{-1})=
A(z^{-1})(1-z^{-1})+B(z^{-1})Q(z^{-1})z^{-N},
$$

where $A=1-1.80967484z^{-1}+0.81873075z^{-2}$ and $B=0.00467884z^{-1}+0.00437708z^{-2}$. For P, the reduced polynomial is $A+B K_pz^{-N}$. The zeros of the resulting polynomial in $z$ are the closed-loop poles.

For a real polynomial $p(z)=a_0z^m+a_1z^{m-1}+\cdots+a_m$, the Schur-Cohn recursion requires $|a_m|<|a_0|$ and that the reduced polynomial

$$
p_1(z)=\frac{a_0p(z)-a_mp^\#(z)}{z}, \qquad
p^\#(z)=z^mp(1/z),
$$

also be Schur stable. Repeating the test down to degree zero is necessary and sufficient for all roots to lie strictly inside the unit circle; this is the recursive form of the Jury/Schur stability test. The computations evaluate the equivalent roots directly and require $\max_i|z_i|<1$.

The implementation applies this recursion numerically to every scheduled polynomial. At each reduced degree it records the reflection ratio $r_m=a_m/a_0$, then forms the next polynomial using the expression above; stability requires $|r_m|<1$ at every step. For each controller and delay, the largest absolute reflection ratio is:

| Delay $N$ | P | I | PI | PID |
|---:|---:|---:|---:|---:|
| 1 | 0.976195 | 0.999262 | 0.997887 | 0.996766 |
| 2 | 0.979515 | 0.999302 | 0.998062 | 0.996905 |
| 3 | 0.982781 | 0.999338 | 0.998215 | 0.997210 |
| 4 | 0.984994 | 0.999372 | 0.998356 | 0.997439 |
| 5 | 0.986789 | 0.999401 | 0.998454 | 0.997641 |

As an explicit $N=5$ recursion trace, the reflection ratios in descending degree order are:

| Controller | $r_m$ values, recursion from highest degree to degree 1 |
|---|---|
| P | $0.010710,\ 0.030833,\ 0.047083,\ 0.060152,\ 0.070713,\ 0.907953,\ -0.986789$ |
| I | $0.000106,\ 0.000412,\ 0.000879,\ 0.001472,\ -0.816568,\ 0.996202,\ -0.999401$ |
| PI | $-0.004534,\ -0.012846,\ -0.019113,\ -0.023656,\ -0.026759,\ -0.848893,\ 0.993492,\ -0.998454$ |
| PID | $0.049942,\ 0.086140,\ 0.054082,\ 0.027771,\ 0.006373,\ -0.010687,\ -0.857175,\ 0.987572,\ -0.997641$ |

All reported ratios have magnitude strictly below one, consistent with the pole-radius results. The polynomial arrays are stored as coefficients in ascending powers of $z^{-1}$; clearing denominators makes this ordering the descending-power coefficient sequence in $z$ expected by `numpy.roots`, so no reversal is applied before finding roots.

### Scheduled closed-loop pole radii

| Delay $N$ | P | I | PI | PID |
|---:|---:|---:|---:|---:|
| 1 | 0.934931 | 0.974288 | 0.965751 | 0.914454 |
| 2 | 0.949599 | 0.975071 | 0.965886 | 0.925551 |
| 3 | 0.958228 | 0.975811 | 0.965844 | 0.930243 |
| 4 | 0.965433 | 0.976511 | 0.966187 | 0.935530 |
| 5 | 0.970507 | 0.977195 | 0.967743 | 0.939316 |

The companion summary `results.json` contains each pole, maximum pole radius, and radial stability margin $1-\max_i|z_i|$. The scheduled designs are required by the script to have positive radial margin for every $N=1,\ldots,5$; execution aborts rather than emitting a stability claim if this check fails.

### Augmented-state Lyapunov certificate at $N=5$

The delay line is included explicitly in the state. Let

$$
\xi(k)=\begin{bmatrix}x(k)^T & u(k-1)&\cdots&u(k-5)&e(k-1)&e(k-2)\end{bmatrix}^T.
$$

With the PID recursion and zero reference, the state update is $\xi(k+1)=A_{\mathrm{aug}}\xi(k)$. A quadratic Lyapunov function $V(\xi)=\xi^TP\xi$ proves asymptotic stability if

$$
P=P^T\succ0,\qquad A_{\mathrm{aug}}^TPA_{\mathrm{aug}}-P\prec0.
$$

This augmented-delay Lyapunov formulation captures the finite memory of the transport delay (often described as a Lyapunov-Krasovskii construction). CVXPY/CLARABEL solves the LMI for the scheduled PID at each delay; the finite-dimensional delay realization changes dimension with $N$. All five solves returned **optimal**, and for each, $\lambda_{\min}(P)>0$ and $\lambda_{\max}(A_{\mathrm{aug}}^TPA_{\mathrm{aug}}-P)<0$:

| $N$ | Matrix dimension | Solver status | $\lambda_{\min}(P)$ | $\lambda_{\max}(A_{\mathrm{aug}}^TPA_{\mathrm{aug}}-P)$ | Maximum augmented pole magnitude |
|---:|---:|---|---:|---:|---:|
| 1 | 5 | optimal | $1.5330\times10^{-4}$ | $-1.5169\times10^{-4}$ | 0.914454 |
| 2 | 6 | optimal | $8.1216\times10^{-5}$ | $-7.9806\times10^{-5}$ | 0.925551 |
| 3 | 7 | optimal | $6.9222\times10^{-5}$ | $-6.8079\times10^{-5}$ | 0.930243 |
| 4 | 8 | optimal | $5.9109\times10^{-5}$ | $-5.8145\times10^{-5}$ | 0.935530 |
| 5 | 9 | optimal | $5.2786\times10^{-5}$ | $-5.1953\times10^{-5}$ | 0.939316 |

For every delay, the script also asserts that the eigenvalues of $A_{\mathrm{aug}}$ match the roots of the independently formed characteristic polynomial. The detailed solver results, numerical signs, and eigenvalues are included in `results.json`.

## 4. Simulation Results

Responses are simulated over $40$ seconds. Overshoot is relative to the unit reference; settling time is the first sample after which the response stays within a $\pm2\%$ band for the remainder of the run. The steady-state error is the absolute final tracking error. The fixed-gain comparison holds the $N=1$ PID gains $(3.9861,2.0000,1.8792)$ at $N=5$, while the scheduled comparison uses the $N=5$ gains $(1.8498,0.9094,1.1410)$; these values are rounded.

![Unit-step response comparing the four controller structures at N=5](assets/fig_step_comparison.png)

![Scheduled PID responses for N=1 through N=5, with fixed N=1 gains versus scheduled N=5 gains compared at N=5](assets/fig_delay_sweep.png)

![Scheduled PID closed-loop pole migration as delay increases](assets/fig_zplane_poles.png)

### Performance at $N=5$

| Controller | Settling time (s) | Overshoot (%) | Steady-state error | Maximum pole radius $|\lambda|_{\max}$ |
|---|---:|---:|---:|---:|
| P | Not reached within 40 s | 10.18 | 0.290121 | 0.970507 |
| I | 16.2 | 9.88 | 0.000101 | 0.977195 |
| PI | 9.0 | 4.32 | $<10^{-6}$ | 0.967743 |
| PID | 3.9 | 1.88 | $<10^{-6}$ | 0.939316 |
| PID, fixed $N=1$ gains | 13.5 | 71.06 | 0.000005 | 0.970701 |

The fixed-gain row uses the same $N=5$ plant and delay as the scheduled PID row, changing only the PID gains. The machine-readable summary includes these fixed-gain metrics, response metrics for every scheduled controller across $N=1,\ldots,5$, Schur-Cohn recursion quantities, augmented-realization checks, and Lyapunov results.

## 5. Conclusion

For this sampled second-order plant and the computed gain table, the designs are strictly Schur stable across integer delays $N=1,\ldots,5$. The discrete poles and separate fixed-delay Lyapunov certificates provide complementary checks of asymptotic stability; they do not prove stability under arbitrary switching between delays or gains. Integral action removes the proportional controller's steady-state offset. At $N=5$, the scheduled PID's lower overshoot and settling time than the fixed PID demonstrate the effect of this tuning and scheduling on this model, not a general guarantee; the unfiltered derivative and unconstrained control effort limit real-world applicability.
