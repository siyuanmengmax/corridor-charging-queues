"""
Batch-arrival M/M/c queueing model with PWL epigraph for EV charging stations.

Physical context
---------------
At each 5-minute MPC step, up to n vehicles are assigned to a station that
has c_free free chargers.  Vehicles arrive in a BATCH at the start of the
interval (not as a continuous Poisson stream), so the standard steady-state
M/M/c formula (Erlang-C / (cμ-λ)) is inapplicable — charging takes ~22 min
while ΔT = 5 min, making the steady-state arrival rate always super-critical.

Batch-arrival formula (derived from M/M/c transient analysis, §1 below):
    g_s(n; c_free, μ) = overflow·(overflow+1) / (2·n·max(1,c_free)·μ)
    where overflow = max(0, n − c_free)

Properties:
  • g_s = 0 when n ≤ c_free  (all served immediately)
  • g_s grows as O((n − c_free)²) when n > c_free  (convex overload penalty)
  • Convex in n → PWL epigraph via tangent lines (mmc_queue.md §3.1) is valid

At each MPC step c_free is read from the real-time station state (dynamic
capacity); the PWL is rebuilt every step so the epigraph reflects actual
charger availability.

Functions exported
------------------
erlang_c, expected_wait_ss  — steady-state M/M/c (kept for paper analysis /
                               comparison with continuous-arrival scenarios).
batch_wait                  — transient batch-arrival expected wait  (primary).
compute_wq_curve            — g_s(n) table for n = 0..n_max.
fit_pwl_epigraph            — K tangent lines for MILP injection.
StationQueueModel           — per-station convenience wrapper.
pwl_approximation_error     — diagnostic.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# §0: Steady-state M/M/c (kept for completeness / paper analysis)
# ---------------------------------------------------------------------------

def erlang_c(c: int, a: float) -> float:
    """
    Erlang-C P(wait) for M/M/c in steady state.
    Numerically stable iterative formulation (avoids computing a^c/c! directly).
    Returns 1.0 when offered load ρ = a/c >= 1 (unstable).
    """
    if c < 1:
        raise ValueError(f"c >= 1 required, got {c}")
    if a <= 0:
        return 0.0
    rho = a / c
    if rho >= 1.0:
        return 1.0
    # Build Σ_{k=0}^{c-1} a^k/k! iteratively
    p = 1.0
    sum_p = 1.0
    for k in range(1, c):
        p = p * a / k
        sum_p += p
    p_c = p * a / c          # a^c / c!
    num = p_c / (1.0 - rho)  # Erlang-C numerator
    return num / (sum_p + num)


def expected_wait_ss(c: int, lam: float, mu: float) -> float:
    """
    Steady-state M/M/c expected queue wait Wq (minutes).
    Valid only when ρ = λ/(c·μ) < 1.
    Returns a large sentinel for near-saturated queues.
    """
    if lam <= 0:
        return 0.0
    rho = lam / (c * mu)
    if rho >= 0.99:
        return 999.0
    ec = erlang_c(c, lam / mu)
    denom = c * mu - lam
    return (ec / denom) if denom > 1e-9 else 999.0


# ---------------------------------------------------------------------------
# §1: Batch-arrival expected wait (primary model for MPC context)
# ---------------------------------------------------------------------------

def batch_wait(
    n: int,
    c_free: int,
    mu: float,
    gamma_arrival: float = 0.0,
    lam_hat_fraction: float = 0.2,
) -> float:
    """
    Expected mean queue wait (minutes) when n vehicles arrive simultaneously
    to a station with c_free currently free chargers, each with service rate μ.

    Derivation (exponential service times, FCFS):
      Vehicles 1..c_free: served immediately (wait = 0).
      Vehicle c_free+j: waits for j completions; each completion takes
        E[min of k remaining exp(μ) services] = 1/(k·μ) so
        Wq(c_free+j) = Σ_{k=c_free, c_free-1, ..., 1} 1/(k·μ)  [approx: j/(c_free·μ)]
      Mean wait = (1/n) · Σ_{j=1}^{overflow} j / (c_free · μ)
                = overflow · (overflow+1) / (2 · n · c_free · μ)

    With arrival-time uncertainty buffer (D5):
      Use c_free_eff = max(1, c_free - Γ_a · n · lam_hat_fraction)
      to conservatively assume more overflow.

    Parameters
    ----------
    n              : number of vehicles assigned this interval
    c_free         : currently free chargers at this station
    mu             : service rate per charger (vehicles / minute)
    gamma_arrival  : Γ_a arrival-uncertainty budget (D5 conservative buffer)
    lam_hat_fraction : fraction of n that is uncertain (default ±20%)
    """
    if n <= 0 or mu <= 0:
        return 0.0

    # D5 conservative buffer: reduce effective free chargers
    c_eff = max(1, c_free - int(gamma_arrival * n * lam_hat_fraction))
    overflow = max(0, n - c_eff)
    if overflow == 0:
        return 0.0

    return overflow * (overflow + 1) / (2.0 * n * c_eff * mu)


# ---------------------------------------------------------------------------
# §2: Wq curve and PWL epigraph
# ---------------------------------------------------------------------------

def compute_wq_curve(
    c_free: int,
    mu: float,
    n_max: int,
    gamma_arrival: float = 0.0,
    lam_hat_fraction: float = 0.2,
) -> np.ndarray:
    """
    Compute g_s(n) for n = 0, 1, ..., n_max using the batch-arrival formula.

    Parameters
    ----------
    c_free         : free chargers at the station right now
    mu             : service rate per charger (vehicles / minute)
    n_max          : maximum vehicles to consider
    gamma_arrival  : Γ_a for D5 arrival buffer
    lam_hat_fraction : ±fraction of n that is uncertain
    """
    wq = np.zeros(n_max + 1)
    for n in range(1, n_max + 1):
        wq[n] = batch_wait(n, c_free, mu, gamma_arrival, lam_hat_fraction)
    return wq


def fit_pwl_epigraph(
    wq_values: np.ndarray,
    n_breakpoints: int = 6,
) -> "PWLEpigraph":
    """
    Fit convex PWL epigraph to a Wq(n) curve via tangent lines.

    For a convex non-decreasing g_s:
      W_s ≥ slope_k · n_s + intercept_k  for k = 1..K
    Minimising W_s drives it to the upper envelope of tangent lines ≈ g_s.

    Breakpoints use denser spacing near the high-load end (where the curve
    rises steeply) to minimise approximation error where it matters most.

    Returns
    -------
    PWLEpigraph with K tangent lines.
    """
    n_max = len(wq_values) - 1
    if n_max == 0 or n_breakpoints < 1:
        return PWLEpigraph(tangents=[(0.0, 0.0)], n_max=n_max)

    # Breakpoint spacing: denser near the tail (quadratic distribution)
    t = np.linspace(0.0, 1.0, n_breakpoints) ** 1.5
    indices = np.unique(np.clip((t * n_max).astype(int), 0, n_max))

    tangents: List[Tuple[float, float]] = []
    for idx in indices:
        n0 = float(idx)
        w0 = float(wq_values[idx])

        # Numerical derivative (central difference, one-sided at boundaries)
        if idx == 0:
            slope = float(wq_values[min(1, n_max)] - wq_values[0])
        elif idx == n_max:
            slope = float(wq_values[n_max] - wq_values[n_max - 1])
        else:
            slope = float(wq_values[idx + 1] - wq_values[idx - 1]) / 2.0

        slope = max(0.0, slope)  # non-decreasing: clip numerical noise
        intercept = w0 - slope * n0
        tangents.append((slope, intercept))

    return PWLEpigraph(tangents=tangents, n_max=n_max)


@dataclass
class PWLEpigraph:
    """
    K tangent-line epigraph of a convex function g(n).

    Constraint injected into MILP:  W_s ≥ slope_k · n_s + intercept_k  ∀k
    """
    tangents: List[Tuple[float, float]] = field(default_factory=list)
    n_max: int = 0

    def evaluate(self, n: float) -> float:
        """Evaluate the PWL approximation at integer n."""
        if not self.tangents:
            return 0.0
        return max(0.0, max(m * n + b for m, b in self.tangents))


# ---------------------------------------------------------------------------
# §3: Per-station queue model (convenience wrapper)
# ---------------------------------------------------------------------------

@dataclass
class StationQueueModel:
    """
    M/M/c batch-arrival queue model for one charging station.

    Rebuilt each MPC step with the current c_free (free charger count)
    so that the PWL epigraph reflects real-time occupancy.

    Parameters
    ----------
    station_id     : str
    n_chargers     : int    Total chargers c = N_s.
    power_kw       : float  Rated power per charger (kW).
    battery_kwh    : float  Representative battery capacity for mean service time.
    target_soc     : float  Charging target SOC (default 0.8).
    delta_t        : float  Control interval (minutes).
    n_breakpoints  : int    K tangent lines in PWL.
    gamma_arrival  : float  Arrival-uncertainty budget Γ_a for D5 buffer.
    """
    station_id: str
    n_chargers: int
    power_kw: float
    battery_kwh: float = 60.0
    target_soc: float = 0.80
    delta_t: float = 5.0
    n_breakpoints: int = 6
    gamma_arrival: float = 0.0

    def __post_init__(self):
        # μ: service rate per charger (vehicles per minute)
        # Mean service time = time to charge from 0 to target_soc
        mean_service_min = self.target_soc * self.battery_kwh / self.power_kw * 60.0
        self.mu = 1.0 / mean_service_min

    def build_pwl(
        self,
        c_free: int | None = None,
        gamma_arrival: float | None = None,
    ) -> PWLEpigraph:
        """
        Build PWL epigraph for the current station state.

        Parameters
        ----------
        c_free        : currently free chargers (default: all chargers are free)
        gamma_arrival : Γ_a override for this step
        """
        c = c_free if c_free is not None else self.n_chargers
        ga = gamma_arrival if gamma_arrival is not None else self.gamma_arrival
        n_max = min(3 * self.n_chargers + 2, 20)  # realistic upper bound on assignments
        wq = compute_wq_curve(
            c_free=c,
            mu=self.mu,
            n_max=n_max,
            gamma_arrival=ga,
        )
        return fit_pwl_epigraph(wq, n_breakpoints=self.n_breakpoints)

    def nominal_charge_time(self, current_soc: float, battery_kwh: float | None = None) -> float:
        """
        Station-specific nominal charging duration d̄_{i,s} (minutes, D4).

        d̄_{i,s} = (target_soc − soc) · battery_kwh / power_kw · 60
        """
        batt = battery_kwh if battery_kwh is not None else self.battery_kwh
        return max(0.0, self.target_soc - current_soc) * batt / self.power_kw * 60.0

    def charge_uncertainty(self, nominal: float, epsilon: float = 0.20) -> float:
        """Uncertainty d̂_{i,s} = ε · d̄_{i,s}."""
        return epsilon * nominal


# ---------------------------------------------------------------------------
# §4: Diagnostics
# ---------------------------------------------------------------------------

def pwl_approximation_error(
    wq_values: np.ndarray,
    pwl: PWLEpigraph,
) -> dict:
    """Max and mean absolute error of PWL vs true Wq curve."""
    errs = np.array([abs(pwl.evaluate(n) - wq_values[n]) for n in range(len(wq_values))])
    wq_range = wq_values.max() - wq_values.min()
    return {
        "max_abs": float(errs.max()),
        "mean_abs": float(errs.mean()),
        "max_rel": float(errs.max() / wq_range) if wq_range > 0 else 0.0,
    }
