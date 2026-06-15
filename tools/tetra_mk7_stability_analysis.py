#!/usr/bin/env python3
"""VSPAERO derivative summary for the teTra Mk-7 EM2 Gazebo model."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from pathlib import Path


MODEL_DIR = Path("models/teTra_mk-7_EM2")
STAB_FILE = MODEL_DIR / "teTra_mk-7_EM2.stab"


@dataclass(frozen=True)
class Aircraft:
    mass_kg: float = 1560.0
    wing_area_m2: float = 12.0
    wing_span_m: float = 8.0
    mac_m: float = 1.5
    horizontal_tail_area_m2: float = 2.0777
    horizontal_tail_span_m: float = 2.900
    horizontal_tail_z_m: float = 0.960
    pusher_prop_z_m: float = 0.450
    pusher_prop_diameter_m: float = 1.800
    pusher_prop_pitch_deg: float = 24.0
    pusher_motor_constant_n_per_radps2: float = 0.0443472
    pusher_moment_constant_m: float = 0.169572
    pusher_max_omega_radps: float = 251.327412
    pusher_cruise_speed_mps: float = 72.0222
    pusher_cruise_thrust_n: float = 1912.3
    pusher_cruise_shaft_power_w: float = 172160.0
    ixx_kgm2: float = 6148.0
    iyy_kgm2: float = 6463.0
    izz_kgm2: float = 11850.0
    ixz_kgm2: float = -66.0


@dataclass(frozen=True)
class StabDerivatives:
    rows: dict[str, list[float]]
    static_margin: float | None
    neutral_point_x: float | None

    def value(self, coef: str, column: str) -> float:
        columns = {
            "base": 0,
            "alpha": 1,
            "beta": 2,
            "p": 3,
            "q": 4,
            "r": 5,
            "mach": 6,
            "u": 7,
        }
        return self.rows[coef][columns[column]]


def read_stab(path: Path) -> StabDerivatives:
    rows: dict[str, list[float]] = {}
    static_margin: float | None = None
    neutral_point_x: float | None = None

    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[0] in {"CL", "CD", "CS", "CMl", "CMm", "CMn"}:
            rows[parts[0]] = [float(value) for value in parts[1:9]]
        elif len(parts) >= 2 and parts[0] == "SM":
            static_margin = float(parts[1])
        elif len(parts) >= 2 and parts[0] == "X_np":
            neutral_point_x = float(parts[1])

    missing = {"CL", "CD", "CS", "CMl", "CMm", "CMn"} - rows.keys()
    if missing:
        raise RuntimeError(f"Missing derivative rows in {path}: {sorted(missing)}")

    return StabDerivatives(rows=rows, static_margin=static_margin, neutral_point_x=neutral_point_x)


def eig2(a11: float, a12: float, a21: float, a22: float) -> tuple[complex, complex]:
    trace = a11 + a22
    det = a11 * a22 - a12 * a21
    root = cmath.sqrt(trace * trace - 4.0 * det)
    return (0.5 * (trace + root), 0.5 * (trace - root))


def mode_summary(lam: complex) -> str:
    wn = abs(lam)
    if wn <= 1e-12:
        return f"{lam.real:+.4f}{lam.imag:+.4f}j"

    zeta = -lam.real / wn
    freq_hz = abs(lam.imag) / (2.0 * math.pi)
    return f"{lam.real:+.4f}{lam.imag:+.4f}j  wn={wn:.3f}/s  zeta={zeta:.3f}  f={freq_hz:.3f}Hz"


def longitudinal_modes(ac: Aircraft, d: StabDerivatives, speed_mps: float) -> tuple[complex, complex]:
    rho = 1.2041
    qbar = 0.5 * rho * speed_mps**2

    z_alpha = -qbar * ac.wing_area_m2 * d.value("CL", "alpha") / (ac.mass_kg * speed_mps)
    m_alpha = qbar * ac.wing_area_m2 * ac.mac_m * d.value("CMm", "alpha") / ac.iyy_kgm2
    m_q = (
        qbar
        * ac.wing_area_m2
        * ac.mac_m**2
        * d.value("CMm", "q")
        / (2.0 * speed_mps * ac.iyy_kgm2)
    )
    return eig2(z_alpha, 1.0, m_alpha, m_q)


def directional_beta_r_modes(
    ac: Aircraft, d: StabDerivatives, speed_mps: float
) -> tuple[complex, complex]:
    rho = 1.2041
    qbar = 0.5 * rho * speed_mps**2
    iz_eff = ac.izz_kgm2 - ac.ixz_kgm2**2 / ac.ixx_kgm2

    y_beta = qbar * ac.wing_area_m2 * d.value("CS", "beta") / (ac.mass_kg * speed_mps)
    n_beta = qbar * ac.wing_area_m2 * ac.wing_span_m * d.value("CMn", "beta") / iz_eff
    n_r = (
        qbar
        * ac.wing_area_m2
        * ac.wing_span_m**2
        * d.value("CMn", "r")
        / (2.0 * speed_mps * iz_eff)
    )
    return eig2(y_beta, -1.0, n_beta, n_r)


def roll_yaw_rate_modes(ac: Aircraft, d: StabDerivatives, speed_mps: float) -> tuple[complex, complex]:
    rho = 1.2041
    qbar = 0.5 * rho * speed_mps**2
    scale = qbar * ac.wing_area_m2 * ac.wing_span_m**2 / (2.0 * speed_mps)

    l_p = scale * d.value("CMl", "p")
    l_r = scale * d.value("CMl", "r")
    n_p = scale * d.value("CMn", "p")
    n_r = scale * d.value("CMn", "r")

    det_i = ac.ixx_kgm2 * ac.izz_kgm2 - ac.ixz_kgm2**2
    a11 = (ac.izz_kgm2 * l_p + ac.ixz_kgm2 * n_p) / det_i
    a12 = (ac.izz_kgm2 * l_r + ac.ixz_kgm2 * n_r) / det_i
    a21 = (ac.ixz_kgm2 * l_p + ac.ixx_kgm2 * n_p) / det_i
    a22 = (ac.ixz_kgm2 * l_r + ac.ixx_kgm2 * n_r) / det_i
    return eig2(a11, a12, a21, a22)


def pusher_thrust(ac: Aircraft, throttle: float) -> float:
    omega = ac.pusher_max_omega_radps * throttle
    return ac.pusher_motor_constant_n_per_radps2 * omega**2


def propwash_dynamic_pressure_ratio(ac: Aircraft, speed_mps: float, throttle: float) -> float:
    rho = 1.2041
    thrust = pusher_thrust(ac, throttle)
    disk_area = math.pi * (0.5 * ac.pusher_prop_diameter_m) ** 2
    induced_velocity = -0.5 * speed_mps + math.sqrt((0.5 * speed_mps) ** 2 + thrust / (2.0 * rho * disk_area))
    wake_speed = speed_mps + 2.0 * induced_velocity
    return (wake_speed / speed_mps) ** 2


def horizontal_tail_wash_fraction(ac: Aircraft) -> float:
    radius = 0.5 * ac.pusher_prop_diameter_m
    dz = abs(ac.horizontal_tail_z_m - ac.pusher_prop_z_m)
    if dz >= radius:
        return 0.0

    washed_width = 2.0 * math.sqrt(radius**2 - dz**2)
    return min(1.0, washed_width / ac.horizontal_tail_span_m)


def propwash_corrected_longitudinal(
    ac: Aircraft, d: StabDerivatives, speed_mps: float, throttle: float
) -> dict[str, float]:
    """First-order pusher slipstream correction for the horizontal tail.

    This is an upper-bound sensitivity check, not a replacement for a coupled
    propeller VSPAERO run.  It applies actuator-disk momentum theory to the
    part of the horizontal tail inside the propeller disk and scales the tail's
    pitch derivatives only.
    """

    wash_fraction = horizontal_tail_wash_fraction(ac)
    q_ratio = propwash_dynamic_pressure_ratio(ac, speed_mps, throttle)
    tail_multiplier = 1.0 + wash_fraction * (q_ratio - 1.0)
    tail_area_ratio = ac.horizontal_tail_area_m2 / ac.wing_area_m2

    # Estimate the horizontal-tail lift share from the current geometry. Pitch
    # stiffness and pitch-rate damping are treated as tail-dominated at this
    # fidelity; replace this with component VSPAERO loads once available.
    tail_cl_alpha = 4.7 * tail_area_ratio
    base_cl_alpha = d.value("CL", "alpha")
    corrected_cl_alpha = base_cl_alpha + (tail_multiplier - 1.0) * tail_cl_alpha

    corrected_cm_alpha = d.value("CMm", "alpha") * tail_multiplier
    corrected_cm_q = d.value("CMm", "q") * tail_multiplier
    corrected_sm = -corrected_cm_alpha / corrected_cl_alpha

    return {
        "thrust_n": pusher_thrust(ac, throttle),
        "wash_fraction": wash_fraction,
        "q_ratio": q_ratio,
        "tail_multiplier": tail_multiplier,
        "CLa": corrected_cl_alpha,
        "Cema": corrected_cm_alpha,
        "Cemq": corrected_cm_q,
        "SM": corrected_sm,
    }


def print_advanced_lift_drag_values(d: StabDerivatives) -> None:
    print("AdvancedLiftDrag values from VSPAERO .stab")
    print(f"CLa={d.value('CL', 'alpha'):.7f}")
    print(f"CD0={d.value('CD', 'base'):.7f}")
    print(f"Cem0={d.value('CMm', 'base'):.7f}")
    print(f"Cema={d.value('CMm', 'alpha'):.7f}")
    print(f"CYb={d.value('CS', 'beta'):.7f}")
    print(f"Cellb={d.value('CMl', 'beta'):.7f}")
    print(f"Cenb={d.value('CMn', 'beta'):.7f}")
    print(f"CLq={d.value('CL', 'q'):.7f}")
    print(f"Cemq={d.value('CMm', 'q'):.7f}")
    print(f"CYr={d.value('CS', 'r'):.7f}")
    print(f"Cenr={d.value('CMn', 'r'):.7f}")
    if d.static_margin is not None and d.neutral_point_x is not None:
        print(f"SM={d.static_margin:.7f}, X_np={d.neutral_point_x:.7f}")
    print()


def print_propwash_summary(ac: Aircraft, d: StabDerivatives) -> None:
    print("Pusher propwash upper-bound estimate on horizontal tail")
    print(
        f"Prop source: HK-UK3-3B 175 cm / {ac.pusher_prop_pitch_deg:.0f} deg, "
        "scaled to 180 cm static data."
    )
    print(
        f"Static full throttle: {pusher_thrust(ac, 1.0):.0f} N at "
        f"{ac.pusher_max_omega_radps * 60.0 / (2.0 * math.pi):.0f} rpm"
    )
    print(
        f"Cruise target: {ac.pusher_cruise_speed_mps:.1f} m/s, "
        f"{ac.pusher_cruise_thrust_n:.0f} N, "
        f"{ac.pusher_cruise_shaft_power_w / 1000.0:.0f} kW shaft at eta=0.80"
    )
    print("throttle  V[m/s]  thrust[N]  tail_q_mult  CLa       Cema       Cemq        SM")
    for throttle in (0.5, 0.75, 1.0):
        for speed in (40.0, 50.0, 60.0, ac.pusher_cruise_speed_mps):
            p = propwash_corrected_longitudinal(ac, d, speed, throttle)
            print(
                f"{throttle:7.2f} {speed:7.0f} {p['thrust_n']:10.0f} "
                f"{p['tail_multiplier']:11.3f} {p['CLa']:8.3f} "
                f"{p['Cema']:10.3f} {p['Cemq']:10.3f} {p['SM']:9.3f}"
            )
    print(
        "Assumption: the 1.8 m pusher disk washes "
        f"{100.0 * horizontal_tail_wash_fraction(ac):.1f}% of the horizontal-tail span."
    )
    print("These rows are sensitivity estimates; the SDF uses the VSPAERO .stab values above.")
    print()


def main() -> None:
    ac = Aircraft()
    d = read_stab(STAB_FILE)

    print("teTra Mk-7 EM2 VSPAERO stability analysis")
    print(f"Source: {STAB_FILE}")
    print(f"Reference: S={ac.wing_area_m2:.3f} m^2, b={ac.wing_span_m:.3f} m, MAC={ac.mac_m:.3f} m")
    print(f"Inertia: Ixx={ac.ixx_kgm2:.1f}, Iyy={ac.iyy_kgm2:.1f}, Izz={ac.izz_kgm2:.1f}, Ixz={ac.ixz_kgm2:.1f}")
    print()
    print_advanced_lift_drag_values(d)
    print_propwash_summary(ac, d)

    for speed in (40.0, 50.0, 60.0, ac.pusher_cruise_speed_mps):
        print(f"V={speed:.0f} m/s")
        roots = longitudinal_modes(ac, d, speed)
        print(f"  longitudinal alpha/q  {mode_summary(roots[0])}")
        roots = directional_beta_r_modes(ac, d, speed)
        print(f"  directional beta/r    {mode_summary(roots[0])}")
        roots = roll_yaw_rate_modes(ac, d, speed)
        print(f"  roll-yaw p/r damping  {mode_summary(roots[0])}")
        print()

    print("Notes:")
    print("- Values are VSPAERO VLM derivatives with the current pusher actuator disk settings.")
    print("- Control-surface derivatives are still approximate in SDF; run AVL/VSPAERO control cases to replace them.")
    print("- Bodies, booms, landing gear, detailed rotating-blade effects, and nonlinear stall are not included.")


if __name__ == "__main__":
    main()
