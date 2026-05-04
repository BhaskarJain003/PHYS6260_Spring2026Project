"""
PHYS 6260 - Term Project
White Dwarf Mass-Radius Relation via the Chandrasekhar Equation of State

Everything is in this one file on purpose (easier to read for a class project).

Usage:
    uv run python main.py

Outputs:
    mass_radius.png  -- plot of the M-R curve with observational data
    mass_radius.csv  -- numerical M-R table (mass, radius in R_sun, radius in km)
"""

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving figures from a script
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# 1. Physical constants & Chandrasekhar EOS (CGS)
# =============================================================================
# The EOS uses dimensionless Fermi momentum x = p_F / (m_e c):
#   rho = B * x^3
#   P   = A * [ x(2x^2 - 3)sqrt(1 + x^2) + 3 sinh^{-1}(x) ]

G = 6.67430e-8  # gravitational constant      [cm^3 g^-1 s^-2]
c = 2.99792e10  # speed of light                [cm/s]
hbar = 1.05457e-27  # reduced Planck constant    [erg s]
m_e = 9.10938e-28  # electron mass               [g]
m_H = 1.67262e-24  # proton mass                 [g]

M_sun = 1.98892e33  # solar mass                 [g]
R_sun = 6.95700e10  # solar radius               [cm]

# Mean molecular weight per electron (C/O white dwarf, fully ionized)
mu_e = 2.0

lambda_e = hbar / (m_e * c)  # [cm]
# P = A f(x); A uses 24 pi^2 (full degenerate-electron gas), not 8 pi^2.
A_eos = m_e * c**2 / (24.0 * np.pi**2 * lambda_e**3)  # [dyne/cm^2]
B_eos = mu_e * m_H / (3.0 * np.pi**2 * lambda_e**3)  # [g/cm^3]


# =============================================================================
# 2. Equation of state (cold degenerate electron gas)
# =============================================================================
def x_from_rho(rho):
    """x from density rho [g/cm^3]."""
    return (rho / B_eos) ** (1.0 / 3.0)


def rho_from_x(x):
    """rho [g/cm^3] from x."""
    return B_eos * x**3


def _chandrasekhar_f(x):
    """f(x) so that P = A_eos * f(x)."""
    return x * (2.0 * x**2 - 3.0) * np.sqrt(1.0 + x**2) + 3.0 * np.arcsinh(x)


def pressure_from_x(x):
    return A_eos * _chandrasekhar_f(x)


def pressure_from_rho(rho):
    return pressure_from_x(x_from_rho(rho))


def dpdrho(rho):
    """dP/drho [cm^2/s^2] — used in hydrostatic equilibrium."""
    x = x_from_rho(rho)
    return 8.0 * A_eos * x**2 / (3.0 * B_eos * np.sqrt(1.0 + x**2))


# =============================================================================
# 3. ODE system for structure: y = [M(r), rho(r)]
# =============================================================================
def make_ode(rho_c: float):
    """RHS and surface event for central density rho_c [g/cm^3]."""
    P_c = pressure_from_rho(rho_c)
    eps = 1e-6
    P_surface = eps * P_c

    def derivatives(r, y):
        M, rho = y
        if rho <= 0.0 or M < 0.0:
            return [0.0, 0.0]
        dM_dr = 4.0 * np.pi * r**2 * rho
        dP_dr = -G * M * rho / r**2
        drho_dr = dP_dr / dpdrho(rho)
        return [dM_dr, drho_dr]

    def surface_event(r, y):
        _, rho = y
        if rho <= 0.0:
            return -1.0
        return pressure_from_rho(rho) - P_surface

    surface_event.terminal = True
    surface_event.direction = -1
    return derivatives, surface_event


def integrate_star(rho_c, r0=1.0e3, r_max=2.0e10):
    """Integrate one star; return total mass M [g] and radius R [cm]."""
    derivatives, surface_event = make_ode(rho_c)
    M0 = (4.0 / 3.0) * np.pi * r0**3 * rho_c
    y0 = [M0, rho_c]

    sol = solve_ivp(
        derivatives,
        t_span=[r0, r_max],
        y0=y0,
        method="RK45",
        events=surface_event,
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
        max_step=5.0e6,
    )

    if len(sol.t_events[0]) > 0:
        R_total = sol.t_events[0][0]
        M_total = sol.y_events[0][0][0]
    else:
        R_total = sol.t[-1]
        M_total = sol.y[0, -1]

    return float(M_total), float(R_total)


def compute_mass_radius_curve(rho_c_arr, verbose=True):
    """Loop over central densities; return masses and radii in M_sun and R_sun."""
    masses = []
    radii = []
    n = len(rho_c_arr)
    for i, rho_c in enumerate(rho_c_arr):
        if verbose and (i % 10 == 0 or i == n - 1):
            print(f"  [{i+1:3d}/{n}]  rho_c = {rho_c:.3e} g/cm^3", end="")
        try:
            M, R = integrate_star(rho_c)
            masses.append(M / M_sun)
            radii.append(R / R_sun)
            if verbose and (i % 10 == 0 or i == n - 1):
                print(f"  ->  M = {M/M_sun:.4f} M_sun,  R = {R/R_sun:.5f} R_sun")
        except Exception as exc:
            if verbose:
                print(f"  WARNING: integration failed — {exc}")
    return np.array(masses), np.array(radii)


# =============================================================================
# 4. Observational data (literature)
# =============================================================================
OBSERVED_WDS = [
    {
        "name": "Sirius B",
        "M": 1.018,
        "M_err": 0.011,
        "R": 0.00864,
        "R_err": 0.00012,
        "ref": "Barstow et al. 2005",
    },
    {
        "name": "40 Eri B",
        "M": 0.501,
        "M_err": 0.011,
        "R": 0.01360,
        "R_err": 0.00020,
        "ref": "Provencal et al. 1998",
    },
    {
        "name": "Procyon B",
        "M": 0.604,
        "M_err": 0.018,
        "R": 0.00960,
        "R_err": 0.00050,
        "ref": "Provencal et al. 1998",
    },
    {
        "name": "Stein 2051 B",
        "M": 0.500,
        "M_err": 0.050,
        "R": 0.01110,
        "R_err": 0.00070,
        "ref": "Provencal et al. 1998",
    },
    {
        "name": "EG 50",
        "M": 0.550,
        "M_err": 0.030,
        "R": 0.01240,
        "R_err": 0.00100,
        "ref": "Provencal et al. 1998",
    },
    {
        "name": "GD 140",
        "M": 0.790,
        "M_err": 0.030,
        "R": 0.00850,
        "R_err": 0.00100,
        "ref": "Provencal et al. 1998",
    },
]


def plot_mass_radius(masses, radii, chandrasekhar_mass, output_path="mass_radius.png"):
    """Theory curve + observations + Chandrasekhar limit line."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        masses,
        radii,
        color="steelblue",
        linewidth=2.2,
        label=r"Chandrasekhar EOS  ($\mu_e = 2$,  C/O)",
        zorder=2,
    )

    ax.axvline(
        chandrasekhar_mass,
        color="steelblue",
        linestyle="--",
        linewidth=1.4,
        alpha=0.6,
        label=rf"Chandrasekhar limit: ${chandrasekhar_mass:.3f}\,M_\odot$",
        zorder=1,
    )

    for wd in OBSERVED_WDS:
        ax.errorbar(
            wd["M"],
            wd["R"],
            xerr=wd["M_err"],
            yerr=wd["R_err"],
            fmt="o",
            color="crimson",
            markersize=6,
            capsize=4,
            elinewidth=1.5,
            ecolor="crimson",
            zorder=3,
        )
        label_offsets = {
            "Sirius B": (4, -12),
            "40 Eri B": (4, 5),
            "Procyon B": (-70, -12),
            "Stein 2051 B": (4, 5),
            "EG 50": (4, 5),
            "GD 140": (4, 5),
        }
        dx, dy = label_offsets.get(wd["name"], (4, 5))
        ax.annotate(
            wd["name"],
            xy=(wd["M"], wd["R"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color="crimson",
        )

    ax.errorbar(
        [],
        [],
        xerr=[],
        yerr=[],
        fmt="o",
        color="crimson",
        markersize=6,
        capsize=4,
        elinewidth=1.5,
        label="Observed white dwarfs\n(Provencal et al. 1998; Barstow et al. 2005)",
    )

    km_per_Rsun = R_sun / 1e5
    ax2 = ax.secondary_yaxis(
        "right",
        functions=(
            lambda r: r * km_per_Rsun,
            lambda r: r / km_per_Rsun,
        ),
    )
    ax2.set_ylabel(r"Radius  [km]", fontsize=12)
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_xlabel(r"Mass  [$M_\odot$]", fontsize=13)
    ax.set_ylabel(r"Radius  [$R_\odot$]", fontsize=13)
    ax.set_title("White Dwarf Mass-Radius Relation", fontsize=14, pad=10)

    ax.set_xlim(0.0, 1.55)
    ax.set_ylim(0.0, 0.024)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)

    ax.legend(fontsize=9.5, loc="upper right")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved -> {output_path}")

    return fig


# Analytic Chandrasekhar mass limit: M_Ch = (5.87 / mu_e^2) M_sun
M_ch_analytic = 5.87 / mu_e**2


def main():
    print("=" * 60)
    print("  White Dwarf Mass-Radius Relation  (PHYS 6260)")
    print("=" * 60)

    print("\nChandrasekhar EOS coefficients (CGS, derived from first principles):")
    print(f"  A = {A_eos:.4e}  dyne/cm^2")
    print(f"  B = {B_eos:.4e}  g/cm^3")
    print(f"  mu_e = {mu_e}  (C/O white dwarf)")

    rho_c_arr = np.logspace(4, 10, 120)

    print(f"\nIntegrating {len(rho_c_arr)} stellar models ...")
    masses, radii = compute_mass_radius_curve(rho_c_arr, verbose=True)

    M_ch_numerical = np.max(masses)
    print("\nChandrasekhar mass limit:")
    print(f"  Numerical  : {M_ch_numerical:.4f} M_sun")
    print(f"  Analytic   : {M_ch_analytic:.4f} M_sun")
    print(f"  Discrepancy: {abs(M_ch_numerical - M_ch_analytic) / M_ch_analytic * 100:.2f}%")

    output_csv = "mass_radius.csv"
    header = "mass_Msun,radius_Rsun,radius_km"
    data = np.column_stack([masses, radii, radii * R_sun / 1e5])
    np.savetxt(output_csv, data, delimiter=",", header=header, comments="")
    print(f"\nNumerical data saved -> {output_csv}")

    plot_mass_radius(masses, radii, M_ch_numerical, output_path="mass_radius.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
