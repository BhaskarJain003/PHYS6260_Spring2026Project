"""
PHYS 6260 Term Project - White Dwarf Mass-Radius Relation
Numerically integrates the hydrostatic equilibrium equations using the
Chandrasekhar equation of state for a cold degenerate electron gas.

Run with: uv run python main.py
Outputs:  mass_radius.png, mass_radius.csv
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Physical constants, all in CGS
G    = 6.67430e-8   # gravitational constant [cm^3 g^-1 s^-2]
c    = 2.99792e10   # speed of light [cm/s]
hbar = 1.05457e-27  # reduced Planck constant [erg s]
m_e  = 9.10938e-28  # electron mass [g]
m_H  = 1.67262e-24  # proton mass [g]
M_sun = 1.98892e33  # solar mass [g]
R_sun = 6.95700e10  # solar radius [cm]

# mu_e = mean molecular weight per electron = 2 for a C/O white dwarf
mu_e = 2.0

# Reduced electron Compton wavelength: lambda_e = hbar / (m_e c) [cm]
lambda_e = hbar / (m_e * c)

# Chandrasekhar EOS coefficients derived from first principles.
# The EOS is parametric in x = p_F / (m_e c), the dimensionless Fermi momentum:
#   rho = B * x^3          [g/cm^3]
#   P   = A * f(x)         [dyne/cm^2]
# where f(x) = x(2x^2-3)sqrt(1+x^2) + 3*arcsinh(x)
#
# A comes from integrating the relativistic Fermi pressure (factor 24*pi^2 from
# spin degeneracy + phase space), B from the electron number density.
A_eos = m_e * c**2 / (24.0 * np.pi**2 * lambda_e**3)  # [dyne/cm^2] ~ 6e22
B_eos = mu_e * m_H / (3.0 * np.pi**2 * lambda_e**3)   # [g/cm^3]    ~ 2e6


# Convert density [g/cm^3] to dimensionless Fermi momentum x = p_F/(m_e c)
def x_from_rho(rho):
    return (rho / B_eos) ** (1.0 / 3.0)


# Convert x back to density [g/cm^3]
def rho_from_x(x):
    return B_eos * x**3


# Chandrasekhar function f(x); pressure is P = A_eos * f(x) [dyne/cm^2]
def _chandrasekhar_f(x):
    return x * (2.0 * x**2 - 3.0) * np.sqrt(1.0 + x**2) + 3.0 * np.arcsinh(x)


def pressure_from_x(x):
    return A_eos * _chandrasekhar_f(x)   # [dyne/cm^2]


def pressure_from_rho(rho):
    return pressure_from_x(x_from_rho(rho))   # [dyne/cm^2]


# Analytic derivative dP/drho [cm^2/s^2], used in the ODE.
# Comes from differentiating P(x) and rho(x) w.r.t. x and applying chain rule.
def dpdrho(rho):
    x = x_from_rho(rho)
    return 8.0 * A_eos * x**2 / (3.0 * B_eos * np.sqrt(1.0 + x**2))


# Build the ODE right-hand side and surface-detection event for a given rho_c.
# State vector: y = [M(r) in g, rho(r) in g/cm^3], independent variable r in cm.
def make_ode(rho_c):
    P_c       = pressure_from_rho(rho_c)
    P_surface = 1e-6 * P_c   # stop when pressure drops to 1 ppm of central value

    # Hydrostatic equilibrium + mass continuity, rewritten for rho as state var
    def derivatives(r, y):
        M, rho = y
        if rho <= 0.0 or M < 0.0:
            return [0.0, 0.0]
        dM_dr   =  4.0 * np.pi * r**2 * rho          # mass shell [g/cm]
        dP_dr   = -G * M * rho / r**2                 # hydrostatic eq [dyne/cm^3]
        drho_dr =  dP_dr / dpdrho(rho)                # chain rule [g/cm^4]
        return [dM_dr, drho_dr]

    # Terminal event: integration stops when P falls to P_surface
    def surface_event(r, y):
        _, rho = y
        if rho <= 0.0:
            return -1.0
        return pressure_from_rho(rho) - P_surface

    surface_event.terminal  = True
    surface_event.direction = -1
    return derivatives, surface_event


# Integrate the structure equations for one star with central density rho_c [g/cm^3].
# Returns total mass M [g] and radius R [cm].
def integrate_star(rho_c, r0=1.0e3, r_max=2.0e10):
    derivatives, surface_event = make_ode(rho_c)

    # Start at r0 = 1000 cm (10 m) to avoid the r=0 singularity.
    # The enclosed mass there is negligibly small.
    M0 = (4.0 / 3.0) * np.pi * r0**3 * rho_c   # [g]
    y0 = [M0, rho_c]

    sol = solve_ivp(
        derivatives,
        t_span=[r0, r_max],   # r in [cm]
        y0=y0,
        method="RK45",        # Dormand-Prince adaptive stepper
        events=surface_event,
        rtol=1e-8,
        atol=1e-10,
        max_step=5.0e6,       # max step = 50 km [cm]
    )

    if len(sol.t_events[0]) > 0:
        R_total = sol.t_events[0][0]        # [cm]
        M_total = sol.y_events[0][0][0]     # [g]
    else:
        R_total = sol.t[-1]
        M_total = sol.y[0, -1]

    return float(M_total), float(R_total)


# Sweep over a grid of central densities and collect (M, R) pairs.
def compute_mass_radius_curve(rho_c_arr, verbose=True):
    masses, radii = [], []
    n = len(rho_c_arr)
    for i, rho_c in enumerate(rho_c_arr):
        if verbose and (i % 10 == 0 or i == n - 1):
            print(f"  [{i+1:3d}/{n}]  rho_c = {rho_c:.3e} g/cm^3", end="")
        try:
            M, R = integrate_star(rho_c)
            masses.append(M / M_sun)   # store in solar units
            radii.append(R  / R_sun)
            if verbose and (i % 10 == 0 or i == n - 1):
                print(f"  ->  M = {M/M_sun:.4f} Msun,  R = {R/R_sun:.5f} Rsun")
        except Exception as exc:
            if verbose:
                print(f"  WARNING: failed for rho_c = {rho_c:.2e} -- {exc}")
    return np.array(masses), np.array(radii)


# Observed white dwarfs with measured masses and radii (literature values)
OBSERVED_WDS = [
    {"name": "Sirius B",     "M": 1.018, "M_err": 0.011, "R": 0.00864, "R_err": 0.00012, "ref": "Barstow et al. 2005"},
    {"name": "40 Eri B",     "M": 0.501, "M_err": 0.011, "R": 0.01360, "R_err": 0.00020, "ref": "Provencal et al. 1998"},
    {"name": "Procyon B",    "M": 0.604, "M_err": 0.018, "R": 0.00960, "R_err": 0.00050, "ref": "Provencal et al. 1998"},
    {"name": "Stein 2051 B", "M": 0.500, "M_err": 0.050, "R": 0.01110, "R_err": 0.00070, "ref": "Provencal et al. 1998"},
    {"name": "EG 50",        "M": 0.550, "M_err": 0.030, "R": 0.01240, "R_err": 0.00100, "ref": "Provencal et al. 1998"},
    {"name": "GD 140",       "M": 0.790, "M_err": 0.030, "R": 0.00850, "R_err": 0.00100, "ref": "Provencal et al. 1998"},
]


def plot_mass_radius(masses, radii, chandrasekhar_mass, output_path="mass_radius.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(masses, radii, color="steelblue", linewidth=2.2,
            label=r"Chandrasekhar EOS  ($\mu_e = 2$,  C/O)", zorder=2)

    ax.axvline(chandrasekhar_mass, color="steelblue", linestyle="--",
               linewidth=1.4, alpha=0.6,
               label=rf"Chandrasekhar limit: ${chandrasekhar_mass:.3f}\,M_\odot$",
               zorder=1)

    label_offsets = {"Sirius B": (4, -12), "40 Eri B": (4, 5),
                     "Procyon B": (-70, -12), "Stein 2051 B": (4, 5),
                     "EG 50": (4, 5), "GD 140": (4, 5)}

    for wd in OBSERVED_WDS:
        ax.errorbar(wd["M"], wd["R"], xerr=wd["M_err"], yerr=wd["R_err"],
                    fmt="o", color="crimson", markersize=6,
                    capsize=4, elinewidth=1.5, ecolor="crimson", zorder=3)
        dx, dy = label_offsets.get(wd["name"], (4, 5))
        ax.annotate(wd["name"], xy=(wd["M"], wd["R"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color="crimson")

    # Dummy errorbar entry just for the legend
    ax.errorbar([], [], xerr=[], yerr=[], fmt="o", color="crimson",
                markersize=6, capsize=4, elinewidth=1.5,
                label="Observed white dwarfs\n(Provencal et al. 1998; Barstow et al. 2005)")

    # Secondary y-axis in km (1 Rsun = 695,700 km)
    km_per_Rsun = R_sun / 1e5
    ax2 = ax.secondary_yaxis("right",
                              functions=(lambda r: r * km_per_Rsun,
                                         lambda r: r / km_per_Rsun))
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


# Analytic Chandrasekhar limit: M_Ch = 5.87 / mu_e^2 Msun ~ 1.47 Msun for mu_e=2
M_ch_analytic = 5.87 / mu_e**2


def main():
    print("=" * 60)
    print("  White Dwarf Mass-Radius Relation  (PHYS 6260)")
    print("=" * 60)

    print("\nEOS coefficients (CGS):")
    print(f"  A = {A_eos:.4e}  dyne/cm^2")
    print(f"  B = {B_eos:.4e}  g/cm^3")

    # 120 central densities log-spaced from 10^4 to 10^10 g/cm^3
    rho_c_arr = np.logspace(4, 10, 120)

    print(f"\nIntegrating {len(rho_c_arr)} stellar models ...")
    masses, radii = compute_mass_radius_curve(rho_c_arr, verbose=True)

    M_ch_numerical = np.max(masses)
    print("\nChandrasekhar mass limit:")
    print(f"  Numerical  : {M_ch_numerical:.4f} Msun")
    print(f"  Analytic   : {M_ch_analytic:.4f} Msun")
    print(f"  Discrepancy: {abs(M_ch_numerical - M_ch_analytic) / M_ch_analytic * 100:.2f}%")

    # Save M-R table to CSV [Msun, Rsun, km]
    data = np.column_stack([masses, radii, radii * R_sun / 1e5])
    np.savetxt("mass_radius.csv", data, delimiter=",",
               header="mass_Msun,radius_Rsun,radius_km", comments="")
    print("\nData saved -> mass_radius.csv")

    plot_mass_radius(masses, radii, M_ch_numerical, output_path="mass_radius.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
