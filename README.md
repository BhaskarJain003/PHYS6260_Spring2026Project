# PHYS 6260 — White dwarf mass–radius relation

Term project: integrate the Newtonian structure equations for a cold, degenerate electron gas using the **Chandrasekhar equation of state** (parameterized by the dimensionless Fermi momentum $x = p_F / m_e c$), then compare the theoretical mass–radius curve to observed white dwarfs.

All of the code lives in **`main.py`** (single file, easier to read for a course project).

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or any environment with **NumPy**, **SciPy**, and **Matplotlib**

Dependencies are listed in `pyproject.toml`.

## How to run

From this directory:

```bash
uv sync
uv run python main.py
```

If you already have a virtual environment with the packages installed, you can also run `python main.py` using that interpreter.

## Outputs

| File | Description |
|------|-------------|
| `mass_radius.csv` | Columns: `mass_Msun`, `radius_Rsun`, `radius_km` |
| `mass_radius.png` | Theory curve, analytic Chandrasekhar-mass marker, and literature data points |

The script prints progress for a grid of central densities, then the numerical maximum mass next to the textbook estimate $M_{\mathrm{Ch}} \approx 5.87/\mu_e^2$ in solar masses (with $\mu_e = 2$ for a C/O composition).

## Physics (short)

- **Structure:** spherical hydrostatic equilibrium with mass continuity, integrated outward in radius until the pressure drops to a small fraction of the central pressure (surface condition).
- **EOS:** $\rho = B x^3$, $P = A f(x)$ with the standard Chandrasekhar $f(x)$; the pressure coefficient uses $A \propto 1/(24\pi^2)$ in the usual degenerate spin-½ normalization so the non-relativistic limit matches $P = K\rho^{5/3}$.

## License / academic use

Use and adapt for coursework as appropriate; cite your sources and observational references from the code comments when writing up the project.
