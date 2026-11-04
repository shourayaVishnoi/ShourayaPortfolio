"""
Simulation 1 - Gauss's Law: flux of a point charge through a spherical Gaussian surface.

Numerically integrates E.dA over spheres of different radii and compares with Q/eps0.
Parameters: Q = 1 nC at the origin, radii 0.1 m to 5 m.
Run:  python sim1_gauss_flux.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS0 = 8.854187817e-12
K = 1 / (4 * np.pi * EPS0)
Q = 1e-9  # 1 nC


def E_field(q, q_pos, pts):
    r = pts - q_pos
    rmag = np.linalg.norm(r, axis=-1, keepdims=True)
    return K * q * r / rmag**3


def flux_sphere(q, q_pos, R, n=300):
    """Midpoint-rule surface integral of E.n dA over a sphere of radius R centred at the origin."""
    dth, dph = np.pi / n, 2 * np.pi / (2 * n)
    th = (np.arange(n) + 0.5) * dth
    ph = (np.arange(2 * n) + 0.5) * dph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    nrm = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)
    dA = R**2 * np.sin(TH) * dth * dph
    return np.sum(np.sum(E_field(q, q_pos, R * nrm) * nrm, -1) * dA)


def main():
    theory = Q / EPS0
    origin = np.zeros(3)
    print("=" * 66)
    print("SIMULATION 1: Gauss's Law - flux through a spherical Gaussian surface")
    print("=" * 66)
    print(f"Q = {Q:.1e} C at the origin   Theoretical flux Q/eps0 = {theory:.4f} V.m\n")
    print(f"{'R (m)':>8} {'|E| on surface (V/m)':>22} {'Flux (V.m)':>14} {'Error %':>10}")
    table_r = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    for R in table_r:
        phi = flux_sphere(Q, origin, R)
        print(f"{R:8.2f} {K * Q / R**2:22.4f} {phi:14.4f} {abs(phi - theory) / theory * 100:10.5f}")
    print("\n|E| falls as 1/R^2 while the area grows as R^2, so the flux stays equal to Q/eps0.")

    radii = np.linspace(0.1, 5, 50)
    flux_c = np.array([flux_sphere(Q, origin, R, 150) for R in radii])
    # Extra check: charge displaced to x = 2 m, flux only counts once it is enclosed
    q_off = np.array([2.0, 0, 0])
    flux_off = np.array([flux_sphere(Q, q_off, R, 150) for R in radii])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    x = np.linspace(-2, 2, 40)
    X, Y = np.meshgrid(x, x)
    E = E_field(Q, origin, np.stack([X, Y, np.zeros_like(X)], -1) + 1e-9)
    ax.streamplot(X, Y, E[..., 0], E[..., 1], color=np.log10(np.linalg.norm(E, axis=-1)),
                  cmap="viridis", density=1.3)
    for R in (0.5, 1.0, 1.5):
        ax.add_patch(plt.Circle((0, 0), R, fill=False, ec="tab:red", lw=1.5, ls="--"))
    ax.plot(0, 0, "ro", ms=9)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("E-field lines with Gaussian spheres (z = 0)")

    ax = axes[1]
    ax.semilogy(radii, K * Q / radii**2, "m-", lw=2)
    ax.set_xlabel("R (m)"); ax.set_ylabel("|E| at surface (V/m)")
    ax.set_title("Field magnitude on the sphere, |E| ∝ 1/R²"); ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.plot(radii, flux_c, "bo", ms=4, label="Numerical flux, charge at centre")
    ax.plot(radii, flux_off, "c-", lw=2, label="Numerical flux, charge at x = 2 m")
    ax.axhline(theory, color="g", ls="--", lw=2, label="Theory Q/ε0")
    ax.set_xlabel("Sphere radius R (m)"); ax.set_ylabel("Flux (V·m)")
    ax.set_ylim(-10, theory * 1.25)
    ax.set_title("Flux vs radius"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("sim1_gauss_flux.png", dpi=130)
    print("Displaced-charge check: flux is 0 for R < 2 m and Q/eps0 for R > 2 m.")
    print("Plot saved: sim1_gauss_flux.png")


if __name__ == "__main__":
    main()
