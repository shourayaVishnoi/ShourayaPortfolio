"""
Simulation 2 - Displacement current in a parallel-plate capacitor.

A sinusoidal voltage V(t) = V0 sin(wt) drives an air-filled parallel-plate capacitor.
E = V/d, D = eps E, and the displacement current is Id = A dD/dt = eps A/d * dV/dt.
Parameters: V0 = 10 V, d = 1 mm, eps_r = 1, base area 0.01 m^2 (10 cm x 10 cm), base f = 1 MHz.
Run:  python sim2_displacement_current.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS0 = 8.854187817e-12
MU0 = 4e-7 * np.pi
V0, d, eps_r = 10.0, 1e-3, 1.0
A0, f0 = 0.01, 1e6
eps = EPS0 * eps_r


def displacement_current(t, f, A):
    w = 2 * np.pi * f
    E = V0 * np.sin(w * t) / d
    Jd = eps * V0 * w * np.cos(w * t) / d   # dD/dt
    return E, Jd, Jd * A


def main():
    print("=" * 66)
    print("SIMULATION 2: Displacement current in a parallel-plate capacitor")
    print("=" * 66)
    C = eps * A0 / d
    print(f"V0 = {V0} V, d = {d*1e3} mm, A = {A0} m^2, eps_r = {eps_r}  ->  C = {C*1e12:.2f} pF")
    print(f"Peak E = V0/d = {V0/d:.0f} V/m\n")

    t = np.linspace(0, 2 / f0, 1000)
    E, Jd, Id = displacement_current(t, f0, A0)
    Ic = C * np.gradient(V0 * np.sin(2 * np.pi * f0 * t), t)   # conduction current in the wires
    print(f"At f = 1 MHz: peak Jd = {Jd.max():.4f} A/m^2, peak Id = {Id.max()*1e3:.4f} mA")
    print(f"Peak wire conduction current C dV/dt = {Ic.max()*1e3:.4f} mA (equal to Id, as Maxwell predicts)")
    # Magnetic field between the plates at radius r (circular-equivalent plate radius)
    a = np.sqrt(A0 / np.pi)
    print(f"Peak B at the plate edge (r = {a*100:.2f} cm): {MU0 * Id.max() / (2*np.pi*a)*1e9:.3f} nT\n")

    freqs = np.logspace(2, 9, 200)
    Id_f = eps * A0 * V0 * 2 * np.pi * freqs / d
    areas = np.linspace(1e-4, 0.05, 200)
    Id_A = eps * areas * V0 * 2 * np.pi * f0 / d

    print("Effect of frequency (A = 0.01 m^2):")
    for f in [1e3, 1e5, 1e6, 1e8, 1e9]:
        print(f"  f = {f:10.0e} Hz  ->  Id,peak = {eps*A0*V0*2*np.pi*f/d:.4e} A")
    print("Effect of plate area (f = 1 MHz):")
    for A in [0.001, 0.005, 0.01, 0.02, 0.05]:
        print(f"  A = {A:6.3f} m^2   ->  Id,peak = {eps*A*V0*2*np.pi*f0/d*1e3:.4f} mA")
    print("Id is proportional to both f and A (doubling either doubles Id).")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(t * 1e6, E / 1e3, "b", label="E (kV/m)")
    ax2 = ax.twinx()
    ax2.plot(t * 1e6, Id * 1e3, "r", label="Id (mA)")
    ax.set_xlabel("t (µs)"); ax.set_ylabel("E (kV/m)", color="b"); ax2.set_ylabel("Id (mA)", color="r")
    ax.set_title("E and Id vs time at 1 MHz (Id leads E by 90°)"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    x = np.linspace(-0.08, 0.08, 25); y = np.linspace(-2e-3, 2e-3, 25)
    X, Y = np.meshgrid(x, y)
    plate = np.sqrt(A0) / 2
    Ey = np.where((np.abs(X) < plate) & (np.abs(Y) < d / 2), -V0 / d, 0.0)
    ax.quiver(X * 100, Y * 1e3, np.zeros_like(Ey), Ey, color="tab:blue", scale=2e5)
    for yp in (d / 2, -d / 2):
        ax.plot([-plate * 100, plate * 100], [yp * 1e3] * 2, "k", lw=4)
    ax.text(0, 0.65, "+V", ha="center"); ax.text(0, -0.8, "−V", ha="center")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (mm)")
    ax.set_title("Uniform E between the plates (instant of peak V)")

    ax = axes[1, 0]
    ax.loglog(freqs, Id_f, "g", lw=2)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Peak Id (A)")
    ax.set_title("Displacement current vs frequency (A = 0.01 m²)"); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.plot(areas * 1e4, Id_A * 1e3, "m", lw=2)
    ax.set_xlabel("Plate area (cm²)"); ax.set_ylabel("Peak Id (mA)")
    ax.set_title("Displacement current vs plate area (f = 1 MHz)"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("sim2_displacement_current.png", dpi=130)
    print("Plot saved: sim2_displacement_current.png")


if __name__ == "__main__":
    main()
