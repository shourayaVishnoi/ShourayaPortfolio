"""
Simulation 3 - Lossy transmission line from its RLGC parameters.

gamma = sqrt((R + jwL)(G + jwC)) = alpha + j beta,  Z0 = sqrt((R + jwL)/(G + jwC))
Parameters (typical RG-58-like coax): R = 0.1 ohm/m, L = 250 nH/m, G = 1e-5 S/m, C = 100 pF/m.
With constant R and G the attenuation saturates at R/(2Z0) + G Z0/2, so a realistic model is
also run: skin-effect R(f) = R*sqrt(f/1 MHz) (min R) and dielectric G(f) = G + wC tan(delta).
Run:  python sim3_lossy_tline.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, L, G, C = 0.1, 250e-9, 1e-5, 100e-12
TAN_D = 2e-4  # polyethylene-like dielectric loss tangent


def line_params(f, realistic=False):
    w = 2 * np.pi * f
    Rf = np.maximum(R, R * np.sqrt(f / 1e6)) if realistic else R
    Gf = G + w * C * TAN_D if realistic else G
    Z, Y = Rf + 1j * w * L, Gf + 1j * w * C
    gamma = np.sqrt(Z * Y)
    return gamma.real, gamma.imag, np.sqrt(Z / Y)


def main():
    print("=" * 66)
    print("SIMULATION 3: Lossy transmission line (RLGC model)")
    print("=" * 66)
    print(f"R = {R} ohm/m, L = {L*1e9:.0f} nH/m, G = {G} S/m, C = {C*1e12:.0f} pF/m")
    print(f"Lossless Z0 = sqrt(L/C) = {np.sqrt(L/C):.2f} ohm, v = 1/sqrt(LC) = {1/np.sqrt(L*C):.3e} m/s\n")
    print(f"{'f (Hz)':>10} {'alpha (Np/m)':>13} {'alpha (dB/m)':>13} {'beta (rad/m)':>13} {'|Z0| (ohm)':>11} {'v_p (m/s)':>11}")
    for f in [1e3, 1e5, 1e6, 1e7, 1e8, 1e9]:
        a, b, Z0 = line_params(f)
        print(f"{f:10.0e} {a:13.4e} {a*8.686:13.4e} {b:13.4e} {abs(Z0):11.2f} {2*np.pi*f/b:11.3e}")
    print("Low-loss limit: alpha ~ R/(2Z0) + G Z0/2 =", f"{R/(2*np.sqrt(L/C)) + G*np.sqrt(L/C)/2:.4e} Np/m")
    print("With constant R and G, alpha stops rising once wL >> R (above ~1 MHz).")

    print(f"\nRealistic model (skin effect + dielectric loss, tan d = {TAN_D}):")
    print(f"{'f (Hz)':>10} {'alpha (dB/m)':>13} {'dB per 100 m':>13}")
    for f in [1e6, 1e7, 1e8, 1e9]:
        a, _, _ = line_params(f, True)
        print(f"{f:10.0e} {a*8.686:13.4e} {a*868.6:13.2f}")

    print(f"\nEffect of length (realistic model):")
    print(f"{'l (m)':>6} {'f (MHz)':>8} {'atten (dB)':>11} {'|V/V0|':>8} {'phase (deg)':>12}")
    for f in [10e6, 100e6, 1e9]:
        a, b, _ = line_params(f, True)
        for l in [1, 10, 50, 100]:
            print(f"{l:6d} {f/1e6:8.0f} {a*l*8.686:11.3f} {np.exp(-a*l):8.4f} {np.degrees(b*l):12.1f}")
    print("Attenuation (dB) and phase shift both grow linearly with length;"
          " both also increase with frequency.")

    freqs = np.logspace(3, 9.5, 400)
    alpha, beta, Z0f = line_params(freqs)
    alpha_r, _, _ = line_params(freqs, True)
    lengths = np.linspace(0, 100, 400)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.loglog(freqs, alpha * 8.686, "r", lw=2, label="Constant R, G")
    ax.loglog(freqs, alpha_r * 8.686, "b--", lw=2, label="Skin effect + dielectric loss")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Attenuation α (dB/m)")
    ax.set_title("Attenuation constant vs frequency"); ax.grid(alpha=0.3, which="both"); ax.legend()

    ax = axes[0, 1]
    ax.loglog(freqs, beta, "b", lw=2, label="β")
    ax2 = ax.twinx()
    ax2.semilogx(freqs, np.abs(Z0f), "k--", label="|Z0|")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("β (rad/m)", color="b"); ax2.set_ylabel("|Z0| (Ω)")
    ax.set_title("Phase constant and characteristic impedance"); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    for f in [1e6, 1e7, 1e8, 1e9]:
        a, _, _ = line_params(f, True)
        ax.plot(lengths, a * lengths * 8.686, lw=2, label=f"{f/1e6:g} MHz")
    ax.set_xlabel("Line length (m)"); ax.set_ylabel("Total attenuation (dB)")
    ax.set_title("Attenuation vs length (realistic model)"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1, 1]
    for f in [1e6, 1e7, 1e8]:
        a, b, _ = line_params(f, True)
        z = np.linspace(0, 20, 2000)
        ax.plot(z, np.exp(-a * z) * np.cos(b * z), lw=1.4, label=f"{f/1e6:g} MHz")
    ax.set_xlabel("Position z (m)"); ax.set_ylabel("Re{V(z)/V0}")
    ax.set_title("Travelling voltage wave along the line (phase shift βz)")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig("sim3_lossy_tline.png", dpi=130)
    print("Plot saved: sim3_lossy_tline.png")


if __name__ == "__main__":
    main()
