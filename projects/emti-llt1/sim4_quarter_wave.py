"""
Simulation 4 - Quarter-wave transformer matching a 50 ohm line to a 100 ohm load.

Z_T = sqrt(Z0 * ZL); section is lambda/4 long at the design frequency f0 = 1 GHz.
Zin(f) = Z_T (ZL + j Z_T tan(beta l)) / (Z_T + j ZL tan(beta l)),  beta l = (pi/2)(f/f0)
Run:  python sim4_quarter_wave.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z0, ZL, f0 = 50.0, 100.0, 1e9
ZT = np.sqrt(Z0 * ZL)


def gamma_in(f):
    t = np.tan(np.pi / 2 * f / f0)
    Zin = ZT * (ZL + 1j * ZT * t) / (ZT + 1j * ZL * t)
    return Zin, (Zin - Z0) / (Zin + Z0)


def main():
    print("=" * 66)
    print("SIMULATION 4: Quarter-wave transformer (50 ohm line -> 100 ohm load)")
    print("=" * 66)
    print(f"Required transformer impedance Z_T = sqrt(50 x 100) = {ZT:.3f} ohm")
    print(f"Section length = lambda/4 = {3e8/f0/4*100:.2f} cm at f0 = 1 GHz (air)")
    g_un = (ZL - Z0) / (ZL + Z0)
    print(f"Without transformer: |Gamma| = {abs(g_un):.4f}, VSWR = {(1+abs(g_un))/(1-abs(g_un)):.3f},"
          f" return loss = {-20*np.log10(abs(g_un)):.2f} dB\n")

    print(f"{'f (GHz)':>8} {'Zin (ohm)':>22} {'|Gamma|':>9} {'VSWR':>7} {'RL (dB)':>8}")
    for f in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        Zin, g = gamma_in(f * 1e9)
        rl = f"{-20 * np.log10(abs(g)):8.2f}" if abs(g) > 1e-9 else "     inf"
        print(f"{f:8.2f} {Zin.real:10.2f}{Zin.imag:+10.2f}j {abs(g):9.4f} {(1+abs(g))/(1-abs(g)):7.3f} {rl}")

    f = np.linspace(0.01, 2, 2000) * 1e9
    Zin, g = gamma_in(f)
    mag = np.abs(g)
    vswr = (1 + mag) / (1 - mag)
    for lim, name in [(0.1, "|Gamma| < 0.1"), (0.2, "|Gamma| < 0.2")]:
        band = f[mag < lim]
        print(f"Bandwidth for {name}: {band.min()/1e9:.3f}-{band.max()/1e9:.3f} GHz"
              f" ({(band.max()-band.min())/f0*100:.1f}% fractional)")
    print("Perfect match (|Gamma| = 0) at 1 GHz and odd multiples; mismatch returns at 0 and 2 GHz.")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    ax.plot(f / 1e9, mag, "b", lw=2, label="With λ/4 transformer")
    ax.axhline(abs(g_un), color="r", ls="--", label="No transformer")
    ax.axhline(0.2, color="gray", ls=":", label="|Γ| = 0.2")
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("|Γ|"); ax.set_title("Reflection coefficient")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(f / 1e9, np.minimum(vswr, 5), "g", lw=2)
    ax.axhline(2, color="gray", ls=":", label="VSWR = 2")
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("VSWR"); ax.set_title("VSWR vs frequency")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[2]
    ax.plot(f / 1e9, Zin.real, "b", lw=2, label="Re{Zin}")
    ax.plot(f / 1e9, Zin.imag, "r", lw=2, label="Im{Zin}")
    ax.axhline(50, color="k", ls=":", label="50 Ω")
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Ω"); ax.set_title("Input impedance seen by 50 Ω line")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig("sim4_quarter_wave.png", dpi=130)
    print("Plot saved: sim4_quarter_wave.png")


if __name__ == "__main__":
    main()
