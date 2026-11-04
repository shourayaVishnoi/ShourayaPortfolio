"""
Simulation 5 - Smith chart analysis of ZL = 75 + j25 ohm on a 50 ohm lossless line at 1 GHz.
Run:  python sim5_smith_chart.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f, Z0, ZL = 1e9, 50.0, 75 + 25j


def draw_smith(ax):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), "k", lw=1.5)
    for r in [0, 0.2, 0.5, 1, 2, 5]:
        c, rad = r / (1 + r), 1 / (1 + r)
        ax.plot(c + rad * np.cos(t), rad * np.sin(t), color="gray", lw=0.6)
        ax.text(c - rad + 0.01, 0.02, f"{r:g}", fontsize=7, color="gray")
    for x in [0.2, 0.5, 1, 2, 5]:
        for s in (1, -1):
            z = 1j * s * x + np.linspace(0, 50, 3000) ** 2 / 50
            g = (z - 1) / (z + 1)
            g = g[np.abs(g) <= 1.0001]
            ax.plot(g.real, g.imag, color="gray", lw=0.6)
            gt = (1j * s * x - 1) / (1j * s * x + 1)
            ax.text(gt.real * 1.07, gt.imag * 1.07, f"{s*x:+g}j", fontsize=7, color="gray",
                    ha="center", va="center")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.axis("off")


def main():
    print("=" * 66)
    print("SIMULATION 5: Smith chart, ZL = 75 + j25 ohm, Z0 = 50 ohm, f = 1 GHz")
    print("=" * 66)
    zL = ZL / Z0
    print(f"(1) Normalized load impedance zL = ZL/Z0 = {zL.real:.2f} {zL.imag:+.2f}j")
    G = (zL - 1) / (zL + 1)
    mag, ang = abs(G), np.degrees(np.angle(G))
    print(f"(2) Reflection coefficient Gamma = (zL-1)/(zL+1) = {G.real:.4f} {G.imag:+.4f}j")
    print(f"    |Gamma| = {mag:.4f}, angle = {ang:.2f} deg")
    S = (1 + mag) / (1 - mag)
    print(f"(3) VSWR = (1+|Gamma|)/(1-|Gamma|) = {S:.4f}")
    print(f"    Return loss = {-20*np.log10(mag):.2f} dB, mismatch loss = {-10*np.log10(1-mag**2):.3f} dB,"
          f" power reflected = {mag**2*100:.2f}%")
    lam = 3e8 / f
    d_max = (np.radians(ang) % (2 * np.pi)) / (4 * np.pi) * lam
    print(f"    lambda = {lam*100:.1f} cm; first voltage maximum {d_max*100:.2f} cm from the load"
          f" (Zmax = {Z0*S:.2f} ohm), first minimum {(d_max+lam/4)*100:.2f} cm (Zmin = {Z0/S:.2f} ohm)")
    print("(4)/(5) zL, Gamma and the constant-VSWR circle are plotted in sim5_smith_chart.png")
    print(f"(6) Matched? {'YES' if mag < 1e-6 else 'NO'}. A match needs ZL = Z0 (zL = 1, Gamma = 0, VSWR = 1).")
    print(f"    Here Gamma = {mag:.3f} and VSWR = {S:.2f}, so {mag**2*100:.1f}% of incident power is reflected")
    print("    and a standing wave forms. The point lies in the upper half (inductive, x = +0.5) on the")
    print("    r = 1.5 circle, away from the chart centre. It could be matched with a quarter-wave")
    print("    transformer at a voltage max/min, or a single-stub tuner where the VSWR circle meets r = 1.")

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    draw_smith(ax)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(mag * np.cos(t), mag * np.sin(t), "r--", lw=2, label=f"Constant-VSWR circle (VSWR = {S:.2f})")
    ax.annotate("", xy=(G.real, G.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.plot(G.real, G.imag, "bo", ms=10, label=f"zL = {zL.real:.2f}+j{zL.imag:.2f}  (Γ = {mag:.3f}∠{ang:.1f}°)")
    ax.plot(0, 0, "g*", ms=14, label="Matched point (z = 1, Γ = 0)")
    ax.plot([mag], [0], "ms", ms=8, label=f"r = VSWR = {S:.2f} (Vmax)")
    ax.set_title("Smith chart: ZL = 75 + j25 Ω on a 50 Ω line, f = 1 GHz", fontsize=12)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig("sim5_smith_chart.png", dpi=130)
    print("Plot saved: sim5_smith_chart.png")


if __name__ == "__main__":
    main()
