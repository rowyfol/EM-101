"""
The Neuron as an Electric Cable
================================
Part 1 — Membrane as a Capacitor
Part 2 — Hodgkin-Huxley Action Potential
Part 3 — Propagating Action Potential
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
os.makedirs("report/figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
BLUE = "#2563EB"
RED = "#DC2626"
GREEN = "#16A34A"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "#f9fafb",
        "axes.edgecolor": "#d1d5db",
        "axes.grid": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.6,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    }
)


# ===========================================================================
# PART 1 — Membrane as a Capacitor
# ===========================================================================
print("=" * 60)
print("PART 1 — Membrane as a Capacitor")
print("=" * 60)

eps0 = 8.854e-12  # F/m
eps_r = 2.1
d = 5e-9  # m  (membrane thickness)
eps_m = eps_r * eps0

Vm_rest = 70e-3  # V
E_rest = Vm_rest / d
Cm_mem = eps_m / d
we_rest = 0.5 * eps_m * E_rest**2

print(f"  E  = {E_rest / 1e6:.2f} MV/m")
print(f"  Cm = {Cm_mem * 1e3:.4f} mF/m²  ({Cm_mem * 1e2:.4f} µF/cm²)")
print(f"  we = {we_rest:.4f} J/m³")

# Sweep 0–120 mV
Vm_sweep = np.linspace(0, 120e-3, 500)
E_sweep = Vm_sweep / d
we_sweep = 0.5 * eps_m * E_sweep**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle("Part 1 — Membrane as a Capacitor", fontweight="bold")

# Left: E-field vs Vm
ax1.plot(Vm_sweep * 1e3, E_sweep / 1e6, color=BLUE)
ax1.axvline(70, color=RED, linestyle="--", linewidth=1.4, label="Vm = 70 mV")
ax1.text(
    0.05,
    0.92,
    "E = Vm / d",
    transform=ax1.transAxes,
    fontsize=9,
    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
)
ax1.set_xlabel("Membrane voltage (mV)")
ax1.set_ylabel("Electric field (MV/m)")
ax1.set_title("Electric Field")
ax1.legend()

# Right: energy density vs Vm
ax2.plot(Vm_sweep * 1e3, we_sweep, color=PURPLE)
ax2.axvline(70, color=RED, linestyle="--", linewidth=1.4, label="Vm = 70 mV")
ax2.text(
    0.05,
    0.92,
    "we = ½εE²",
    transform=ax2.transAxes,
    fontsize=9,
    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
)
ax2.set_xlabel("Membrane voltage (mV)")
ax2.set_ylabel("Energy density (J/m³)")
ax2.set_title("Stored Energy Density")
ax2.legend()

fig.tight_layout()
fig.savefig("report/figures/fig1_membrane.pdf", bbox_inches="tight")
fig.savefig("report/figures/fig1_membrane.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig1_membrane.pdf / .png")


# ===========================================================================
# PART 2 — Hodgkin-Huxley Action Potential
# ===========================================================================
print()
print("=" * 60)
print("PART 2 — Hodgkin-Huxley Action Potential")
print("=" * 60)

# HH parameters (mV, ms, µA/cm², mS/cm², µF/cm²)
Cm = 1.0
gNa = 120.0
ENa = 50.0
gK = 36.0
EK = -77.0
gL = 0.3
EL = -54.387


def alpha_m(V):
    return np.where(
        np.abs(V + 40) < 1e-7, 1.0, 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    )


def beta_m(V):
    return 4.0 * np.exp(-(V + 65) / 18)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)


def beta_h(V):
    return 1.0 / (1 + np.exp(-(V + 35) / 10))


def alpha_n(V):
    return np.where(
        np.abs(V + 55) < 1e-7, 0.1, 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    )


def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)


# Resting state at V0 = -65 mV
V0 = -65.0
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
print(f"  Resting gates — m0={float(m0):.4f}, h0={float(h0):.4f}, n0={float(n0):.4f}")


def hh_rhs(t, y):
    V, m, h, n = y
    I_ext = 10.0 if 5.0 < t < 30.0 else 0.0
    I_Na = gNa * m**3 * h * (V - ENa)
    I_K = gK * n**4 * (V - EK)
    I_L = gL * (V - EL)
    dVdt = (I_ext - I_Na - I_K - I_L) / Cm
    dmdt = float(alpha_m(V)) * (1 - m) - float(beta_m(V)) * m
    dhdt = float(alpha_h(V)) * (1 - h) - float(beta_h(V)) * h
    dndt = float(alpha_n(V)) * (1 - n) - float(beta_n(V)) * n
    return [dVdt, dmdt, dhdt, dndt]


t_eval = np.linspace(0, 60, 3000)
sol = solve_ivp(
    hh_rhs,
    (0, 60),
    [V0, m0, h0, n0],
    method="RK45",
    t_eval=t_eval,
    rtol=1e-7,
    atol=1e-9,
)
t_hh, V_hh, m_hh, h_hh, n_hh = sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"hspace": 0.4})
fig.suptitle("Part 2 — Hodgkin-Huxley Action Potential", fontweight="bold")

# Row 1: membrane voltage
ax1.plot(t_hh, V_hh, color=BLUE, label="V(t)")
ax1.axhline(V0, color="gray", linestyle="--", linewidth=1.2, label="Resting (−65 mV)")
ax1.axvspan(5, 30, color=ORANGE, alpha=0.15, label="Stimulus on")
ax1.set_xlim(0, 60)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Voltage (mV)")
ax1.set_title("Membrane Voltage")
ax1.legend(loc="upper right")

# Row 2: gating variables
ax2.plot(t_hh, m_hh, color=RED, label="m")
ax2.plot(t_hh, h_hh, color=BLUE, label="h")
ax2.plot(t_hh, n_hh, color=GREEN, label="n")
ax2.set_xlim(0, 60)
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Gate probability")
ax2.set_title("Gating Variables")
ax2.legend(loc="upper right")

fig.savefig("report/figures/fig2_hh.pdf", bbox_inches="tight")
fig.savefig("report/figures/fig2_hh.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig2_hh.pdf / .png")


# ===========================================================================
# PART 3 — Propagating Action Potential
# ===========================================================================
print()
print("=" * 60)
print("PART 3 — Propagating Action Potential")
print("=" * 60)

N_ax = 120
a_ax = 0.005  # cm  (50 µm)
Ri_ax = 0.1  # kΩ·cm
dx_ax = 0.05  # cm
gc = a_ax / (2 * Ri_ax * dx_ax**2)

print(f"  gc = {gc:.4f} mS/cm²  (axial coupling conductance)")

# Resting initial conditions for all compartments
y0_prop = np.zeros(4 * N_ax)
for i in range(N_ax):
    y0_prop[4 * i] = V0
    y0_prop[4 * i + 1] = m0
    y0_prop[4 * i + 2] = h0
    y0_prop[4 * i + 3] = n0


def prop_rhs(t, y):
    dy = np.zeros_like(y)
    V = y[0::4]
    m = y[1::4]
    h = y[2::4]
    n = y[3::4]

    # Stimulus: first 4 compartments, 0.5–2.0 ms, 20 µA/cm²
    I_ext = np.zeros(N_ax)
    if 0.5 < t < 2.0:
        I_ext[:4] = 20.0

    # HH ionic currents (vectorized)
    I_Na = gNa * m**3 * h * (V - ENa)
    I_K = gK * n**4 * (V - EK)
    I_L = gL * (V - EL)

    # Axial (cable) current via finite differences
    I_ax = np.zeros(N_ax)
    I_ax[1:-1] = gc * (V[:-2] - 2 * V[1:-1] + V[2:])
    I_ax[0] = gc * (V[1] - V[0])
    I_ax[-1] = gc * (V[-2] - V[-1])

    dVdt = (I_ext - I_Na - I_K - I_L + I_ax) / Cm
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    dy[0::4] = dVdt
    dy[1::4] = dmdt
    dy[2::4] = dhdt
    dy[3::4] = dndt
    return dy


t_eval_p = np.linspace(0, 15, 300)
sol_p = solve_ivp(
    prop_rhs,
    (0, 15),
    y0_prop,
    method="RK45",
    t_eval=t_eval_p,
    rtol=1e-4,
    atol=1e-6,
    max_step=0.05,
)

# Reshape: V_prop[compartment, time]
V_prop = sol_p.y[0::4]  # shape (N_ax, 300)
x_ax = np.arange(N_ax) * dx_ax * 10  # mm

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Part 3 — Propagating Action Potential", fontweight="bold")

# Left: space-time map
im = ax1.imshow(
    V_prop,
    aspect="auto",
    origin="lower",
    extent=[0, 15, 0, x_ax[-1]],
    cmap="RdBu_r",
    vmin=-80,
    vmax=45,
)
fig.colorbar(im, ax=ax1, label="Voltage (mV)")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Position (mm)")
ax1.set_title("Space–Time Map")

# Right: traces at selected compartments
probe_idx = [0, 25, 50, 75, 100, 115]
colors_p = [BLUE, RED, GREEN, ORANGE, PURPLE, "#0891B2"]
for idx, col in zip(probe_idx, colors_p):
    ax2.plot(sol_p.t, V_prop[idx], color=col, label=f"x = {x_ax[idx]:.1f} mm")
ax2.axhline(V0, color="gray", linestyle="--", linewidth=1.2)
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Voltage (mV)")
ax2.set_title("Voltage at Different Positions")
ax2.legend(loc="upper right")

fig.tight_layout()
fig.savefig("report/figures/fig3_propagation.pdf", bbox_inches="tight")
fig.savefig("report/figures/fig3_propagation.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig3_propagation.pdf / .png")

print()
print("Done! All figures saved to report/figures/")
