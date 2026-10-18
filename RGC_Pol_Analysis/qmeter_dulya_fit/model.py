"""Liverpool Q-meter with a deuteron (Dulya) susceptibility in the coil.

Raw sweep model:
    V(f) = Re[-I * Ztotal(f; L(f)) * exp(i*phi)] + DC_offset
    L(f) = L0 * (1 + 4*pi*eta_fill * chi(f)),   chi = chi' - i*chi''
    chi''(f) = A * [r*I_plus(x) + I_minus(x)],   x = (f - f0) / wq
    chi'(f)  = Kramers-Kronig (Hilbert) partner of chi''
    P        = (r^2 - 1) / (r^2 + r + 1)   (spin-1, equal spin temperature)

Circuit topology and fixed constants follow data_creation/physics/baseline.py. Fitted:
the run knobs (current/gain, tuning C, cable trim, reference phase, DC offset) plus
coil L/R, cable loss and stray C, all bounded to physical (positive) ranges.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_creation"))
from physics.lineshape import PowderLineshape

R, R1, R_SMALL = 619.0, 50.0, 10.0
BETA1, Z_CABLE, D_CAB, M_CAB = 4.752e-9, 50.0, 1.027e-10, 2.542e-7
F_RES_MHZ = 32.68
ETA_FILL = 0.0104

# U [V], Cknob [x20 pF], trim [rad of cable phase at F_RES], phi_deg [deg],
# DC_offset [V], L0 [x10 nH], Rcoil [ohm], alpha [cable loss], Cstray [pF]
CIRCUIT_NAMES = ("U", "Cknob", "trim", "phi_deg", "DC_offset", "L0", "Rcoil", "alpha", "Cstray")
CIRCUIT_LO = (0.001, 0.5, 0.0, -360.0, -50.0, 0.5, 0.01, 0.001, 0.0)
CIRCUIT_HI = (100.0, 500.0, 20.0, 360.0, 50.0, 30.0, 10.0, 1.0, 50.0)

SIGNAL_NAMES = ("A", "P", "f0", "wq", "eta_q", "g")
SIGNAL_LO = (-np.inf, -0.95, 32.60, 0.05, 0.0, 0.001)
SIGNAL_HI = (np.inf, 0.95, 32.77, 0.15, 0.4, 0.08)
POWDER_NPHI = 32
PAD_MHZ = 3.0


def qmeter(f_mhz, U, Cknob, trim, phi_deg, DC_offset, L0, Rcoil, alpha, Cstray, chi=0.0):
    w = 2 * np.pi * np.asarray(f_mhz) * 1e6
    w_res = 2 * np.pi * F_RES_MHZ * 1e6
    I = U * 1000 / R
    Z0 = np.sqrt((2 * Z_CABLE * alpha + 1j * w * M_CAB) / (1j * w * D_CAB))
    gamma = alpha + 1j * BETA1 * w
    L = L0 * 1e-8 * (1 + 4 * np.pi * ETA_FILL * chi)
    ZL_pure = 1j * w * L + Rcoil
    ZL = ZL_pure / (1 + 1j * w * Cstray * 1e-12 * ZL_pure)
    t = np.tanh(gamma * trim / (BETA1 * w_res))
    ZT = Z0 * (ZL + Z0 * t) / (Z0 + ZL * t)
    Zleg = R_SMALL + 1 / (1j * w * 20e-12 * Cknob) + ZT
    Ztot = R1 / (1 + R1 / Zleg)
    return np.real(-I * Ztot * np.exp(1j * np.deg2rad(phi_deg))) + DC_offset


def p_to_r(P):
    return (np.sqrt(4 - 3 * P**2) + P) / (2 - 2 * P)


def chi_components(f_mhz, A, P, f0, wq, eta_q, g):
    """chi'' and chi' on the data grid. chi' via Hilbert on a padded uniform grid."""
    f_mhz = np.asarray(f_mhz)
    df = float(np.median(np.diff(f_mhz)))
    grid = np.arange(f_mhz[0] - PAD_MHZ, f_mhz[-1] + PAD_MHZ, df)
    x = (grid - f0) / wq
    i_plus = PowderLineshape(x, 1, eta_q, g, nphi=POWDER_NPHI)
    i_minus = PowderLineshape(x, -1, eta_q, g, nphi=POWDER_NPHI)
    chi2 = A * (p_to_r(P) * i_plus + i_minus)
    chi1 = -np.imag(hilbert(chi2))
    return np.interp(f_mhz, grid, chi2), np.interp(f_mhz, grid, chi1)


def full_model(f_mhz, circuit, signal):
    chi2, chi1 = chi_components(f_mhz, *signal)
    return qmeter(f_mhz, *circuit, chi=chi1 - 1j * chi2)


def signal_component(f_mhz, circuit, signal):
    """What the signal adds to the Q-meter output (the analogue of online fitsub)."""
    return full_model(f_mhz, circuit, signal) - qmeter(f_mhz, *circuit)


def absorption_dispersion_parts(f_mhz, circuit, signal):
    """Split the output signal into the part driven by chi'' and the part driven by chi'."""
    chi2, chi1 = chi_components(f_mhz, *signal)
    bg = qmeter(f_mhz, *circuit)
    return qmeter(f_mhz, *circuit, chi=-1j * chi2) - bg, qmeter(f_mhz, *circuit, chi=chi1 + 0j) - bg
