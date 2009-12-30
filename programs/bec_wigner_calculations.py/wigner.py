import math

class Dim:

    fx = 11.96 # Hz
    fy = 97.6
    fz = 97.6

    wx = 2 * math.pi * fx
    wy = 2 * math.pi * fy
    wz = 2 * math.pi * fz

    fxyz = (fx * fy * fz) ** (1.0 / 3)
    wxyz = (wx * wy * wz) ** (1.0 / 3)

    N = 10000
    m = 1.443160648e-25 # kg
    hbar = 1.054571628e-34 # J * s
    a0 = 5.2917720859e-11 # m

    a11 = 100.4 * a0
    a22 = 95.0 * a0
    a12 = 97.66 * a0

    g11 = 4 * math.pi * hbar * hbar * a11 / m
    g22 = 4 * math.pi * hbar * hbar * a22 / m
    g12 = 4 * math.pi * hbar * hbar * a12 / m

    mu = 0.5 * wxyz * hbar * ((15 * N * a11 / math.sqrt(hbar / (wxyz * m))) ** 0.4)

    xmax = math.sqrt(2 * mu / (m * (wx ** 2)))
    ymax = math.sqrt(2 * mu / (m * (wy ** 2)))
    zmax = math.sqrt(2 * mu / (m * (wz ** 2)))


class NU:

    mu = Dim.mu / (Dim.hbar * Dim.wz)
    lx = math.sqrt(Dim.hbar / (Dim.m * Dim.wx))
    ly = math.sqrt(Dim.hbar / (Dim.m * Dim.wy))
    lz = math.sqrt(Dim.hbar / (Dim.m * Dim.wz))

    g11 = Dim.a11 * 4 * math.pi / lx
    g22 = Dim.a22 * 4 * math.pi / lx
    g12 = Dim.a12 * 4 * math.pi / lx


