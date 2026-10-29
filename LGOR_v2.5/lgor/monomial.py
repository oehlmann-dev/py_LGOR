"""Oscillator monomials.

A monomial is a tuple of 4*n integers (n = number of fields)::

    (B_1..B_n, BB_1..BB_n, F_1..F_n, BF_1..BF_n)

holding the powers of the bosonic modes phi (B), phibar (BB), and the fermionic
modes psi (F), psibar (BF) of each field, acting on the twisted vacuum.  Negative
powers mean annihilation operators (they only occur inside Qbar).  This is the
same encoding as the Monomial class of the C++ code.
"""

from __future__ import annotations

from typing import Tuple

Mono = Tuple[int, ...]

B, BB, F, BF = 0, 1, 2, 3
KIND_NAMES = ("B", "bB", "F", "bF")          # names used by the C++ printout
KIND_NAMES_PHYS = ("phi", "phibar", "psi", "psibar")
KIND_NAMES_TEX = (r"\phi", r"\bar\phi", r"\psi", r"\bar\psi")


def zero(n: int) -> Mono:
    return (0,) * (4 * n)


def part(m: Mono, kind: int, n: int) -> Tuple[int, ...]:
    return m[kind * n:(kind + 1) * n]


def falling(n: int, k: int) -> int:
    """n (n-1) ... (n-k+1) ("choosesort" in the C++ code); 0 if k < 0."""
    if k < 0:
        return 0
    r = 1
    for x in range(n, n - k, -1):
        r *= x
    return r


def choose(n: int, k: int) -> int:
    """Binomial coefficient with the conventions of the C++ ``choose``."""
    if n < k:
        return 0
    if n < 0:
        n = -n
    if k < 0:
        k = -k
    r = 1
    i = 1
    while i <= k:
        r *= n
        r = int(r / i) if r < 0 else r // i
        i += 1
        n -= 1
    return r


def product(m1: Mono, m2: Mono, n: int):
    """Multiply two monomials: returns (coefficient, monomial).

    Direct port of Monomial::IsProductOf(m1, m2): bosonic annihilators acting on
    creators give falling-factorial factors; the fermion reordering sign follows
    the same ordering convention as the C++ code (fields ordered by index, psi
    before psibar).  States are acted on as ``product(state, qbar_term)``.
    """
    res = tuple(a + b for a, b in zip(m1, m2))
    factor = 1
    nsigns = 0
    f2f = 0
    for i in range(n):
        b1, b2 = m1[i], m2[i]
        if b1 < 0 < b2:
            factor *= falling(b2, -b1)
        if b2 < 0 < b1:
            factor *= falling(b1, -b2)
        bb1, bb2 = m1[n + i], m2[n + i]
        if bb1 < 0 < bb2:
            factor *= falling(bb2, -bb1)
        if bb2 < 0 < bb1:
            factor *= falling(bb1, -bb2)
        if m1[3 * n + i] != 0:
            nsigns += 1
        if m2[2 * n + i] != 0:
            f2f += 1
    nsigns *= f2f
    for i in range(1, n):
        f1i = m1[2 * n + i]
        bf1i = m1[3 * n + i]
        if not f1i and not bf1i:
            continue
        for j in range(i):
            if f1i and m2[2 * n + j]:
                nsigns += 1
            if bf1i and m2[3 * n + j]:
                nsigns += 1
    if nsigns % 2:
        factor = -factor
    return factor, res


def regular_nonzero(m: Mono, n: int) -> bool:
    for i in range(n):
        if m[i] < 0 or m[n + i] < 0:
            return False
        f, bf = m[2 * n + i], m[3 * n + i]
        if f < 0 or f > 1 or bf < 0 or bf > 1:
            return False
    return True


def holomorphic(m: Mono, n: int) -> bool:
    for i in range(n):
        if m[i] < 0 or m[2 * n + i] < 0 or m[n + i] > 0 or m[3 * n + i] > 0:
            return False
    return True
