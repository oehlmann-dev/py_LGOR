"""gtlines: group-theoretical (space-group / class-algebra) topological lines of an arbitrary LG orbifold, found from
2D data alone and checked on both sides of the BHK mirror.  LGOR 3.8; supersedes the 3.7 modules grouplines and
mirrorlines (kept as thin aliases).

Why: Verlinde-type lines (`tdl`, `lines`, `blocklines`) are diagonal on monomials and only see su(2)_k labels.  A
block LG_S / <J_S> that is a torus orbifold (or any block whose twisted ground states are permuted like fixed
points) also carries lines from translations that theta permutes; G = A x| <theta>, lines = irreps of G
(Bhardwaj-Tachikawa C(G, Z_m) after gauging the dual of the translations).  They mix monomials of equal charge.

Pipeline for a model (any .lgo; mirror built automatically for Fermat W):
 1. blocks (blocks.py): unions S of connected components of W with integer c_hat(S) in {1,2,3}; x^2 fields dropped.
    Non-Fermat components only get steps 2-3 [missing: non-Fermat TFT].
 2. 2D fixed-point table (fixedpoints.py): chi(k,l) from the J^a-twisted J^b-twined Witten indices (KYY summand,
    sign (-1)^{c(a+b)} fixed by T^2/Z_n Lefschetz tables [inferred]); valid for any quasi-homogeneous W.
    Non-positive rows are flipped (uniform fermion number) [inferred].
 3. candidates (groups.py): every abelian A of order chi(k*,0) with an automorphism sigma reproducing chi(k*, .).
 4. labels + TFT certification (tft.py): monomials of sector k = K q mod K; sectors with several classes get an
    unknown orbit basis, one per (sector, charge); all class-forbidden genus-0 couplings n = 3..nmax with total
    charge c_hat + n - 3 must vanish.  B picture: Saito flat coordinates; A picture: FJRW of narrow sectors
    (Chiodo / splitting / r-spin theorem [literature]).  Same solutions in both pictures under the Krawitz map.
 5. existence (pipeline.py): B-type in a model iff invariant under the gauged group; A-type iff every twist change
    it uses lies in the gauged group.  Mirror symmetry requires B(X) = A(Y) and A(X) = B(Y).
 6. optional (states.py): action on the 27 / 27bar of an LGOR run (c_hat = 1 blocks), X-27 = Y-27bar.

Statistics per candidate: class-forbidden couplings, those not zero by charge/phase, and those allowed by all
invertible (1-dim) lines of G ("line-only at block level").  The 4D tests (u1_forced, lattice) are separate.

Original 3.7 notes:
grouplines: group-theoretical (space-group / class-algebra) lines of c=3 Fermat sub-blocks, found from
2D data of the LG / Gepner phase alone, and how they sit in a full LG orbifold (original vs mirror).

Why: Verlinde-type lines (`tdl`, `lines`, `blocklines`) are diagonal on monomials and only see the
su(2)_k labels l_i.  A torus block LG(x^6+y^3) = T^2/Z6 also carries lines from torus translations that
theta permutes (fixed points of theta^3 form Z2^2, theta cycles three of them: A4 = Z2^2 x| Z3).  Such a
line mixes monomials of equal charge (x^3 and x y in the theta^3 sector) and is invisible to Verlinde
labels.  This module finds them without geometric input.

Pipeline, with tags:
 1. blocks: subsets S of fields of a Fermat W with c_hat(S) = 1 (x^2 fields dropped).  For Fermat
    exponents this gives (3,3,3), (4,4), (6,3); LG_S = T^2/Z_K at an enhanced point [literature].
 2. 2D fixed-point data [verified numerically, tau-independent]:
        chi(k,l) = (1/K) sum_{a,b} e^{2 pi i (k b - l a)/K} Z_{a,b}(z=0),
    Z_{a,b} = (-1)^{c(a+b)} e^{i pi c (a^2 tau + 2 a z)} prod_i theta1((1-q_i)(z+a tau+b))/theta1(q_i(z+a tau+b)),
    the J^a-twisted, J^b-twined Witten index (Kawai-Yamada-Yang summand).  The sign (-1)^{c(a+b)} is
    fixed by reproducing the T^2/Z_n Lefschetz tables [inferred]; with it the totals give 24 for (4)^4
    and (3)^6, 200 for the quintic, 168 for P11222[8].  chi(k, .) must be a permutation character of
    Z_K; its orbit decomposition = fixed points of theta^k and how theta permutes them.
 3. candidates: for each sector k* with a non-trivial orbit, every abelian group A of order chi(k*,0)
    with an automorphism sigma (theta^l -> sigma^l) reproducing chi(k*, .); G = A x| <sigma>.  Lines =
    irreps of G, acting on a state of class C by chi_rho(C) (class-algebra rule: a coupling is allowed
    iff the product of the class sums contains e).  dim rho > 1: non-invertible.
 4. state labels: sector of a chiral monomial k = K q mod K.  k = 0: [e] [assumption: untwisted chiral
    states carry no translation charge].  sigma^k != 1: class of (a, sigma^k), must be unique.
    sigma^k = 1, k != 0: the fixed set must again be A (as a Z_K-set), states <-> sigma-orbits on A in an
    unknown basis.
 5. TFT certification: all class-forbidden genus-0 block couplings (n = 3..nmax, flat coordinates,
    lgor.frobenius) must vanish; solved for the unknown orbit bases (invertible).  Solutions = lines
    consistent with the block TFT [verified up to nmax points; not a proof of existence].
 6. Verlinde comparison: is the line diagonal on monomials and equal to a product of su(2) lines?
 7. descent: each generator of the gauged group of a model, restricted to the block, acts on the
    solution set (fixes or permutes).  A solution fixed by all generators descends as an untwisted line
    [inferred].  Permuted solutions must reappear through twisted-sector objects of the gauged theory
    [missing: bimodule construction not implemented].
"""
from .fixedpoints import twined_index, fixed_point_table, euler_total, orbit_counts, table_signs, chat, block_order
from .groups import AbGroup, SpaceGroup, abelian_groups, candidate_actions
from .tft import (block_states, sector, charge, label_states, certify, fjrw_corr, chiodo4, line_on_monomials,
                  verlinde_match, act_on_solution)
from .blocks import Block, find_blocks, components, gauged_generators, group_closure, gauge_group
from .pipeline import analyse_block, analyse_model, summary, compare_states
