# mfcat — defect lines of LG blocks as matrix factorisations (LGOR 3.14)

B-type topological defect lines of an LG block W(x) are graded matrix factorisations (MFs) Q of W(x) − W(x′).
This package builds them and computes what LGOR needs to compare them with the lines found elsewhere in LGOR
(Verlinde `tdl`/`lines`, `gtlines` group-theoretical lines).

## What it computes

| object | function | notes |
|---|---|---|
| block | `LG.fermat(ns)`, `LG.from_string(W, names)`, `LG(W, q)` | any quasi-homogeneous W with rational coefficients |
| Jacobian ring, residue | `lg.jac()` (`basis`, `nf`, `res`) | numerical linear algebra per charge; Res[Hess] = μ |
| defects | `identity_defect`, `phase_defect`, `perm_defect`, `ideal_defect(lg, fs)` (Koszul; the g_i are solved for), `parse_ideal(lg, ["x1^2 - x4p", ...])` | primed variables are `<name>p` |
| operations | `direct_sum`, `shift`, `cone` (even morphism, K-class [B]−[A]), `odd_cone` (odd morphism, [A]+[B]) | grading: x_i has charge 2 q_i D, W has 2D, Q has D |
| bulk action | `bulk_action(X)` | on RR ground states = Jacobian ring, Kapustin–Li type formula, normalised by the identity defect; additive on cones |
| morphisms | `hom_dim`, `hom_dims(A,B)`, `closed_reps`, `generic_morphism` | graded cohomology of Hom(A,B) by linear algebra |
| simplicity | `is_simple(X)` | End⁰ in charge 0 is one-dimensional |
| realiser | `library(lg)`, `realise(target, lib)`, `realise_fast`, `realise_bound_state`, `gtlines_targets(ns)`, `realise_block(ns, search=True)` | integer combinations of library defects matching a target bulk action, then glued into a bound state |
| Fourier defects | `search.rotation_defect`, `search.fourier_library` | quadric pairs x_i^{2a}+x_j^{2b} = U²+V²: ideal (U − R U′, V − R V′) with R ∈ O(2); R = Hadamard is the fixed-point Fourier transform |
| rank-one / Verlinde | `search.rank_one_defect(lg, Ss)`, `search.rank_one_library` | ⊗_i P_{S_i}(x_i, x_i′) = ∏_{a∈S}(x_i − η^a x_i′) (Brunner–Roggenkamp); RR action ∏ Σ_{a∈S} η^{−a(j+1)} |
| general MF search | `search.mf_search(lg, deg_even, deg_odd)`, `search.koszul_gradings(lg, nf)`, `search.search_components` | any rank and grading, arbitrary homogeneous entries; Q² = W − W′ solved by Levenberg–Marquardt from random starts; components labelled by bulk action |
| closure | `search.twist(X, g, h)`, `search.twisted_closure`, `search.conjugate` | X(g x, h x′) and complex conjugates; `lines.extended_library(lg)` = library + Fourier + rank-one + search, closed |

The phase defect of g acts on RR ground states as det(g)⁻¹·g⁻¹, so the NS-normalised symmetry operator is
det(g)·(bulk action) (`det_phase`).

Power-swap defects: if W ⊃ x_i^{ap} + x_j^{bp} = U^p + V^p with U = x_i^a and V = x_j^b of equal charge, the Koszul MF on
(x_i^a − c x_j′^b, x_j^b − c⁻¹ x_i′^a), with c^p = 1, is the involution U ↔ V of the p-ic. These defects are not
invertible. They act only on states that are odd under x_i → −x_i (for a = 2).

## CLI

```
python -m lgor mfcat a4demo                     # Z6-II block x1^6 + x4^3: the three A4 lines as bound states
python -m lgor mfcat realise 6,3 4,4 --max-terms 4
python -m lgor mfcat defect --W "x1^6 + x4^3" --names x1,x4 --ideal "x1^2 - x4p; x4 - x1p^2" --hom
python -m lgor mfcat defect --W "x1^6 + x4^3" --names x1,x4 --phase 1/6,1/3
```

## Extended finder (2026-11-25)

`realise_block(ns, search=True)` uses `extended_library` (phases, swaps, perm∘phase, Fourier, rank-one tensor defects,
general MF-search components of rank 2, all phase twists and conjugates).

* **(6,3): 7/7 gtlines lines realised as simple bound states** [verified: Q² = W − W′, End⁰₀ = ℂ, bulk action = target].
  The three non-Verlinde S3 lines are Cone(M_h → 1), where M is a rank-2 MF of grading (0,0)|(−2,2) (Koszul charges 4, 8)
  that is *not* a Koszul defect, and h is a phase twist of x1. Vacuum eigenvalue 2. They act on the x1-even states, which
  every swap/pull-back defect misses. The A4 lines are found again with the new components.
* **(4,4): 16/24 realised.**
  * 12 match the target directly.
  * 4 (sol 0, 1, 20, 21) match only after an R-type dressing of the target, A·D_θ with θ = (1/8, 3/8) (W → −W). They are
    realised by the same objects as sol 2/3. Reading: these NS-picture lines act on the spectral-flow operator with a
    phase [inferred].
  * 8 (sol 4–7, 13–16; diagonal (1, −2, 1) on charge 1/2) are not realised, even with rank-4 search components. Lattice
    reduction (LLL) finds **2×** their bulk action in the lattice of library actions but not the action itself. This
    index-2 obstruction fits objects that exist only in the orbifold (equivariant MFs that split into J-eigen summands)
    [inferred]. The (4,4) Fourier defects alone act only on x1x2 (±4, independent of the angle).
* Use `realise_fast` (hash tables, ≤ 4 terms) for libraries with more than 40 entries.

## Results so far (2026-11-24)

* **(6,3) block of Z6-II.**
  * The three gtlines A4 lines are V_ζ = Cone(J³ ⊕ S_a → J² ⊕ J⁴), with J = J_B = (1/6, 1/3), S_a the power swap x1² ↔ x4 and a = ζ⁻¹.
  * Each V_ζ satisfies Q² = W − W′, is simple, and has bulk action equal to the gtlines A4 line. Its vacuum eigenvalue is 3, and V² = 1 + 1′ + 1″ + 2V and V·1′ = V hold on the bulk action. [verified]
  * The Verlinde line L_2 is realised as a simple iterated odd cone of 1, g(1/6, 0), g(5/6, 0) [verified], in agreement with the Brunner–Roggenkamp construction of Verlinde MFs [literature].
  * The three non-Verlinde S3 lines are not realised by this library [open]. They mix x4 with x1² on even-x1 states, and every power swap vanishes there.
* **(4,4).** The D4 gtlines lines (Fourier-type mixing) were not realised with ≤ 4 phase or power-swap terms. A run with permutation∘phase defects is pending.
* **(3,3,3).** gtlines finds no lines, so there is nothing to realise.

## Caveats

* Bulk actions are computed on the block chiral ring only, not on twisted sectors of an orbifold.
* `is_simple` tests End⁰₀ = ℂ in the graded MF category of the plain block LG.
* Fusion is checked on bulk actions, which are not faithful. Object-level fusion, i.e. tensor products of MFs, is not implemented. In the plain block, V⊗1′ ≇ V as objects: the K-classes differ, but the bulk actions agree. The Rep(A4) relations of objects need the Gepner-orbifold identifications [open].
* Cones are built from a generic morphism. For V the morphism space is 4-dimensional with one rescaling-invariant cross-ratio, so the result is a one-parameter family of simple objects with the same bulk action.
* Numerics are in complex floating point, with rank tolerance 1e-8.
