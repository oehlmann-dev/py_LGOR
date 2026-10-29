# LGOR 2.5 (Python)

Massless spectrum of heterotic (2,2) Landau–Ginzburg orbifolds with standard
embedding: all twisted sectors, the explicit Q̄-cohomology states (oscillators
acting on the twisted vacua), their q₋/q₊ charges, and the assembly into E6
representations 1, 27, 27̄, 78.

This is a port of the C++ LGOR program used for Blaszczyk & Oehlmann,
JHEP04(2016)068. It follows the same algorithm (thesis chapter, section 3.1) and
fixes the bugs listed below. It needs only Python 3.8 or newer, with no extra packages
(numpy only for `blocklines` and the Rep(G) matching in `defects`).

## Features at a glance

| command | what it does | since |
|---|---|---|
| `run` | massless spectrum of an LG orbifold: all twisted sectors, Q̄-cohomology states, q₋/q₊ charges, E6 representations 1, 27, 27̄, 78 | 1 |
| `show`, `compare`, `check-lrt` | print a summary or one sector; group models with identical spectra; compare with old C++ `.lrt` files | 1 |
| `shell` | interactive shell with help for every command (`states`, `state`, `sector`, `summary`, ...) | 2 |
| `symmetries` (shell), summary | gauge group, U(1) and R charges, complete search for discrete (non-R) symmetries modulo gauge | 2 (2.2 fix) |
| `anomalies` (shell) | anomaly check of all symmetries | 2 |
| `deform`, `diff` | non-Fermat deformation; state-by-state comparison; VEV proposal with F- and D-flatness (Higgs-mechanism check) | 2 |
| `yukawa` (shell) | 27³ and 27³·1 couplings from the chiral ring (`lgor/yukawa.py`) | 1 |
| `mirror` | Greene–Plesser mirror of any Fermat model, with a spectrum check | 2 |
| `tdl` | Verlinde lines (TDLs) of Fermat models: labels, loci, 27³·1 selection rules, `--lattice` test against abelian explanations | 2 |
| `lines` | which deformations (complex structure, Kähler, blow-up moduli) keep which lines and products of lines | 2.2 |
| `defects` | general defect lines: dressed odd lines, self-duality (Tambara–Yamagami) defects with an exact torus test and an LGOR spectrum cross-check, group-theoretical lines (Rep(G), C(S_m,K)), α-induction and Σ Z², su(2)_k F-symbols and anomalies; every statement tagged `[verified]`/`[literature]`/`[inferred]`/`[missing]` | 2.3 |
| overlap warning | warns when orbifold generators overlap (LGOR would overcount) | 2 |
| non-Fermat W | `run` for any non-degenerate quasi-homogeneous W (Kazama–Suzuki, E7, D blocks, deformations): Jacobian-ring zero modes instead of the Fermat box; warning for an unreduced KOrder | 2.4 |
| `blocklines` | consistent lines of every block (minimal, Kazama–Suzuki CPⁿ_k, D, E7): commute with N=2, spectral flow, GSO/alignment currents, J; fusion type and whether it is new; reads saved results or run folders, recomputes nothing | 2.5 |

Each command is described in its own section below. Tests are in `tests/` (see Checks), and example
models are in `examples/` (with `examples/defects/run_all.sh` for the 2.3 checks).

## New in version 2.5: consistent lines of non-Fermat blocks (`python3 -m lgor blocklines PATH ...`)

`tdl`, `lines` and `defects` need Fermat blocks. `blocklines` handles every block type that `run`
accepts since 2.4. It recognises each block of W as a rational N=2 SCFT and keeps only the lines
that are consistent in the full Gepner model:

| block | W | RCFT | undressed consistent lines ("core") |
|---|---|---|---|
| A_k | x^(k+2) | su(2)_k ⊕ u(1) / u(1) | PSU(2)_k (even l); Fib for k = 3 |
| CPⁿ_k | p_d(e_1..e_n), d = n+k+1 | SU(n+1)_k × SO(2n)_1 / (SU(n)_(k+1) × U(1)) | N-ality-0 lines of su(n+1)_k ⊠ N-ality-0 lines of su(n)_(k+1) |
| D_(m+1), m even | x^m + x y² | su(2)_(2m−2), permutation invariant | PSU(2)_(2m−2) |
| D_(m+1), m odd | x^m + x y² | su(2)_(2m−2) extended by l = 2m−2 | Verlinde lines of the extension (fixed point split) |
| E7 | x³ + x y³ | automorphism invariant of the D10 extension of su(2)_16 | lines of the D10 extension |

**Checks, for every block and model** (the report tags each one):

1. *Commutes with N=2, spectral flow, the alignment/GSO currents and the J orbifold, and is well
   defined on the coset.* All of these operators are simple currents whose non-abelian labels are
   trivial up to field identification [literature: Kazama–Suzuki, Gepner]. A line of the non-abelian
   factors therefore passes iff its monodromy with every simple current of those factors vanishes.
   This is computed from the S-matrices: S_(L,J)/S_(0,J) = d_L for all J, and every line outside
   the core fails it [verified]. For CPⁿ_k this core is complete among undressed lines, since the
   identification current has charges a/(n+1) and b/n, and gcd(n, n+1) = 1.
2. *The modular data are consistent.* The su(N)_k S-matrix (Kac–Peterson, rows on demand) is unitary.
   The D-series extended S-matrix has a fixed-point resolution chosen so that S is unitary, S² is a
   permutation, and the fusion is integral and non-negative [verified].
3. *Central charge.* The sum of the block c/3 must be 3, and each block's c/3 must match the weights
   of its fields [verified].
4. *Fusion type.* The Verlinde fusion ring of every core factor is compared by backtracking ring
   isomorphism with PSU(2)_k, SU(2)_k and products of two of them, at any level. The result is:
   - `known` if the type and level occur among the 161 Fermat Gepner models,
   - `new level` if it is a known type at a level that does not occur there,
   - `NEW` if it is none of these.

   NEW rings are also compared with PSU(N)_k (N = 3..6) and with the D-series extensions to name
   coincidences, for example PSU(3)_4 = PSU(4)_3 = D8. Ring isomorphism is [verified]; equivalence
   of the categories (F-symbols) is [inferred].

**Not covered**, and listed as `not covered` in each report:
- lines of non-zero N-ality (they need a U(1)/fermion dressing),
- lines that commute with N=2 but not with the full W-algebra of a KS block,
- for D/E7 blocks, lines that preserve su(2)_k but not the extended algebra,
- whether the lines give selection rules beyond invertible symmetries, and which deformations keep
  them.

**Input.** The input can be a `.lgo` file, a saved `NAME.lgor.json` from any LGOR 2.x (2.2 and 2.4
files work: only the model is read), a run folder holding one model, or a folder of either. Nothing
is recomputed. Each model takes well under a second, except CP4_15 at about 1 s.

```
python3 -m lgor blocklines runs/K_CP2_4_D8_1_1_1               # full report, writes K_CP2_4_D8_1_1_1.blocklines.txt
python3 -m lgor blocklines runs/ --oneline -o blocklines/      # one line per model, reports into blocklines/
python3 -m lgor blocklines MODEL.lgor.json --no-write
```

A one-line example: `K_CP2_4_D8_1_1_1: CP2_4: PSU(3)_4 NEW (=PSU(4)_3, D_8 = su(2)_12/Z2) x PSU(2)_5; D8: su(2)_12/Z2 NEW (=PSU(3)_4, PSU(4)_3); (1): invertible only; ...`.
If a check fails, the line ends in `[CHECK FAILED]` and the exit status is 1.

**Calibration** (`tests/test_blocklines.py`):
- CP2_2 = D6, and both give Fib ⊠ Fib.
- Level-rank PSU(N)_k ≅ PSU(k)_N holds for five pairs.
- D4 is pointed (Z3), matching (1)(1).
- D8 = PSU(3)_4.
- Fermat blocks give PSU(2)_k, and the 135 Fermat survey run folders give no NEW type.

## New in version 2.4: non-Fermat superpotentials

`run` (also `deform`, `diff`) accepts any non-degenerate quasi-homogeneous W: each block of W whose fields are
all fixed in a sector gets a monomial basis of its Jacobian ring as zero modes (Fermat: the box a ≤ n−2);
blocks with only some fields fixed keep all window-allowed zero modes and the Koszul part of Q̄ does the
reduction. Fermat models give the same states as 2.3. A warning flags an unreduced KOrder (all R-charge
numerators sharing a factor g: an extra Z_g orbifold). `mirror`, `tdl`, `lines` and `defects` stop with a
message on non-Fermat models. Tests: `tests/test_nonfermat.py`; examples in `examples/nonfermat/`.

## New in version 2.3: general defect lines (`python3 -m lgor defects MODEL.lgo`)

`defects` (also a shell command) writes `MODEL.defects.txt`. It works on the
Gepner (RCFT) description of a Fermat model and goes beyond the Verlinde lines of `tdl`.
The file opens with a STATUS table, and every statement carries a tag:

| tag | meaning |
|---|---|
| `[verified]` | computed by the code in this run |
| `[literature]` | a standard result that is quoted, not recomputed |
| `[inferred]` | argued in the text, not checked by code |
| `[missing]` | not computed in this version (it says what is missing) |

If a section fails, it shows up as `[missing]` with the error, and the other sections still run.

| section | what it computes |
|---|---|
| 1 dressed lines | Which odd-label lines of even-k factors (e.g. Ising σ, su(2)₄ l=1) survive after dressing with abelian labels. The surviving parity patterns form a group. Also: the number of surviving line types, symmetric lines vs α⁺/α⁻ pairs, and the per-factor fusion. |
| 2 self-duality (Tambara–Yamagami) defects | Scans the subgroups H of the phase symmetries (and the quantum symmetries of the .lgo orbifold) for which C/H ≅ φ(C), with φ = identity or left charge conjugation ω, times permutations of identical factors. Test: the exact simple-current torus partition function (Kreuzer–Schellekens, NS sector, symmetric torsion), compared by Hermite normal form. Independent cross-check: LGOR computes the massless spectrum of C/H and compares it state by state with φ(C). Output: <D>=√\|H\|, and whether TY(H) can be group-theoretical (\|H\| a square or not). |
| 3 group-theoretical lines | (a) integral fusion subcategories generated by surviving lines, with the fusion ring compared to Rep(G) (Burnside characters, small-group library and products); (b) the permutation group S_m of identical factors, with C(S_m, K) fusion rings and tables. |
| 4 lines beyond the Verlinde lines | α-induction statement, fixed points of the current group, the number of simple defects Σ Z² (NS–NS). |
| 5 F-symbols and anomalies | su(2)_k quantum 6j symbols with pentagon check, Frobenius–Schur indicators, algebra objects (ADE), the surviving invertible phase lines with the N·h∈Z anomaly criterion, and the quantum lines of the orbifold. |

Options: `--sections 1,2`, `--no-lgor` (skip the LGOR spectrum cross-check), `--max-subgroups N`
(default 5000), `--max-perms N` (default 120), `-j N`. Python modules: `lgor/defects.py`,
`lgor/rcft.py` (N=2 minimal-model data), `lgor/zgroups.py` (integer lattices, HNF/Smith, subgroups),
`lgor/fsymbols.py`, `lgor/fingroups.py`. numpy is optional; without it, only the Rep(G) ring matching is skipped.

**Calibration.** The torus test is set up so that it reproduces the Greene–Plesser mirrors
of Q1Fermat2, G44441 and G66222 (Z_mirror(λ,μ) = Z(ωλ,μ)), and so that the identity fails for them
(unit tests). That LGOR's orbifolds carry exactly this symmetric torsion is `[inferred]`; for every
gauging it reports, the LGOR spectrum cross-check backs it up independently.

**Not done in 2.3 (all tagged `[missing]` in the output):**
- the TY data χ and τ;
- the F-symbols of dressed product lines;
- eigenvalues of dressed lines on the massless states;
- the spectrum of permutation orbifolds;
- general-subgroup and mixed phase/quantum gaugings;
- categorical (not just fusion-ring) equivalence with Rep(G);
- fixed-point resolution;
- subgroups beyond the cap.

**Examples** (`examples/defects/`, `sh examples/defects/run_all.sh`, with the outputs `*.defects.txt` and `*.log`):

| model | surviving non-inv. line types | TY defect (found) | LGOR check | invertible lines | time |
|---|---|---|---|---|---|
| Q1Fermat2 (quintic) | 31 (Fib⁵ minus 1) | \|H\|=125, with ω, unique | (101,1) → (1,101) ✓ | Z5³ phase, anomaly free | 4 s |
| Q1Mirror | 31 | \|H\|=125 by ungauging the quantum Z5³ | (1,101) → (101,1) ✓ | trivial phase + quantum Z5³ | 4 s |
| G44441 | 40 | \|H\|=108 | (103,1) → (1,103) ✓ | Z6³ | 3 s |
| G66222 | 63 (σ dressings: 16/32 patterns) | \|H\|=64 | (86,2) → (2,86) ✓ | Z2×Z4³ | 3 s |
| G4_2_2_2_2_1 | 23 | \|H\|=64 | (61,1) → (1,61) ✓ | Z2³×Z4² | 7 s |
| G2_2_2_2_2_2 | 31 | \|H\|=512 (subgroup scan capped) | (90,0) → (0,90) ✓ | Z2²×Z4⁴ | 61 s |
| G2_2_2_2_1_1_1 (K3×T² type) | 7 | \|H\|=48 | (21,21) → (21,21) ✓ | Z2×Z4×Z12 | 3 s |
| A19Z33Mirror20 | 0 | \|H\|=27 | (36,0) → (0,36) ✓ | Z3³ phase + quantum Z3² | 45 s |

In every Fermat example, the TY defect is the Greene–Plesser half gauging (H = the GP group, φ = ω).
Group-theoretical rings found: the l=2 line of su(2)₄ in G44441 generates the Rep(S3) ring, and the Ising
σσ products in G66222 generate a ring equal to Rep(D4) = Rep(Q8) = TY(Z2×Z2).
These are fusion-ring statements only; the F-symbols are `[missing]`.

## New in version 2

| feature | how |
|---|---|
| interactive shell with help for every command | `python3 -m lgor` or `python3 -m lgor MODEL.lgo`; inside: `help`, `help <command>`, `guide` |
| browse every state | `states 27`, `states 1 --R 6`, `state 27-001`, `sector 1 0 0`, `style modes/lgor/latex` |
| anomaly check of all symmetries | `anomalies` (shell), `MODEL.anomalies.txt`, also at the end of the summary |
| non-Fermat deformation + comparison + VEV proposal with F- and D-flatness | `deform 5 x1 x2 x3 x4 x5` (shell) or `python3 -m lgor deform MODEL.lgo "5 x1 x2 x3 x4 x5"` |
| Greene–Plesser mirror and mirror check (any Fermat W) | `mirror` (shell) or `python3 -m lgor mirror MODEL.lgo` |
| Verlinde lines (TDLs): labels, loci, 27³·1 selection rules | `tdl` (shell) or `python3 -m lgor tdl MODEL.lgo [--lattice]` |
| warning when orbifold generators overlap (LGOR would overcount) | printed by `run` |
| complete search for discrete (non-R) symmetries modulo gauge | in `symmetries` and in the summary ("Complete search …") |

### A session

```
$ python3 -m lgor examples/Q1Fermat2.lgo        # loads and computes
lgor[Q1Fermat2]> summary                        # 4D spectrum, gauge group, charge generators
lgor[Q1Fermat2]> states 27                      # every 27 with U(1), R, discrete charges
lgor[Q1Fermat2]> state 27-001                   # its 16 / 1 / 10 components as oscillators on twisted vacua
lgor[Q1Fermat2]> anomalies
lgor[Q1Fermat2]> deform 5 x1 x2 x3 x4 x5 as Q1_deformed
lgor[Q1_deformed]> compare                      # state by state: lost / kept / new, broken symmetries
lgor[Q1_deformed]> vev                          # VEV proposal, D-/F-flatness, Higgs-mechanism check
lgor[Q1_deformed]> save results/Q1_deformed
lgor[Q1_deformed]> use Q1Fermat2
lgor[Q1Fermat2]> mirror Q1Mirror
lgor[Q1Fermat2]> quit
```

Multiplet labels (`27-001`, `27bar-001`, `1-017`, `V-003`, `78-001`) are the same as
in the summary file. A session can be scripted: `python3 -m lgor MODEL.lgo < commands.txt`.

### Anomalies (`lgor/anomalies.py`)

Left-handed 4D fermions: chiral multiplets (27, 27bar, singlets, chiral 78) and gauginos
(E6, E8, extra). Checked:

- continuous: U(1)–gravity², U(1)–E6², U(1)–E8² and all cubic U(1)ₐU(1)_bU(1)_c traces, with the
  Cartan generators of non-Abelian extra factors included as U(1)s (so their cubic and
  mixed anomalies are covered). All must vanish.
- discrete (Ibáñez–Ross): for every orbifold Z_n, every discrete remnant and the
  R-symmetry Z_M^R, A_E6 = Σ 2T(r) q and A_E8 must agree mod N (Green–Schwarz universality),
  with 2T(27) = 6, 2T(78) = 24, 2T(248) = 60, chiral fermion R = R(Φ) + 3, gaugino R = −3.
- Not imposed: discrete gravitational and cubic anomalies (they depend on heavy states),
  discrete anomalies involving the non-Abelian extra factors.

All models tested (quintic, its deformations and mirror, the A₁⁹ models and their
mirrors) are anomaly free.

### Deformations (`lgor/deform.py`)

`deform TERMS` adds terms to W (they must have R-charge 1 and respect the discrete
symmetries), computes the deformed model completely and writes `NAME.deformation.txt`:

1. **Comparison**: spectrum before/after, broken gauge factors, discrete remnants of
   broken U(1)s, R-symmetry, and a state-by-state list of lost / surviving / new states
   per class of preserved quantum numbers.
2. **VEV proposal**: the deformation modulus is the worldsheet state Σᵢ ψᵢ ∂ᵢ(δW)|1⟩ in
   T(1; 0). It is projected onto the massless Fermat singlets, giving the 4D fields that
   get a VEV and the relative VEVs.
3. **D-flatness**: D_a = Σ Q_a |⟨Φ⟩|² for every extra U(1) and every Cartan of a
   non-Abelian factor, computed exactly, plus the gauge-invariant holomorphic monomial
   of the VEV fields.
4. **F-flatness**: (a) trilinear couplings with ≥ 2 VEV legs, (b) an all-orders
   argument from the R-symmetry when it applies, (c) an explicit scan of all
   symmetry-allowed terms X⟨Φ⟩ⁿ up to order 11.
5. **Higgs-mechanism check**: unbroken group from the roots orthogonal to the VEV
   weights, number of eaten singlets, generic mass-matrix rank, discrete remnant and R
   (possibly mixed with a gauge rotation), each compared with the worldsheet result.

Limitations: for non-Abelian extra factors D-flatness is checked for the Cartans only,
and the generic mass rank ignores Clebsch–Gordan relations between couplings, so it
is reported as an upper bound.

### Mirrors (`lgor/mirror.py`)

For W = Σ xᵢⁿ with a common prime n and Z_n orbifold groups, `mirror` builds the
Greene–Plesser mirror (dual group H° = {Q : Q·H = 0, ΣQ = 0}/⟨J⟩). For any other Fermat W
(mixed or non-prime exponents) it uses the Berglund–Hübsch–Krawitz dual group: the annihilator
of H in ∏ Z_{nᵢ}, with independent generators (LGOR sums over Z_KOrder × ∏ Z_N as a direct
product). Checked on (4)⁴(1), (6)²(2)³ and two Z4 orbifolds of it. It then computes the mirror and checks:
27 ↔ 27bar, singlets, gauge group, all states one by one under
(q₊, q₋, qⁱ) → (q₊, −q₋, −qⁱ), the complete discrete group modulo gauge on both sides,
and anomalies.

### Overlapping orbifold generators

LGOR sums over Z_KOrder × ∏ Z_N as if it were a direct product. If J and the listed
generators are not independent (example: Z8 acting as (x1, x2) → (ω x1, ω⁻¹ x2) on
x1⁸ + x2⁸ + x3⁴ + x4⁴ + x5⁴, whose 4th power equals J⁴), every group element is counted
several times and the spectrum is multiplied by that factor. `run` now prints a warning for
Fermat models (`mirror.generator_overlap`). Choose independent generators instead.

### Verlinde lines (`lgor/tdl.py`)

For W = Σ xᵢ^{nᵢ} the Gepner point is ∏ (kᵢ) with kᵢ = nᵢ − 2. `tdl` gives:

- **A. Labels.** The su(2)_k label lᵢ of every massless state, read off from its right-moving
  charges, since massless right-movers are R ground states: q₊ⁱ = (lᵢ+1)/(kᵢ+2) − 1/2 mod 1.
- **B. Line eigenvalues.** S_{r,l}/S_{0,l} for every line, with dimension
  d_r = sin((r+1)π/(k+2))/sin(π/(k+2)). For k = 3, r = 2 is the Fibonacci line.
- **C. Deformation loci.** For each non-invertible line, the orbifold-invariant marginal
  monomials it keeps, i.e. those with xᵢ exponent 0 or kᵢ.
- **D. 27³·1 couplings.** The flat-coordinate four-point functions ∂_d C_abc of the untwisted
  27s at the Fermat point (residue plus DVV contact terms). Each is compared with:
  - the weak fusion rule: per factor, the fusion of the four labels contains 0 or k;
  - the strict fusion rule: chiral primaries are (a,a,0) and the marginal descendant is (d,d,2),
    so the factor carrying the marginal operator must fuse to 0 and the others to k;
  - whether that factor carries an identity entry.
  In all models tried, a coupling vanishes iff the strict rule forbids it. Results:

  | model | zeros, weak rule | zeros, strict rule only | nonzero |
  |---|---|---|---|
  | quintic | 3180 | 0 | 6515 |
  | (4)⁴(1) | 1452 | 1588 | 7845 |
  | (6)²(2)³ | 458 | 1278 | 4047 |

- **E. Abelian lattice test** (`--lattice`, slow). It asks whether any abelian symmetry
  (R or non-R, continuous or discrete) could explain the zero couplings, using an integer-span
  test on the untwisted 27s and the singlets G_i⁻Φ_d.

Only untwisted 27s enter D and E, and only at first order in the modulus. Odd-r lines need
dressing by invertible phases on the other factors (Córdova–Rizi).

### Complete discrete symmetry search

For each prime p dividing 2K or an orbifold order, all Z_p charges of the form
a k0 + Σ b_j k_j + Σ cᵢ (D qⁱ) + d (2q₋) + (remnant charges) mod p are found that leave
all gauginos neutral and give the three components of every 27 and 27bar the same charge.
They are then counted modulo gauge transformations, meaning every element of the extra
gauge torus of order p (the dual of the U(1) charge lattice) and, for p = 3, the
E6 centre. Results: Fermat quintic and its mirror have none (all their Z5 phases lie inside
U(1)⁴); the non-Fermat quintic has Z5³; the SU(3)⁴ model and its mirror both have Z3².

### Fixed in version 2

- 27-plet assembly for non-Fermat models: the three components are now matched by
  their discrete-remnant charges too. Before, when no U(1) distinguished them,
  a 1₋₂ component could be listed under the wrong 27. Counts and the 16-component
  charges were always right.
- `python -m lgor deform/diff/shell` were sometimes mistaken for a model file name.

## Batch commands (no shell)

Run from this folder (the one containing `lgor/`):

```
python3 -m lgor run examples/Q1Fermat2.lgo          # one model
python3 -m lgor run path/to/folder                  # every .lgo in a folder (old 'a' command)
python3 -m lgor compare path/to/folder              # group identical spectra (old 'i' command)
python3 -m lgor show examples/Q1Fermat2.lgor.json --sector 1 --states --q-plus -1/2
python3 -m lgor show examples/A19Z33Mirror20.lgo --sector 1 0 0 --states --latex
python3 -m lgor check-lrt model.lgo model.lrt       # compare with an old C++ result file
python3 -m lgor diff Fermat.lgo Deformed.lgo        # which states/charges are lost, kept, new
python3 -m lgor deform model.lgo "5 x1 x2 x3 x4 x5" # deformed model + comparison + VEV/flatness report
python3 -m lgor mirror model.lgo                    # Greene-Plesser mirror + mirror check
```

`run` writes these files next to each input file (plus `MODEL.anomalies.txt`):

| file | contents |
|---|---|
| `MODEL.summary.txt` | 4D spectrum, extra gauge group, the charge generators, then **every 27 and 27̄ multiplet** (its U(1), R and discrete charges, and its three component states 16/1/10, matched across sectors by their charges), **every E6 singlet** and **every vector multiplet** with their charges and states, per-sector overview, warnings |
| `MODEL.states.txt` | every massless state as an oscillator polynomial on its twisted vacuum, with q₋, q₊, Cartan charges qⁱ, U(1) charges, R-charge and discrete charges, grouped by 4D multiplet |
| `MODEL.yukawa.txt` | all trilinear superpotential couplings allowed by the symmetries, by type (27³, 27̄³, 27·27̄·1, singlets only), with the number of candidates each symmetry removes and why forbidden ones fail |
| `MODEL.states.csv` | the same, one row per state (opens in Excel / pandas) |
| `MODEL.lgor.json` | everything, machine-readable (reload with `lgor.results.load`) |

It uses all CPU cores (`-j N` to change that), skips models that already have
results (`--force` to recompute), and only writes the files once the computation
has finished. Only q₊ = −3/2 (gauginos) and −1/2 (chiral fermions) are listed;
q₊ = +1/2, +3/2 are their CPT conjugates.

`--sector K0 T1 T2 ...` picks the sector T(k0; k1, k2, ...). With `--states` it prints
every state, e.g. `(-2 psi1[-2/3] psibar1[-1/3] + phi1[-1/6] phibar1[-5/6]) |1; 0, 0>`.
The numbers in brackets are mode numbers. `--lgor-names` prints the old
`B / bB / F / bF` names instead, and `--latex` prints LaTeX.

## From Python / Jupyter

```python
import sys; sys.path.insert(0, "/path/to/this/folder")
from lgor import Model

m = Model.from_file("examples/A19Z33Mirror20.lgo")
spec = m.compute(processes=4)
print(spec.summary())

sec = spec.sector(1, (0, 0))            # T(1; 0, 0)
sec.vacuum                              # energy, q-, q+, discrete charges
sec.oscillator_table()                  # mode numbers of phi, phibar, psi, psibar per field
for st in sec.states(q_minus=0, q_plus="-3/2"):
    print(st)                           # st.terms = [(coeff, monomial), ...]
```

Saved results load back with `lgor.results.load("X.lgor.json")`.

## Yukawa couplings (`lgor/yukawa.py`)

Selection rules only. A coupling ΦₐΦ_bΦ_c of chiral superfields is listed if it is:
- E6-invariant (27·27·27, 27̄·27̄·27̄, 27·27̄·1, 1·1·1);
- neutral under the extra U(1)s, the discrete symmetries and the Z_n remnants;
- such that R(a) + R(b) + R(c) = R(W) = −6 mod M.

For the Fermat quintic, the 566 allowed 27³ couplings coincide exactly with the
non-zero chiral-ring products x^a x^b x^c ∝ ∏xᵢ³, which serves as a check
(`tests/test_yukawa.py`). The size of each non-zero coupling still requires the CFT.

## Comparing two models (`python3 -m lgor diff A.lgo B.lgo`)

This writes `A_vs_B.diff.txt`, which contains:
- both 4D spectra side by side, and the symmetries of both models;
- a comparison of the states, class by class.

A class is a set of states with the same quantum numbers that survive the
deformation: sector, q₋, q₊, R, discrete charges, and the charges under the discrete
remnants of the broken U(1)s. Within a class, B's states are projected onto A's:
- what gets hit **survives**;
- the orthogonal rest of A is **lost**;
- B states with no component along A are **new**.

Lost states are expressed through A's broken U(1)s whenever possible ("gaugino of
U1_a", "Goldstone eaten by U1_a" = Σᵢ Q_a(sᵢ) sᵢ). For the quintic deformed by
5x₁x₂x₃x₄x₅, the result is: U(1)⁴ is Higgsed, 4 gauginos and 4 singlets of sector
T(1) are lost, and the surviving symmetric singlet Σsᵢ is the deformation modulus.
Everything else matches class by class, and Z₅³ remains.

**Discrete remnants.** For every model, the program also computes which discrete
subgroups of the Fermat-point U(1)s survive W: those exp(2πi c·q) under which every
Q̄ term is invariant (Smith normal form), modulo elements that act trivially on all
states. For the deformed quintic it finds Z₅³, generated by U1_1, U1_2, U1_3 mod 5;
U1_4 is dependent because the overall phase rotation acts trivially. States whose
representative mixes monomials with different qⁱ are marked "mixed". Their R and
remnant charges are still well defined, and that is checked.

## Symmetries and charges (`lgor/symmetries.py`)

Follows section 2.2 of the paper:

1. **Cartan charges** qⁱ, one per LG field (eq. 2.17): vacuum
   (αᵢ−1)(ν̃ᵢ+½) − αᵢ(νᵢ−½), plus +αᵢ for φ, −αᵢ for φ̄, αᵢ−1 for ψ and 1−αᵢ for ψ̄.
   Σᵢ qⁱ = q₋ for every state (this is checked).
2. **Extra U(1)s**: the combinations c·q that W conserves (all of them at the Fermat
   point) and under which every E6 gaugino is neutral. Gauginos at q₊ = −3/2, q₋ = 0
   with zero weight are the Cartan gauginos. Those with non-zero weight are roots;
   the program groups them into simple factors (α ~ β if α ± β is a root) and names
   them from (rank, number of roots). The automatic basis is aligned with these
   factors (`SU3a_H1`, `SU3a_H2`, …, then `U1_1`, …), and each generator is rescaled
   so that the charges of all states are coprime integers. For the quintic this gives
   U1_1 = 5(q2 − q1), and so on.
3. **R-symmetry**: Q_R = 3k₀ − 2q₋ + n·q mod 6·KOrder (eq. 2.15). The integer vector n
   is solved for exactly, so that every gaugino (E6 and extra) has Q_R = 3; among the
   solutions the shortest n is taken, which means the R-symmetry isn't mixed with
   Abelian U(1)s. The chiral superfield has R(Φ) = R(ψ) − 3 and R(W) = −6.
4. **Discrete symmetries**: Q_j = k_j + n_j·q mod n_j (eq. 2.18), with every gaugino neutral.

Automatic checks: all gauginos have R = 3 and discrete charge 0, and the three
components of every 27 and 27̄ (1, 16 and 10, which sit in different sectors) carry
identical U(1), R and discrete charges.

**Ambiguities.** n and n_j are only fixed up to symmetries under which every gaugino
is neutral: Abelian U(1)s, and the centres of E6 and of the SU(n) factors. To use a
particular convention, give the generators in the .lgo file (the old C++ program
ignores these lines):

```
U1 "strA" 1 1 1 1 1 1 -2 -2 -2    # coefficients of q1 ... qn (fractions allowed)
RCorrection 2 2 2 2 2 2 -16 2 2   # n in Q_R
DiscreteCorrection 1 2 -1 -1 ...  # symmetry number, then n_1
```

**Comparison with the paper (SU(3)⁴ model, section 4.1).**
- Gauge group: SU(3)⁴ is reproduced, and the automatic Cartans contain the paper's
  isospin operators.
- R-charges: Tables 4 and 5 are reproduced exactly (27-triplets (6, 6, 0) in every
  SU(3), singlets 216 × 6 and 108 × 0) in the convention R(Φ) = R(ψ) + 3.
  That convention would forbid the Fermat quintic's 27³ Yukawas, so the program uses
  R(Φ) = R(ψ) − 3. For KOrder = 3 the two differ by a uniform shift of 6, which no
  cubic coupling can see.
- The paper's own correction 2Σq_str (eq. 4.9) does not give R = 3 to the SU(3)_C
  roots in the k₀ = 1 sectors; 2(q_str^C + 3q_iso^C) does.
- Discrete generators: (4.10) gives the SU(3)_C and SU(3)_D roots non-zero charge
  under Z₃⁽¹⁾; (4.11) is fine. The program instead fits generators under which every
  gaugino is neutral, so its discrete charges differ from Table 4 by such a relabelling.

## Input format (.lgo)

The format is unchanged from the C++ version:

```
Name = "Q1Fermat2"
KOrder = 5              # alpha_i = q_i / KOrder
Order = 5 5             # orders of the discrete symmetries (may be empty)
Field "1" 1 1 0         # name, q_i, discrete charges (must sum to exactly 0)
SPotTerm 1 5 0 0 0 0    # coefficient, powers of all fields
```

## Bugs in the C++ version, fixed here

1. **Integer overflow in the linear algebra (serious).** `Matrix::DivideByRowGCD`
   starts its gcd at 1, so it never divides anything. The integers then grow during
   Gaussian elimination and overflow 32-bit `int` without any warning; compiling with
   `-fsanitize=signed-integer-overflow` shows it already for the deformed quintic.
   The cohomology states, and in larger models the counts, can come out wrong. Python
   integers cannot overflow.
2. **Zero-filled `.lrs` files.** Batch mode (`a`) wrote the `.lrs` summary *before*
   computing. A crashed or interrupted run left a zero-filled `.lrs` behind; later
   runs then skipped that model ("results already exist"), and `i` counted it as an
   empty spectrum.
3. **Loading results twice corrupted the spectrum.** Loading (`l`) after computing,
   or loading twice, added the loaded bunches to the old ones, with the bookkeeping
   indices pointing at the wrong bunch. That gives "27-plet inconsistency" messages
   and wrong counts.
4. **The `.lrt` header summary was all zeros** for models without discrete
   symmetries (`ScanAll` returned before filling it in).
5. **Truncation of bosonic zero modes.** `OrderHCF/2/q_i - 2` was used where
   `KOrder/q_i - 2` was meant. These agree only when every discrete order divides
   2·KOrder; for other orders, Q̄-exact states would have been counted.
6. Smaller issues: an off-by-one bounds check in `Matrix::entry`, and a hidden
   `cin` that made the program look frozen when a spectrum wasn't CPT-invariant.

## Checks

- `tests/test_basic.py`: the quintic (101, 1, 330 singlets, 4 extra U(1)s), its
  x₁x₂x₃x₄x₅ deformation (326 singlets, no extra vectors), and an A₁⁹/Z₃² mirror
  model, including the 9 Cartan gauginos of the paper's Appendix B and a JSON
  round trip.
- `tests/compare_cpp.py`: compares with C++ `.lrt` files space by space (same
  dimension and same span). Every dimension matches. The only differing states were
  C++ states that are not actually orthogonal to im Q̄, caused by bug 1.
- `tests/test_paper_table9.py`: 30 models from Table 9 of the paper, plus an
  independent check of n₂₇ − n₂₇̄ against Vafa's Euler-number formula. All 30 pass
  Vafa. 22 match the table. For the other 8, the original C++ gives the same answer
  as this port and Vafa's formula agrees with both, so the table appears to list them
  in the wrong rows. One table entry, (0,0,1,1,1,1,1,1,1), has charges summing to 7
  (not 0 mod 3) and is skipped.

- `tests/test_symmetries.py`: U(1)⁴ and R(27) = −2 for the quintic; SU(3)⁴ and the
  R-charges of Tables 4 and 5 for the mirror model.

- `tests/test_yukawa.py`: the 27³ couplings of the quintic agree with the chiral ring.
- `tests/test_v2.py`: anomalies, deformation parser, the quintic deformation (equal VEVs of
  the five moduli, D- and F-flat, U(1)⁴ → Z5³) and the SU(3)⁴ deformations of Table 6
  (264 / 252 singlets, SU(3) × U(1)⁴ and SU(3)), predicted = worldsheet.

- `tests/test_tdl.py`: line dimensions, the quintic labels and 50-dimensional Fibonacci loci,
  the coupling counts of the quintic, (4)⁴(1) and (6)²(2)³ (a coupling vanishes iff the strict
  fusion rule forbids it), the general mirror (|H|·|H^∨| = ∏ nᵢ, independent generators, the same
  group as the F₅ construction for the quintic) and the overlap warning.

- `tests/test_blocklines.py` (2.5): unitarity of su(N)_k S-matrices, level-rank ring isomorphisms, D4 pointed,
  CP2_2 = D6 = Fib ⊠ Fib, D8 = PSU(3)_4, fusion-type labels, five models, and reading a saved `.lgor.json` and a run folder.
- `tests/test_nonfermat.py` (2.4): Jacobian-ring dimensions (Gr(2,6), E7, D8, Gr(3,9)), a degenerate block,
  D4(1)⁷ = (1)⁹, five Kazama–Suzuki/E7/D models against the generalised Vafa formula, and the unreduced-KOrder warning.
- `tests/test_defects.py` (2.3): the N=2 monodromy formula against conformal weights, the HNF/Smith/subgroup
  tools (Z5³ has 64 subgroups), the su(2)_k pentagon and Frobenius–Schur indicators, Fibonacci F = 1/φ, the anomaly of
  the su(2)_k Z2, C(S3, Z2) = Rep(S3) dimensions and the S4 character table, the Greene–Plesser mirrors of Q1Fermat2,
  G44441 and G66222 as Z(ωλ, μ), the uniqueness of the quintic's Z5³ self-duality defect, the 31 surviving quintic
  lines, the Rep(S3) ring of G44441, and the mirror quintic's quantum Z5³.

Run them with `python3 tests/test_basic.py`, `python3 tests/test_symmetries.py`, `python3 tests/test_yukawa.py`,
`python3 tests/test_v2.py`, `python3 tests/test_nonfermat.py`, `python3 tests/test_blocklines.py` and `python3 tests/test_paper_table9.py`
(or `python3 -m pytest tests` if pytest is installed).

## Layout

| file | contents | C++ origin |
|---|---|---|
| `lgor/model.py` | .lgo parser, consistency checks | ModelData, Field, Parseline |
| `lgor/sector.py` | twists, vacuum, state enumeration, Q̄, cohomology | Sector, SField, Complex |
| `lgor/monomial.py` | oscillator monomials, products and signs | Monomial |
| `lgor/linalg.py` | exact kernels | Matrix |
| `lgor/spectrum.py` | bunches, E6 assembly, model summary | Bunch, SpectrumSum, ModelSum |
| `lgor/results.py` | JSON results | (replaces .lrt/.lrs) |
| `lgor/symmetries.py` | Cartan / U(1) / R / discrete charges, gauge group | new (paper sec. 2.2) |
| `lgor/report.py` | summary, states and CSV files | new |
| `lgor/compare.py` | comparison of two models (`diff`) | new |
| `lgor/yukawa.py` | symmetry-allowed trilinear couplings | new |
| `lgor/legacy.py` | readers for old .lrt/.lrs | |
| `lgor/anomalies.py` | anomaly check | new (v2) |
| `lgor/deform.py` | deformations, VEV proposal, D-/F-flatness, Higgs check | new (v2) |
| `lgor/mirror.py` | Greene–Plesser / BHK mirror, mirror check, generator-overlap check | new (v2) |
| `lgor/tdl.py` | Verlinde lines: labels, loci, flat 4-point selection rules, abelian lattice test | new (v2.1) |
| `lgor/blocklines.py` | consistent lines of minimal, Kazama–Suzuki, D and E7 blocks; su(N)_k S-matrices, D extensions, fusion-ring isomorphism | new (v2.5) |
| `lgor/shell.py` | interactive shell | new (v2) |
| `lgor/cli.py` | command line | Lgor, main |

Not ported (yet): the A₁⁹ model generator (`z` command, A19Models.cpp), and the
moduli identification `Sector::IdentifySinglets`, which the C++ code never called.
