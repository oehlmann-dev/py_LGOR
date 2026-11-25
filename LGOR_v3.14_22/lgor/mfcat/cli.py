"""CLI for `python -m lgor mfcat ...` (LGOR 3.14)."""
import numpy as np


def cmd_mfcat(args):
    from .core import LG, parse_ideal, ideal_defect, phase_defect
    from .hom import bulk_action, eigen_report, hom_dims, is_simple
    from .lines import realise_block, a4_demo, library
    from .core import identity_defect
    out = []
    if args.mode == "realise":
        for b in args.blocks:
            ns = tuple(int(x) for x in b.split(","))
            lines = realise_block(ns, nmax=args.nmax, max_terms=args.max_terms, verbose=True)
            out.append((ns, lines))
    elif args.mode == "a4demo":
        for z in range(3):
            print("A4 line zeta index %d (Z6-II block x1^6 + x4^3):" % z)
            a4_demo(z)
    elif args.mode == "defect":
        names = args.names.split(",")
        lg = LG.from_string(args.W.replace("^", "**"), names)
        if args.ideal:
            X = ideal_defect(lg, parse_ideal(lg, [g.strip() for g in args.ideal.split(";")]), args.ideal)
        else:
            from fractions import Fraction as Fr
            X = phase_defect(lg, [Fr(x) for x in args.phase.split(",")])
        print("W = %s, charges %s, Jacobian basis %s" % (args.W, [str(q) for q in lg.q], lg.jac().basis))
        print("defect %s: Q^2 = W - W': %s" % (X.name, X.check()))
        A = bulk_action(X)
        np.set_printoptions(precision=4, suppress=True, linewidth=160)
        print("bulk action (columns = images of the Jacobian basis, RR normalisation):")
        print(A)
        for ev, vec in eigen_report(A, lg.jac().basis, names):
            print("  eigenvalue %s : %s" % (np.round(ev, 6), vec))
        print("simple (End^0_0 = C): %s" % is_simple(X))
        if args.hom:
            print("Hom(1, X) dims {(charge, parity): dim}: %s" % hom_dims(identity_defect(lg), X))
            print("End(X)   dims {(charge, parity): dim}: %s" % hom_dims(X, X))
    return 0


def add_parser(sub):
    mf = sub.add_parser("mfcat", help="3.14: B-type defect lines of LG blocks as graded matrix factorisations: bulk "
                        "action, Hom spaces, bound states (cones); realise the gtlines lines of Fermat blocks")
    mf.add_argument("mode", choices=["realise", "a4demo", "defect"],
                    help="realise BLOCK...: build every certified gtlines line of each Fermat block from phase/"
                         "permutation/power-swap defects and bound states; a4demo: Z6-II (6,3) A4 lines; "
                         "defect: one Koszul defect given by --W --names and --ideal or --phase")
    mf.add_argument("blocks", nargs="*", help="realise: Fermat exponents, e.g. 6,3 4,4")
    mf.add_argument("--nmax", type=int, default=5)
    mf.add_argument("--max-terms", type=int, default=4)
    mf.add_argument("--W", help="defect: superpotential, e.g. 'x1^6 + x4^3'")
    mf.add_argument("--names", help="defect: comma separated variable names, e.g. x1,x4 (primed: x1p, x4p)")
    mf.add_argument("--ideal", help="defect: ideal generators separated by ';', e.g. 'x1^2 - x4p; x4 - x1p^2'")
    mf.add_argument("--phase", help="defect: phases of x_i -> e^{2 pi i p_i} x_i, e.g. 1/6,1/3")
    mf.add_argument("--hom", action="store_true", help="defect: also print Hom(1,X) and End(X) dimensions")
    mf.set_defaults(func=cmd_mfcat)
    return mf
