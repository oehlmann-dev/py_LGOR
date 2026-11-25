"""mfcat (LGOR 3.14): B-type defect lines of LG blocks as graded matrix factorisations of W(x) - W(x').

Build defects (identity, phase, permutation, any Koszul/ideal defect), direct sums, shifts and cones (bound states);
compute their bulk action on the Jacobian ring (= RR ground states), graded morphism spaces Hom(A,B), simplicity,
and test a candidate line against target eigenvalues and a fusion rule.  See lgor/mfcat/README.md.
"""
from .core import (LG, Jacobian, MF, koszul, solve_gs, ideal_defect, identity_defect, phase_defect, perm_defect,
                   parse_ideal, direct_sum, shift, cone)
from .hom import (bulk_action, raw_bulk_action, eigen_report, hom_dim, hom_dims, closed_reps, vec_to_morphism,
                  generic_morphism, is_simple)
from .lines import (det_phase, fusion_on_actions, compare_to_target, bound_state, a4_demo)
