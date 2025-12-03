#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## utils.py
##
"""
    Utilities for explanation techniques

    =================
    List of functions
    =================

    .. autosummary::
        :nosignatures:

        make_assump_model
        is_normalised_pb
        get_slack
        is_false
        get_degree_over_sum
        
"""

import copy
import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.utils import is_any_list
from cpmpy.expressions.variables import NegBoolView
from cpmpy.transformations.normalize import toplevel_list

def make_assump_model(soft, hard=[], name=None):
    """
        Construct implied version of all soft constraints.
        Can be used to extract cores (see :func:`tools.mus() <cpmpy.tools.explain.mus.mus>`).
        Provide name for assumption variables with `name` param.
    """
    # ensure toplevel list
    soft2 = toplevel_list(soft, merge_and=False)

    # make assumption variables
    assump = cp.boolvar(shape=(len(soft2),), name=name)

    # hard + implied soft constraints
    hard = toplevel_list(hard)
    model = cp.Model(hard + [assump.implies(soft2)])  # each assumption variable implies a candidate

    return model, soft2, assump

# @profile
def is_normalised_pb(pb_expr):
    """
        Check if a pseudo-Boolean expression is normalised.
        A pseudo-Boolean expression is normalised if it is of the form (wsum of literals) >= degree (int).
        The degree should be a positive integer and the coefficients should be non-negative.

        :param: pb_expr: pseudo-Boolean expression (wsum of literals) >= degree (int)

        Returns True if the pseudo-Boolean expression is normalised, False otherwise.
    """
    if isinstance(pb_expr, cp.expressions.variables._BoolVarImpl):
        return True
    if not isinstance(pb_expr, cp.expressions.core.Comparison):
        return False
    if pb_expr.name != ">=":
        return False
    lhs = pb_expr.args[0]
    rhs = pb_expr.args[1]
    if not isinstance(rhs, int) or rhs <= 0:
        return False
    if not isinstance(lhs, cp.expressions.core.Operator) or (lhs.name != "wsum" and lhs.name != "sum"):
        return False
    if lhs.name == "wsum" and any([coef < 0 for coef in lhs.args[0]]):
        return False
    return True

# @profile
def get_slack(pb_expr):
    """ 
        Get the slack of a normalised pseudo-Boolean constraint.
        The slack is defined as the difference between the sum of the coefficients of the non-falsified literals and the degree (righthandside) of the constraint.

        :param: pb_expr: pseudo-Boolean expression (wsum of literals) >= degree (int)

        Returns the slack as an integer.
    """
    if not is_normalised_pb(pb_expr):
        raise ValueError(f"{pb_expr} is not a normalised pseudo-Boolean expression")
    if isinstance(pb_expr, cp.expressions.variables._BoolVarImpl):
        return int(not is_false(pb_expr))
    lhs = pb_expr.args[0]
    rhs = pb_expr.args[1]
    slack = -rhs # start with the degree
    if lhs.name == "wsum":
        for i in range(len(lhs.args[0])):
            if not is_false(lhs.args[1][i]):
                slack += lhs.args[0][i]
    elif lhs.name == "sum":
        for var in lhs.args:
            if not is_false(var):
                slack += 1
    return slack

def slack_under(pb_expr, const):
    """ 
        Check if the slack of a sorted normalised pseudo-Boolean constraint is under a given constant.
        Only for constraints with positive slack.

        :param: pb_expr: pseudo-Boolean expression (wsum of literals) >= degree (int)
        :param: const: constant to compare the slack against (int)

        Returns A tuple (bool, bool) where the first element is True if the slack is under the constant, False otherwise,
        and the second element is True if the slack is less than zero
    """
    # if get_slack(pb_expr, dec_vars) < 0:
    #     print(f"Warning: {pb_expr} has negative slack {get_slack(pb_expr, dec_vars)}")
    #     raise ValueError(f"slack_under only works for constraints with positive slack")
    
    # if not is_normalised_pb(pb_expr):
    #     raise ValueError(f"{pb_expr} is not a normalised pseudo-Boolean expression")
    lhs = pb_expr.args[0]
    rhs = pb_expr.args[1]
    slack = -rhs # start with the degree
    if lhs.name == "wsum":
        for i in range(len(lhs.args[0])):
            if not is_false(lhs.args[1][i]):
                slack += lhs.args[0][i]
                if slack >= const:
                    return False
    elif lhs.name == "sum":
        for var in lhs.args:
            if not is_false(var):
                slack += 1
                if slack >= const:
                    return False
    return True

# @profile
def is_false(literal):
    """
        Check if a boolean literal is false in the current assignment of the decision variables.
        
        :param: literal: a literal (int or boolvar)
    """
    return literal.value() is False

def get_degree_over_sum(pb_expr):
    """
        Get the degree of a pseudo-Boolean expression over the sum of its literals.
        The degree is defined as the sum of the coefficients of the non-falsified literals.

        :param: pb_expr: pseudo-Boolean expression (wsum of literals) >= degree (int)

        Returns the degree as an integer.
    """
    # if not is_normalised_pb(pb_expr):
    #     raise ValueError(f"{pb_expr} is not a normalised pseudo-Boolean expression")
    lhs = pb_expr.args[0]
    rhs = pb_expr.args[1]
    
    # print(pb_expr)
    
    sum_coefs = 0
    if lhs.name == "wsum":
        for i in range(len(lhs.args[0])):
            sum_coefs += lhs.args[0][i]
    elif lhs.name == "sum":
        sum_coefs = len(lhs.args)
    
    assert sum_coefs >= rhs, f"The constraint {pb_expr} can never be satisfied. This should trivially be detected as UNSAT. (or this is a trivial MUS by itself)"
    return rhs / sum_coefs

def get_max_sat(pb_expr):
    """
        Get the maximum amount of literals that can be satisfied in a pseudo-Boolean expression, without actually satisfying it.

        :param: pb_expr: pseudo-Boolean expression (wsum of literals) >= degree (int)
    """
    if not is_normalised_pb(pb_expr):
        raise ValueError(f"{pb_expr} is not a normalised pseudo-Boolean expression")
    lhs = pb_expr.args[0]
    rhs = pb_expr.args[1]
    
    max_sat = 0
    if lhs.name == "wsum":
        coefs = sorted(lhs.args[0], reverse=True)
        for c in coefs:
            if c < rhs:
                rhs -= c
                max_sat += 1
            else:
                return max_sat
    elif lhs.name == "sum":
        return rhs - 1

def get_min_sat(pb_expr):
    """
        Get the minimum amount of literals that need to be satisfied in a pseudo-Boolean expression, in order such that it could be satisfied.

        :param: pb_expr: pseudo-Boolean expression (wsum of literals) >= degree (int)
    """
    if not is_normalised_pb(pb_expr):
        raise ValueError(f"{pb_expr} is not a normalised pseudo-Boolean expression")
    lhs = pb_expr.args[0]
    rhs = pb_expr.args[1]
    
    min_sat = 0
    if lhs.name == "wsum":
        coefs = sorted(lhs.args[0])
        for c in coefs:
            if c < rhs:
                rhs -= c
                min_sat += 1
            else:
                return min_sat
    elif lhs.name == "sum":
        return rhs

# @profile
def get_length(pb_expr):
    """
        Get the length of a pseudo-Boolean expression.

        :param: pb_expr: pseudo-Boolean expression (wsum of literals) >= degree (int)
    """
    if not is_normalised_pb(pb_expr):
        raise ValueError(f"{pb_expr} is not a normalised pseudo-Boolean expression")
    lhs = pb_expr.args[0]
    
    if lhs.name == "wsum":
        length = len(lhs.args[0])
    elif lhs.name == "sum":
        length = len(lhs.args)
    
    return length

def get_length_gen(expr):
    return len(get_variables(expr))

# @profile
def get_coefficient_lit(pb_expr, literal):
    """
        Get the coefficient of a literal in a pseudo-Boolean expression.

        :param: pb_expr: pseudo-Boolean expression (wsum of literals) >= degree (int)
        :param: literal: a literal (boolvar)
    """
    if not is_normalised_pb(pb_expr):
        raise ValueError(f"{pb_expr} is not a normalised pseudo-Boolean expression")
    if isinstance(pb_expr, cp.expressions.variables._BoolVarImpl):
        if pb_expr == literal:
            return 1
        else:
            return 0
    
    lhs = pb_expr.args[0]
    
    if lhs.name == "wsum":
        for i in range(len(lhs.args[0])):
            if lhs.args[1][i].name == literal.name:
                return lhs.args[0][i]
    elif lhs.name == "sum":
        for i in range(len(lhs.args)):
            if lhs.args[i].name == literal.name:
                return 1
    return 0

def get_coefficient_lit_linear(cpm_expr, literal):
    """
        Get the coefficient of a literal in a pseudo-Boolean expression.

        :param: pb_expr: pseudo-Boolean expression (wsum of literals) >= degree (int)
        :param: literal: a literal (boolvar)
    """
    if isinstance(cpm_expr, cp.expressions.variables._BoolVarImpl):
        if cpm_expr == literal:
            return 1
        else:
            return 0
    
    lhs = cpm_expr.args[0]
    
    if lhs.name == "wsum":
        for i in range(len(lhs.args[0])):
            if lhs.args[1][i].name == literal.name:
                return lhs.args[0][i]
    elif lhs.name == "sum":
        for i in range(len(lhs.args)):
            if lhs.args[i].name == literal.name:
                return 1
    return 0

# @profile
def get_coefficient_var(pb_expr, var):
    """
        Get the coefficient of a variable in a pseudo-Boolean expression.

        :param: pb_expr: pseudo-Boolean expression (wsum of literals) >= degree (int)
        :param: var: a variable (boolvar)
    """
    if not is_normalised_pb(pb_expr):
        raise ValueError(f"{pb_expr} is not a normalised pseudo-Boolean expression")
    if isinstance(pb_expr, cp.expressions.variables._BoolVarImpl):
        if var == pb_expr:
            return 1
        else:
            return 0
    
    lhs = pb_expr.args[0]
    
    if lhs.name == "wsum":
        for i in range(len(lhs.args[0])):
            if lhs.args[1][i] is var:
                return lhs.args[0][i]
    elif lhs.name == "sum":
        for i in range(len(lhs.args)):
            if lhs.args[i] is var:
                return 1
    return 0

# @profile
def get_lits(pb_expr):
    """
        Get the literals in a pseudo-Boolean expression.

        :param: pb_expr: pseudo-Boolean expression (wsum of literals) >= degree (int)
    """
    # if not is_normalised_pb(pb_expr):
    #     raise ValueError(f"{pb_expr} is not a normalised pseudo-Boolean expression")
    if isinstance(pb_expr, cp.expressions.variables._BoolVarImpl):
       return [pb_expr] 
    
    lhs = pb_expr.args[0]
    
    if lhs.name == "wsum":
        return lhs.args[1]
    elif lhs.name == "sum":
        return lhs.args
    return []

def flip_literal(lit):
    """
        Flip the value of a literal.

        :param: lit: a literal (int or boolvar)
    """
    if isinstance(lit, cp.expressions.variables.NegBoolView):
        lit._bv._value = not lit._bv.value()
    else:
        lit._value = not lit.value()
        
def get_bounds(constraint, lit, coefficient):
    
    if isinstance(constraint, cp.expressions.variables._BoolVarImpl):
        return 0, 1
    
    
    comp = constraint.name
    lhs, rhs = constraint.args
    
    lhs_val = lhs.value()
    
    return lit.lb, lit.ub
    
    

# @profile
def rotate_model_old(constraints, constraint, depth=None, recursive=True, found=set(), hard=[]):
    
    # print(f"constraint to rotate on: {constraint} at depth {depth}")
    # print(f"Rotating model at depth {depth}")
    # for constraint_check in constraints:
    #     if constraint_check is constraint:
    #         continue
    #     assert get_slack(constraint_check, assoc) >= 0, f"Constraint {constraint_check} has negative slack {get_slack(constraint_check, assoc)}"
    if depth == 0:
        return
    lits = get_lits(constraint)
    slack = get_slack(constraint)
    # print(f"Rotating over constraint: {constraint} with slack {slack} and lits {lits}")
    # print(f"Current model: {[var.name + '=' + str(var.value()) for var in assoc]}")
    
    under_slack = True # flag to flip the literals with highest coeffs until we reach the slack.
    flipped_lits = set()
    
    for lit in lits:
        if not is_false(lit):
            continue
        c = get_coefficient_lit(constraint, lit)
        count_neg = 0
        if c >= -slack:
            under_slack = False
            count = 0
            count_neg = 0
            
            bad_rot = False # flag to avoid bad rotations
            # loop over constraints in model, if only one will become false then add it to found and rotate recursively
            for constraint_check in constraints:
                if constraint_check is constraint:
                    continue
                neg_lit = ~lit
                # slack_other = get_slack(constraint_check, assoc)
                under_const = slack_under(constraint_check, get_coefficient_lit(constraint_check, neg_lit))
                if under_const:
                    count += 1
                    last = constraint_check
                    if constraint_check in hard:
                        bad_rot = True
                        break
                    if constraint_check in found:
                        bad_rot = True
                        break
                    
                    # print(f"Would become false: {last} by flipping {lit} which has coef {get_coefficient_lit(constraint_check, neg_lit)}")
                    if count > 1:
                        # print("More than one constraint would become false, stopping rotation here")
                        break
            if count == 1:
                if bad_rot:
                    continue
                found.add(last)
                
                if recursive:
                    # print("Rotating model:", assoc)
                    flip_literal(lit)
                    # print(f"flipped {lit.name} to {lit.value()}")
                    rotate_model(constraints, last, depth=depth-1 if depth is not None else None, found=found, recursive=recursive)
                    flip_literal(lit)
                    # print(f"flipped {lit.name} back to {lit.value()}")
                
                # print("Found constraint to rotate:", last)
                # assoc = rotate_model(model, assoc, last)
        elif under_slack:
            flipped_lits.add(lit)
            slack += c
            flip_literal(lit)
        else:
            break # since the lits are sorted by coefficient, we can stop here
        if count_neg > 1:
            break

    return

def rotate_model(constraints, constraint, depth=None, recursive=True, found=set(), hard=[]):    
    if depth == 0:
        return
    lits = get_lits(constraint)
    slack = get_slack(constraint)
    
    for lit in lits:
        if not is_false(lit):
            continue
        c = get_coefficient_lit(constraint, lit)
        if c >= -slack:
            count = 0
            
            bad_rot = False # flag to avoid bad rotations
            # loop over constraints in model, if only one will become false then add it to found and rotate recursively
            for constraint_check in constraints:
                # print(constraint_check)
                if constraint_check is constraint:
                    continue
                neg_lit = ~lit
                # slack_other = get_slack(constraint_check, assoc)
                under_const = slack_under(constraint_check, get_coefficient_lit(constraint_check, neg_lit))
                if under_const:
                    count += 1
                    last = constraint_check
                    if constraint_check in hard:
                        bad_rot = True
                        break
                    if constraint_check in found:
                        bad_rot = True
                        break
                    
                    # print(f"Would become false: {last} by flipping {lit} which has coef {get_coefficient_lit(constraint_check, neg_lit)}")
                    if count > 1:
                        # print("More than one constraint would become false, stopping rotation here")
                        break
            if count == 1:
                if bad_rot:
                    continue
                found.add(last)
                
                # rotated_assoc = assoc.copy()
                # rotated_assoc[lits.index(lit)] = not assoc[lits.index(lit)]
                if recursive:
                    # print("Rotating model:", assoc)
                    flip_literal(lit)
                    # print(f"flipped {lit.name} to {lit.value()}")
                    rotate_model(constraints, last, depth=depth-1 if depth is not None else None, found=found, recursive=recursive)
                    flip_literal(lit)
                    # print(f"flipped {lit.name} back to {lit.value()}")
                
                # print("Found constraint to rotate:", last)
                # assoc = rotate_model(model, assoc, last)
        else:
            slack += c
            flip_literal(lit)

    
    # for lit in flipped_lits:
    #     flip_literal(lit) # fix the flipped literals
    
    # print(f"Finished rotation at depth {depth}")

    return

def rotate_model_group(groups, group_id, depth=None, recursive=True, found=set(), hard=[]):
    group = groups[group_id]
    if len(group) > 1:
        return
    else:
        # print(group)
        # print(group_id)
        constraint = group[0]    
    if depth == 0:
        return
    lits = get_lits(constraint)
    slack = get_slack(constraint)
    
    for lit in lits:
        if not is_false(lit):
            continue
        c = get_coefficient_lit(constraint, lit)
        if c >= -slack:
            count = 0
            
            flip_literal(lit)
            
            bad_rot = False # flag to avoid bad rotations
            # loop over constraints in model, if only one will become false then add it to found and rotate recursively
            for constraint_check in hard:
                # print(constraint_check)
                if constraint_check is constraint:
                    continue
                
                if not constraint_check.value():
                    bad_rot = True
                    break
                    
            if not bad_rot:    
                for id, g in groups:
                    for c in g:
                        if not c.value():
                            if count == 1:
                                bad_rot = True
                                break
                            last = id
                            count += 1
                                
                    
            if count == 1:
                if bad_rot:
                    continue
                found.add(last)
                
                # rotated_assoc = assoc.copy()
                # rotated_assoc[lits.index(lit)] = not assoc[lits.index(lit)]
                if recursive:
                    # print("Rotating model:", assoc)
                    print(f"flipped {lit.name} to {lit.value()}")
                    rotate_model_group(groups, last, depth=depth-1 if depth is not None else None, found=found, recursive=recursive)
                    # print(f"flipped {lit.name} back to {lit.value()}")
                
                # print("Found constraint to rotate:", last)
                # assoc = rotate_model(model, assoc, last)
            flip_literal(lit)
        else:
            slack += c
            flip_literal(lit)

    
    # for lit in flipped_lits:
    #     flip_literal(lit) # fix the flipped literals
    
    # print(f"Finished rotation at depth {depth}")

    return

def rotate_model_group_linear(groups, group_id, depth=None, recursive=True, found=set(), hard=[]):
    group = groups[group_id]
    if len(group) > 1:
        return
    else:
        constraint = group[0]    
    if depth == 0:
        return
    lits = get_lits(constraint)
    
    for lit in lits:
        
        curr_value = lit.value()
        
        c = get_coefficient_lit_linear(constraint, lit)
        
        lower, upper = get_bounds(constraint, lit, c)
        
        lower = lit.lb
        upper = lit.ub
        # if pos_contribution(lit, c, constraint):
        #     continue
        
        for v in range(lower, upper+1):
            if v == curr_value:
                continue
            
        
            lit._value = v
            
            if constraint.value():
                count = 0
                
                
                bad_rot = False # flag to avoid bad rotations
                # loop over constraints in model, if only one will become false then add it to found and rotate recursively
                for constraint_check in hard:
                    # print(constraint_check)
                    if constraint_check is constraint:
                        continue
                    
                    if not constraint_check.value():
                        bad_rot = True
                        break
                        
                if not bad_rot:  
                    for id, g in groups.items():
                        for c in g:
                            if not c.value():
                                if count == 1:
                                    bad_rot = True
                                    break
                                last = id
                                count += 1
                                    
                        
                if count == 1:
                    if bad_rot:
                        continue
                    found.add(last)
                    
                    # rotated_assoc = assoc.copy()
                    # rotated_assoc[lits.index(lit)] = not assoc[lits.index(lit)]
                    if recursive:
                        # print("Rotating model:", assoc)
                        # print(f"flipped {lit.name} to {lit.value()}")
                        rotate_model_group_linear(groups, last, depth=depth-1 if depth is not None else None, found=found, recursive=recursive)
                        # print(f"flipped {lit.name} back to {lit.value()}")
                    
                    # print("Found constraint to rotate:", last)
                    # assoc = rotate_model(model, assoc, last)
        lit._value = curr_value

    
    # for lit in flipped_lits:
    #     flip_literal(lit) # fix the flipped literals
    
    # print(f"Finished rotation at depth {depth}")

    return
        
        
def rotate_model_group_cp(groups, group_id, depth=None, recursive=True, found=set(), hard=[]):
    group = groups[group_id]
    if len(group) > 1:
        return
    else:
        constraint = group[0]    
    if depth == 0:
        return
    vars = get_variables(constraint)
    
    for var in vars:
        
        curr_value = var.value()
        
        lower = var.lb
        upper = var.ub
        
        for v in range(lower, upper+1):
            if v == curr_value:
                continue
            
        
            var._value = v
            
            
            if constraint.value():
            # if True:
                # assert constraint.value() is False, f"After setting {var} to {v}, constraint {constraint} is False"
                count = 0
                
                
                bad_rot = False # flag to avoid bad rotations
                # loop over constraints in model, if only one will become false then add it to found and rotate recursively
                for constraint_check in hard:
                    # print(constraint_check)
                    if constraint_check is constraint:
                        continue
                    
                    if constraint_check.value() is False:
                        bad_rot = True
                        break
                        
                print(f"in {constraint} the value of {var} was set to {var.value()} from {curr_value}")
                
                if not bad_rot:  
                    for id, g in groups.items():
                        for c in g:
                            if c.value() is False:
                                print(f"group {id} is now false because of {c}")
                                print(f"count: {count + 1}")
                                if count == 1:
                                    bad_rot = True
                                    break
                                last = id
                                count += 1
                                break
                        if bad_rot:
                            break
                                    
                        
                if count == 1:
                    if bad_rot:
                        continue
                    found.add(last)
                    
                    # rotated_assoc = assoc.copy()
                    # rotated_assoc[lits.index(lit)] = not assoc[lits.index(lit)]
                    if recursive:
                        # print("Rotating model:", assoc)
                        # print(f"flipped {lit.name} to {lit.value()}")
                        rotate_model_group_cp(groups, last, depth=depth-1 if depth is not None else None, found=found, recursive=recursive)
                        # print(f"flipped {lit.name} back to {lit.value()}")
                    
                    # print("Found constraint to rotate:", last)
                    # assoc = rotate_model(model, assoc, last)
        var._value = curr_value

    
    # for lit in flipped_lits:
    #     flip_literal(lit) # fix the flipped literals
    
    # print(f"Finished rotation at depth {depth}")

    return

def rotate_model_cp(constraints, constraint, depth=None, recursive=True, found=set(), hard=[]):
    if depth == 0:
        return
    vars = get_variables(constraint)
    
    for var in vars:
        
        curr_value = var.value()
        
        lower = var.lb
        upper = var.ub
        
        for v in range(lower, upper+1):
            if v == curr_value:
                continue
            
        
            var._value = v
            
            
            if constraint.value():
            # if True:
                # assert constraint.value() is False, f"After setting {var} to {v}, constraint {constraint} is False"
                count = 0
                
                
                bad_rot = False # flag to avoid bad rotations
                # loop over constraints in model, if only one will become false then add it to found and rotate recursively
                for constraint_check in hard+list(found):
                    if constraint_check is constraint:
                        continue
                    # print(constraint_check)
                    
                    if constraint_check.value() is False:
                        bad_rot = True
                        break
                
                if not bad_rot:  
                    for c in constraints:
                        if c is constraint:
                            continue
                        if c.value() is False:
                            if count == 1:
                                bad_rot = True
                                break
                            last = c
                            count += 1
                                    
                        
                if count == 1:
                    if bad_rot:
                        continue
                    found.add(last)
                    
                    # rotated_assoc = assoc.copy()
                    # rotated_assoc[lits.index(lit)] = not assoc[lits.index(lit)]
                    if recursive:
                        # print("Rotating model:", assoc)
                        # print(f"flipped {lit.name} to {lit.value()}")
                        rotate_model_cp(constraints, last, depth=depth-1 if depth is not None else None, found=found, recursive=recursive)
                        # print(f"flipped {lit.name} back to {lit.value()}")
                    
                    # print("Found constraint to rotate:", last)
                    # assoc = rotate_model(model, assoc, last)
        var._value = curr_value

    
    # for lit in flipped_lits:
    #     flip_literal(lit) # fix the flipped literals
    
    # print(f"Finished rotation at depth {depth}")

    return
def replace_cons_with_assump(cpm_cons, assump_map):
    """
        Replace soft constraints with assumption variables in a Boolean CPMpy expression.
    """

    if is_any_list(cpm_cons):
        return [replace_cons_with_assump(c, assump_map) for c in cpm_cons]
    
    if cpm_cons in assump_map:
        return assump_map[cpm_cons]
    
    elif hasattr(cpm_cons, "args"):
        cpm_cons = copy.copy(cpm_cons)
        cpm_cons.update_args(replace_cons_with_assump(cpm_cons.args, assump_map))
        return cpm_cons

    elif isinstance(cpm_cons, NegBoolView):
        return ~replace_cons_with_assump(cpm_cons._bv, assump_map)
    return cpm_cons

class OCUSException(Exception):
    pass
