"""
    Re-impementation of MUS-computation techniques in CPMPy

    - Deletion-based MUS
    - QuickXplain
    - Optimal MUS
"""
import warnings
import numpy as np
import cpmpy as cp
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.get_variables import get_variables, get_variables_model 
from cpmpy.transformations.int2bool import int2bool
from cpmpy.transformations.linearize import canonical_comparison, linearize_constraint, only_ge_comparison, only_positive_coefficients, sorted_coefficients
from cpmpy.transformations.normalize import simplify_boolean, toplevel_list
from cpmpy.solvers.solver_interface import ExitStatus
import time

from cpmpy.tools.explain.breakid import BreakID
from cpmpy.tools.explain.breakid import BREAKID_PATH
from cpmpy.transformations.reification import only_bv_reifies, only_implies, reify_rewrite
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list

from .utils import make_assump_model, replace_cons_with_assump, OCUSException

from .utils import get_length_gen, make_assump_model, get_slack, get_degree_over_sum, get_max_sat, get_min_sat, get_length, rotate_model, rotate_model_group_cp, rotate_model_cp, rotate_model_group_linear, rotate_model_old, rotate_model_group

# @profile
def pb_mus(soft, hard=[], solver="exact", clause_set_refinement=True, init_check=True, assumption_removal=False, redundancy_removal=False, sorting="length", reversed_order=True, model_rotation=False, maximize_cons=False, recursive=True, assertions=False, use_symmetries=False, time_limit=1800, **kwargs):
    """
        A PB-level deletion-based MUS algorithm using assumption variables
        and unsat core extraction

        For solvers that support s.solve(assumptions=...) and s.get_core()

        All constraint are PB constraints, or groups of PB constraints.

        Will extract an unsat core and then shrink the core further
        by repeatedly ommitting one assumption variable.

        :param: soft: soft constraints, list of expressions
        :param: hard: hard constraints, optional, list of expressions
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: time_limit: optional time limit for the MUS extraction process
        :param: assumption_removal: when true, will permanently remove assumption variables when the it is determined whether they are in the MUS or not
        :param: redundancy_removal: when true, will add the negation of the constraint, of which the inclusion in the MUS is being tested, to the solver call
        :param: clause_set_refinement: when True, will use the solver's UNSAT core to refine the MUS further
        :param: init_check: when True, will check that the model is UNSAT before starting the MUS algorithm 
        :param: model_rotation: when True, apply model rotation after finding a transition constraint in order to find new transition constraints without needing to call the solver again
        :param: dec_vars: decision variables to use for model rotation, if model_rotation is True. Must be provided if model_rotation is True.
        :param: sorting: sorting heuristic to use for the constraints, can be "dgs", "max_sat", "min_sat" or "length"
        :param: reversed_order: if True, will reverse the order of the constraints in the sorting heuristic
        :param: maximize_cons: if True, will maximize the lefthandside of the constraint that is being removed from the core in order to more easily apply model rotation.
    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"mus requires a solver that supports assumption variables, use mus_naive with {solver} instead"

    # make assumption (indicator) variables and soft-constrained model
    (m, soft, assump) = make_assump_model(soft, hard=hard, name="mus_sel")
    
    if use_symmetries:
        breakid = BreakID(BREAKID_PATH)  # use pb branch
        permutations, matrices = breakid.get_generators(m.constraints, format="opb", subset=assump,pb=31, no_row=False)
        symmetries = permutations + matrices
        # print(f"there are {len(symmetries)} symmetries")
        
    s = cp.SolverLookup.get(solver, m)

    # create dictionary from assump to soft
    dmap = dict(zip(assump, soft))
    
    start = time.time()
    unsat_calls = 0
    sat_calls = 0
    total_solve_time = 0
    nb_removed_refinement = 0
    nb_found_mr = 0
    nb_found_symm = 0
    
    core_size = len(assump)
    
    core = set(assump)

    if init_check:
        # setting all assump vars to true should be UNSAT
        # print("Performing initial UNSAT check...")
        
        if solver == "pysat:Cadical195":
            warnings.warn("Can not add time_limit to pysat:Cadical195 solver calls, ignoring time_limit argument")
            assert not s.solve(assumptions=assump, **kwargs), "MUS: model must be UNSAT"
        else:
            assert not s.solve(assumptions=assump, time_limit=time_limit, **kwargs), "MUS: model must be UNSAT"
            
        if time_limit is not None:
            elapsed = time.time() - start
            if elapsed >= time_limit:
                raise TimeoutError("Time's up during initial solve")
            total_solve_time += elapsed
        # print(f"Initial UNSAT check done in {time.time() - start} sec.")
        unsat_calls += 1
        new_core = set(s.get_core())  # start from solver's UNSAT core
        if assumption_removal:
            newly_removed = core - new_core
            for r in newly_removed:
                s += ~r # remove red constraint
        core = new_core
        
        nb_removed_refinement += core_size - len(core)
    else:
        warnings.warn("No initial check, using all assumptions. The initial model may be SAT.", UserWarning)
        core = set(assump) # start from all assumptions, this may avoid an unnecessarily slow UNSAT call, if the model is not severely overconstrained

    # deletion-based MUS
    # order so that constraints with many variables are tried and removed first
    heuristics = {
        "dgs": get_degree_over_sum,
        "max_sat": get_max_sat,
        "min_sat": get_min_sat,
        "length": get_length,
    }
    
    found = set() # keep track of found transition constraints
    found_cons = set()
    
    
    
    for c in sorted(core, key=lambda c : heuristics[sorting](dmap[c]), reverse=reversed_order):
        # print(f"Checking {c}")
        if c not in core:
            # print(f"skipping {dmap[c]}")
            continue # already removed
        if (model_rotation or use_symmetries) and dmap[c] in found_cons:
            found.add(c)
            continue
        
        core_size = len(core)
        
        core.remove(c) # remove from core
        
        if redundancy_removal:
            red_constraint = ~dmap[c]
            red_var = cp.boolvar()
            s += red_var.implies(red_constraint) # add red constraint
            assumps = list(core) + [red_var]
        else:
            assumps = list(core)
            
        start_solve = time.time()
        if solver != "pysat:Cadical195":
            s.solve(assumptions=assumps, time_limit=time_limit-(int(time.time()-start)), **kwargs)
        else:
            s.solve(assumptions=assumps, **kwargs)
        
        last_call_time = time.time() - start_solve
        # print(last_call_time)
        total_solve_time += last_call_time
        if s.status().exitstatus == ExitStatus.FEASIBLE or s.status().exitstatus == ExitStatus.OPTIMAL:
            assert get_slack(dmap[c]) < 0, f"Constraint {dmap[c]} has positive slack {get_slack(dmap[c])}"
            sat_calls += 1
            # hard.append(dmap[c])
            core.add(c)
            if assumption_removal:
                s += c # permanently set to true
            found.add(c) # add to found transition constraints
            found_cons.add(dmap[c])
            if model_rotation:
                if maximize_cons and 3*last_call_time < time_limit - (time.time() - start):
                    s.maximize(dmap[c].args[0])
                    new_t_limit = max(0.001, 3*last_call_time)
                    s.solve(time_limit=new_t_limit, assumptions=assumps, **kwargs)

                # for constraint_check in [dmap[c] for c in core] + hard:
                #     if constraint_check is dmap[c]:
                #         continue
                #     assert get_slack(constraint_check, assoc) >= 0, f"Constraint {dmap[constraint_check]} has negative slack {get_slack(dmap[constraint_check], assoc)}"
                found_size = len(found_cons)
                rotate_model([dmap[c] for c in core] + hard, dmap[c], recursive=recursive, found=found_cons, hard=hard)
                # print(f"Model rotation found {len(found) - found_size} new transition constraints")
                nb_found_mr += len(found_cons) - found_size
                # TODO: skip over found transition constraints
                
            if use_symmetries:
                for symm in symmetries:
                    new_found = symm.get_symmetric_images_in_subset(core, c)
                    for c in new_found:
                        found_cons.add(dmap[c])
                    found_size = len(found)
                    found.update(new_found)
                    nb_found_symm += len(found) - found_size
                
                # get the associatied solution for the found transition constraint
                # print(f"constraint: {dmap[c]}")
                # print(f"vars: {dec_vars.value()}")
                # print(f"constraint slack: {get_slack(dmap[c], dec_vars)}")
        elif s.status().exitstatus == ExitStatus.UNSATISFIABLE: # UNSAT, use new solver core (clause set refinement)
            unsat_calls += 1
            if clause_set_refinement:
                new_core = set(s.get_core()).union(found)
                if redundancy_removal:
                    # s += ~red_var # remove red constraint
                    if red_var in new_core:
                        continue
                nb_removed_refinement += len(core) - len(new_core)
                if assumption_removal:
                    newly_removed = core - new_core
                    for r in newly_removed:
                        s += ~r # permanently set to false
                    s += ~c # permanently set to false
                core = new_core
        # print(f"Model: {m}")
        else:
            raise RuntimeError(f"MUS: solver returned unexpected status {s.status().exitstatus}")
            
    # print(f"Number of solve calls: {nb_sat_calls + nb_unsat_calls} ({nb_sat_calls} SAT, {nb_unsat_calls} UNSAT)")
    # print(f"Total solve time: {total_solve_time}")
    if assertions:
        
        assert len(mus(list(found_cons), hard=hard, solver=solver)[0]) == len(found), "MUS: final core is not a MUS"

    return found, nb_removed_refinement, nb_found_mr, nb_found_symm, sat_calls, unsat_calls, total_solve_time

def pb_mus_group(soft, hard=[], solver="exact", clause_set_refinement=True, init_check=True, assumption_removal=False, redundancy_removal=False, sorting="length", reversed_order=True, model_rotation=False, maximize_cons=False, recursive=True, assertions=False, use_symmetries=False, time_limit=1800, **kwargs):
    """
        A PB-level deletion-based MUS algorithm using assumption variables
        and unsat core extraction.
        All constraints are first translated to the PB-level and are then treated as grouped PB constraints.
        You need the pysat solver for this translation!

        For solvers that support s.solve(assumptions=...) and s.get_core()

        All constraint are PB constraints, or groups of PB constraints.

        Will extract an unsat core and then shrink the core further
        by repeatedly ommitting one assumption variable.

        :param: soft: soft constraints, list of expressions
        :param: hard: hard constraints, optional, list of expressions
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: time_limit: optional time limit for the MUS extraction process
        :param: assumption_removal: when true, will permanently remove assumption variables when the it is determined whether they are in the MUS or not
        :param: redundancy_removal: when true, will add the negation of the constraint, of which the inclusion in the MUS is being tested, to the solver call
        :param: clause_set_refinement: when True, will use the solver's UNSAT core to refine the MUS further
        :param: init_check: when True, will check that the model is UNSAT before starting the MUS algorithm 
        :param: model_rotation: when True, apply model rotation after finding a transition constraint in order to find new transition constraints without needing to call the solver again
        :param: dec_vars: decision variables to use for model rotation, if model_rotation is True. Must be provided if model_rotation is True.
        :param: sorting: sorting heuristic to use for the constraints, can only be "length"
        :param: reversed_order: if True, will reverse the order of the constraints in the sorting heuristic
        :param: maximize_cons: if True, will maximize the lefthandside of the constraint that is being removed from the core in order to more easily apply model rotation.
    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"mus requires a solver that supports assumption variables, use mus_naive with {solver} instead"

    # make assumption (indicator) variables and soft-constrained model
    (m, soft, assump) = make_assump_model(soft, hard=hard, name="mus_sel")
    
    slv = CPM_pysat()
    
    constraints = m.constraints
    
    constraints = toplevel_list(constraints)
    constraints = decompose_in_tree(constraints,supported=frozenset({'alldifferent'}), supported_reified=frozenset({'alldifferent'}), csemap=slv._csemap)  # Alldiff has a specialzed MIP decomp
    constraints = simplify_boolean(constraints)
    constraints = flatten_constraint(constraints, csemap=slv._csemap)  # flat normal form
    constraints = reify_rewrite(constraints, supported=frozenset(['sum', 'wsum', 'alldifferent']), csemap=slv._csemap)  # constraints that support reification
    constraints = only_numexpr_equality(constraints, supported=frozenset(["sum", "wsum", 'alldifferent']), csemap=slv._csemap)  # supports >, <, !=
    constraints = only_bv_reifies(constraints, csemap=slv._csemap)
    constraints = only_implies(constraints, csemap=slv._csemap)  # anything that can create full reif should go above...
    constraints = linearize_constraint(constraints, supported=frozenset({"sum", "wsum"}), csemap=slv._csemap)  # the core of the MIP-linearization
    # constraints = int2bool(constraints, slv.ivarmap, encoding="binary")
    constraints = canonical_comparison(constraints)
    # constraints = only_ge_comparison(constraints)
    # constraints = only_positive_coefficients(constraints)
    # constraints = sorted_coefficients(constraints)
    
    # print(constraints)
    
    model = cp.Model(constraints)
    
    if use_symmetries:
        breakid = BreakID(BREAKID_PATH)  # use pb branch
        permutations, matrices = breakid.get_generators(constraints, format="opb", subset=assump,pb=31, no_row=False)
        symmetries = permutations + matrices
        # print(f"there are {len(symmetries)} symmetries")
        
    s = cp.SolverLookup.get(solver, model)

    # create dictionary from assump to soft
    dmap = dict(zip(assump, soft))
    # create dictionary from assump to group
    groups = {a: [] for a in assump}
    hard_trans = []
    for c in constraints:
        if c.name == "->" and c.args[0] in assump:
            # print(c)
            groups[c.args[0]].append(c.args[1])
        else:
            hard_trans.append(c)
    
    hard.extend(hard_trans)
                
    
    start = time.time()
    unsat_calls = 0
    sat_calls = 0
    total_solve_time = 0
    nb_removed_refinement = 0
    nb_found_mr = 0
    nb_found_symm = 0
    
    core_size = len(assump)
    
    core = set(assump)

    if init_check:
        # setting all assump vars to true should be UNSAT
        # print("Performing initial UNSAT check...")
        
        if solver == "pysat:Cadical195":
            warnings.warn("Can not add time_limit to pysat:Cadical195 solver calls, ignoring time_limit argument")
            assert not s.solve(assumptions=assump, **kwargs), "MUS: model must be UNSAT"
        else:
            assert not s.solve(assumptions=assump, time_limit=time_limit, **kwargs), "MUS: model must be UNSAT"
            
        if time_limit is not None:
            elapsed = time.time() - start
            if elapsed >= time_limit:
                raise TimeoutError("Time's up during initial solve")
            total_solve_time += elapsed
        # print(f"Initial UNSAT check done in {time.time() - start} sec.")
        unsat_calls += 1
        new_core = set(s.get_core())  # start from solver's UNSAT core
        if assumption_removal:
            newly_removed = core - new_core
            for r in newly_removed:
                s += ~r # remove red constraint
        core = new_core
        
        nb_removed_refinement += core_size - len(core)
    else:
        warnings.warn("No initial check, using all assumptions. The initial model may be SAT.", UserWarning)
        core = set(assump) # start from all assumptions, this may avoid an unnecessarily slow UNSAT call, if the model is not severely overconstrained

    # deletion-based MUS
    # order so that constraints with many variables are tried and removed first
    heuristics = {
        "length": get_length_gen,
    }
    
    found = set() # keep track of found transition constraints
    found_cons = set()
    
    
    
    for c in sorted(core, key=lambda c : heuristics[sorting](dmap[c]), reverse=reversed_order):
        # print(f"Checking {c}")
        if c not in core:
            # print(f"skipping {dmap[c]}")
            continue # already removed
        if (model_rotation or use_symmetries) and dmap[c] in found_cons:
            found.add(c)
            continue
        
        core_size = len(core)
        
        core.remove(c) # remove from core
        
        if redundancy_removal:
            red_constraint = ~dmap[c]
            red_var = cp.boolvar()
            s += red_var.implies(red_constraint) # add red constraint
            assumps = list(core) + [red_var]
        else:
            assumps = list(core)
            
        start_solve = time.time()
        if solver != "pysat:Cadical195":
            s.solve(assumptions=assumps, time_limit=time_limit-(int(time.time()-start)), **kwargs)
        else:
            s.solve(assumptions=assumps, **kwargs)
        
        last_call_time = time.time() - start_solve
        # print(last_call_time)
        total_solve_time += last_call_time
        if s.status().exitstatus == ExitStatus.FEASIBLE or s.status().exitstatus == ExitStatus.OPTIMAL:
            # print(dmap[c])
            # print(dmap[c].value())
            # assert not dmap[c].value(), f"Constraint {dmap[c]} is {dmap[c].value()}, should be false"
            # for sel in core:
            #     assert dmap[sel].value()
            # TODO: check satisfiability of group, need actual group dict
            sat_calls += 1
            # hard.append(dmap[c])
            core.add(c)
            if assumption_removal:
                s += c # permanently set to true
            found.add(c) # add to found transition constraints
            found_cons.add(dmap[c])
            if model_rotation:
                if maximize_cons and 3*last_call_time < time_limit - (time.time() - start):
                    s.maximize(dmap[c].args[0])
                    new_t_limit = max(0.001, 3*last_call_time)
                    s.solve(time_limit=new_t_limit, assumptions=assumps, **kwargs)

                found_size = len(found_cons)
                rotate_model_group_linear(groups, c, recursive=recursive, found=found_cons, hard=hard)
                # print(f"Model rotation found {len(found) - found_size} new transition constraints")
                nb_found_mr += len(found_cons) - found_size
                # TODO: skip over found transition constraints
                
            if use_symmetries:
                for symm in symmetries:
                    new_found = symm.get_symmetric_images_in_subset(core, c)
                    for c in new_found:
                        found_cons.add(dmap[c])
                    found_size = len(found)
                    found.update(new_found)
                    nb_found_symm += len(found) - found_size
                
                # get the associatied solution for the found transition constraint
                # print(f"constraint: {dmap[c]}")
                # print(f"vars: {dec_vars.value()}")
                # print(f"constraint slack: {get_slack(dmap[c], dec_vars)}")
        elif s.status().exitstatus == ExitStatus.UNSATISFIABLE: # UNSAT, use new solver core (clause set refinement)
            unsat_calls += 1
            if clause_set_refinement:
                new_core = set(s.get_core()).union(found)
                if redundancy_removal:
                    # s += ~red_var # remove red constraint
                    if red_var in new_core:
                        continue
                nb_removed_refinement += len(core) - len(new_core)
                if assumption_removal:
                    newly_removed = core - new_core
                    for r in newly_removed:
                        s += ~r # permanently set to false
                    s += ~c # permanently set to false
                core = new_core
        # print(f"Model: {m}")
        else:
            raise RuntimeError(f"MUS: solver returned unexpected status {s.status().exitstatus}")
            
    # print(f"Number of solve calls: {nb_sat_calls + nb_unsat_calls} ({nb_sat_calls} SAT, {nb_unsat_calls} UNSAT)")
    # print(f"Total solve time: {total_solve_time}")
    if assertions:
        
        assert len(mus(list(found_cons), hard=hard, solver=solver)[0]) == len(found), "MUS: final core is not a MUS"

    return [dmap[c] for c in found], found, nb_removed_refinement, nb_found_mr, nb_found_symm, sat_calls, unsat_calls, total_solve_time

def cp_mus(soft, hard=[], solver="exact", clause_set_refinement=True, init_check=True, assumption_removal=False, redundancy_removal=False, sorting="length", reversed_order=True, model_rotation=False, maximize_cons=False, recursive=True, assertions=False, use_symmetries=False, time_limit=1800, **kwargs):
    """
        A PB-level deletion-based MUS algorithm using assumption variables
        and unsat core extraction.
        All constraints are first translated to the PB-level and are then treated as grouped PB constraints.
        You need the pysat solver for this translation!

        For solvers that support s.solve(assumptions=...) and s.get_core()

        All constraint are PB constraints, or groups of PB constraints.

        Will extract an unsat core and then shrink the core further
        by repeatedly ommitting one assumption variable.

        :param: soft: soft constraints, list of expressions
        :param: hard: hard constraints, optional, list of expressions
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: time_limit: optional time limit for the MUS extraction process
        :param: assumption_removal: when true, will permanently remove assumption variables when the it is determined whether they are in the MUS or not
        :param: redundancy_removal: when true, will add the negation of the constraint, of which the inclusion in the MUS is being tested, to the solver call
        :param: clause_set_refinement: when True, will use the solver's UNSAT core to refine the MUS further
        :param: init_check: when True, will check that the model is UNSAT before starting the MUS algorithm 
        :param: model_rotation: when True, apply model rotation after finding a transition constraint in order to find new transition constraints without needing to call the solver again
        :param: dec_vars: decision variables to use for model rotation, if model_rotation is True. Must be provided if model_rotation is True.
        :param: sorting: sorting heuristic to use for the constraints, can only be "length"
        :param: reversed_order: if True, will reverse the order of the constraints in the sorting heuristic
        :param: maximize_cons: if True, will maximize the lefthandside of the constraint that is being removed from the core in order to more easily apply model rotation.
    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"mus requires a solver that supports assumption variables, use mus_naive with {solver} instead"

    # make assumption (indicator) variables and soft-constrained model
    (m, soft, assump) = make_assump_model(soft, hard=hard, name="mus_sel")
        
    s = cp.SolverLookup.get(solver, m)

    # create dictionary from assump to soft
    dmap = dict(zip(assump, soft))
                
    
    start = time.time()
    unsat_calls = 0
    sat_calls = 0
    total_solve_time = 0
    nb_removed_refinement = 0
    nb_found_mr = 0
    nb_found_symm = 0
    
    core_size = len(assump)
    
    core = set(assump)

    if init_check:
        # setting all assump vars to true should be UNSAT
        # print("Performing initial UNSAT check...")
    
        if solver == "pysat:Cadical195":
            warnings.warn("Can not add time_limit to pysat:Cadical195 solver calls, ignoring time_limit argument")
            assert not s.solve(assumptions=assump, **kwargs), "MUS: model must be UNSAT"
        else:
            assert not s.solve(assumptions=assump, time_limit=time_limit, **kwargs), "MUS: model must be UNSAT"
            
        if time_limit is not None:
            elapsed = time.time() - start
            if elapsed >= time_limit:
                raise TimeoutError("Time's up during initial solve")
            total_solve_time += elapsed
        # print(f"Initial UNSAT check done in {time.time() - start} sec.")
        unsat_calls += 1
        new_core = set(s.get_core())  # start from solver's UNSAT core
        core = new_core
        
        nb_removed_refinement += core_size - len(core)
    else:
        warnings.warn("No initial check, using all assumptions. The initial model may be SAT.", UserWarning)
        core = set(assump) # start from all assumptions, this may avoid an unnecessarily slow UNSAT call, if the model is not severely overconstrained

    # deletion-based MUS
    # order so that constraints with many variables are tried and removed first
    heuristics = {
        "length": get_length_gen,
    }
    
    found = set() # keep track of found transition constraints
    found_cons = set()
    
    
    
    for c in sorted(core, key=lambda c : heuristics[sorting](dmap[c]), reverse=reversed_order):
        # print(f"Checking {c}")
        if c not in core:
            # print(f"skipping {dmap[c]}")
            continue # already removed
        if (model_rotation or use_symmetries) and dmap[c] in found_cons:
            found.add(c)
            continue
        
        core_size = len(core)
        
        core.remove(c) # remove from core
        
        if redundancy_removal:
            red_constraint = ~dmap[c]
            red_var = cp.boolvar()
            s += red_var.implies(red_constraint) # add red constraint
            assumps = list(core) + [red_var]
        else:
            assumps = list(core)
            
        start_solve = time.time()

        s.solve(assumptions=assumps, **kwargs)
        
        last_call_time = time.time() - start_solve
        # print(last_call_time)
        total_solve_time += last_call_time
        if s.status().exitstatus == ExitStatus.FEASIBLE:
            # print(dmap[c])
            # print(dmap[c].value())
            assert not dmap[c].value(), f"Constraint {dmap[c]} is satisfied"
            for sel in core:
                assert sel.value()
                assert dmap[sel].value()
            # TODO: check satisfiability of group, need actual group dict
            sat_calls += 1
            # hard.append(dmap[c])
            core.add(c)
            if assumption_removal:
                s += c # permanently set to true
            found.add(c) # add to found transition constraints
            found_cons.add(dmap[c])
            if model_rotation:

                found_size = len(found_cons)
                rotate_model_cp([dmap[c] for c in core] + hard, dmap[c], recursive=recursive, found=found_cons, hard=hard)
                # print(f"Model rotation found {len(found) - found_size} new transition constraints")
                nb_found_mr += len(found_cons) - found_size
                # TODO: skip over found transition constraints
                
                # get the associatied solution for the found transition constraint
                # print(f"constraint: {dmap[c]}")
                # print(f"vars: {dec_vars.value()}")
                # print(f"constraint slack: {get_slack(dmap[c], dec_vars)}")
        elif s.status().exitstatus == ExitStatus.UNSATISFIABLE: # UNSAT, use new solver core (clause set refinement)
            unsat_calls += 1
            if clause_set_refinement:
                new_core = set(s.get_core()).union(found)
                if redundancy_removal:
                    # s += ~red_var # remove red constraint
                    if red_var in new_core:
                        continue
                nb_removed_refinement += len(core) - len(new_core)
                core = new_core
        # print(f"Model: {m}")
        else:
            raise RuntimeError(f"MUS: solver returned unexpected status {s.status().exitstatus}")
            
    # print(f"Number of solve calls: {nb_sat_calls + nb_unsat_calls} ({nb_sat_calls} SAT, {nb_unsat_calls} UNSAT)")
    # print(f"Total solve time: {total_solve_time}")
    if assertions:
        
        assert len(mus(list(found_cons), hard=hard, solver=solver)[0]) == len(found), "MUS: final core is not a MUS"

    return [dmap[c] for c in found], found, nb_removed_refinement, nb_found_mr, nb_found_symm, sat_calls, unsat_calls, total_solve_time

def mus_new(soft, hard=[], solver="ortools", redundancy_removal=False, assumption_removal=False, time_limit=None, **kwargs):
    """
        A CP deletion-based MUS algorithm using assumption variables
        and unsat core extraction

        For solvers that support s.solve(assumptions=...) and s.get_core()

        Each constraint is an arbitrary CPMpy expression, so it can
        also be sublists of constraints (e.g. constraint groups),
        contain aribtrary nested expressions, global constraints, etc.

        Will extract an unsat core and then shrink the core further
        by repeatedly ommitting one assumption variable.

        :param: soft: soft constraints, list of expressions
        :param: hard: hard constraints, optional, list of expressions
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: clause_set_refinement: if True, will use the solver's UNSAT core to refine the MUS further
        :param: init_check: if True, will check that the model is UNSAT before starting the MUS algorithm
        :param: redundancy_removal: if True, add the negated constraint to the model as a redundant constraint for each call to the solver
    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"mus requires a solver that supports assumption variables, use mus_naive with {solver} instead"

    if time_limit is None:
        time_limit = 1800
    
    start_time = time.time()
    
    # make assumption (indicator) variables and soft-constrained model
    (m, soft, assump) = make_assump_model(soft, hard=hard, name="mus_sel")
    s = cp.SolverLookup.get(solver, m)

    # create dictionary from assump to soft
    dmap = dict(zip(assump, soft))

    core = set(assump)  # start from all soft constraints
    
    total_solve_time = 0.0
    start_solve = time.time()
    
    # setting all assump vars to true should be UNSAT
    if solver == "pysat:Cadical195":
        warnings.warn("Can not add time_limit to pysat:Cadical195 solver calls, ignoring time_limit argument")
        assert not s.solve(assumptions=assump), "MUS: model must be UNSAT"
    else:
        assert not s.solve(assumptions=assump, time_limit=time_limit, **kwargs), "MUS: model must be UNSAT"
    if time_limit is not None:
        elapsed = time.time() - start_solve
        if elapsed >= time_limit:
            raise TimeoutError("Time's up during initial solve")
        total_solve_time += elapsed
    
    new_core = set(s.get_core())  # start from solver's UNSAT core
    unsat_calls = 1
    sat_calls = 0
    nb_removed_refinement = 0
    nb_found_mr = 0
    
    nb_removed_refinement += len(core) - len(new_core)
    
    if assumption_removal:
        newly_removed = core - new_core
        for r in newly_removed:
            s += ~r # remove red constraint
    core = new_core
    
    found = set()

    # deletion-based MUS
    # order so that constraints with many variables are tried and removed first # TODO: heuristic like degree/sum_coefs or min vars needed for SAT are better PB heuristics, do we want to prioritise strong or weak constraints?
    for c in sorted(core, key=lambda c : -len(get_variables(dmap[c]))):
        # print(f"Checking {c}")
        if c not in core:
            continue # already removed
        core.remove(c)
        if redundancy_removal:
            red_constraint = ~dmap[c]
            red_var = cp.boolvar()
            s += red_var.implies(red_constraint) # add red constraint
            assumps = list(core) + [red_var]
        else:
            assumps = list(core)
        curr_time = time.time()
        if solver != "pysat:Cadical195":
            s.solve(assumptions=assumps, time_limit=time_limit-(curr_time-start_time), **kwargs)
        else:
            s.solve(assumptions=assumps)
        total_solve_time += time.time() - curr_time
        if s.status().exitstatus == ExitStatus.FEASIBLE:
            print("found const")
            sat_calls += 1
            core.add(c)
            found.add(c)
            # print(f"SAT when removing constraint, keeping it, core size {len(core)}")
            if assumption_removal:
                s += c # permanently set to true
        elif s.status().exitstatus == ExitStatus.UNSATISFIABLE:
            print("removing const")
            # UNSAT, use new solver core (clause set refinement)
            unsat_calls += 1
            new_core = set(s.get_core()).union(found)
            if redundancy_removal:
                s += ~red_var # remove red constraint
                if red_var in new_core:
                    continue
            nb_removed_refinement += len(core) - len(new_core)
            if assumption_removal:
                newly_removed = core - new_core
                for r in newly_removed:
                    s += ~r # permanently set to false
                s += ~c # permanently set to false
            core = new_core
        else:
            raise RuntimeError(f"MUS: solver returned unexpected status {s.status().exitstatus}")

    # return [dmap[avar] for avar in found], nb_removed_refinement, nb_found_mr, sat_calls, unsat_calls, total_solve_time
    return [dmap[c] for c in found], found, nb_removed_refinement, nb_found_mr, sat_calls, unsat_calls, total_solve_time

def mus(soft, hard=[], solver="ortools"):
    """
        A CP deletion-based MUS algorithm using assumption variables
        and unsat core extraction

        For solvers that support s.solve(assumptions=...) and s.get_core()

        Each constraint is an arbitrary CPMpy expression, so it can
        also be sublists of constraints (e.g. constraint groups),
        contain aribtrary nested expressions, global constraints, etc.

        Will extract an unsat core and then shrink the core further
        by repeatedly ommitting one assumption variable.

        :param: soft: soft constraints, list of expressions
        :param: hard: hard constraints, optional, list of expressions
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"mus requires a solver that supports assumption variables, use mus_naive with {solver} instead"

    # make assumption (indicator) variables and soft-constrained model
    (m, soft, assump) = make_assump_model(soft, hard=hard)
    s = cp.SolverLookup.get(solver, m)

    # create dictionary from assump to soft
    dmap = dict(zip(assump, soft))

    # setting all assump vars to true should be UNSAT
    assert not s.solve(assumptions=assump), "MUS: model must be UNSAT"
    core = set(s.get_core())  # start from solver's UNSAT core

    # deletion-based MUS
    # order so that constraints with many variables are tried and removed first
    for c in sorted(core, key=lambda c : -len(get_variables(dmap[c]))):
        if c not in core:
            continue # already removed
        core.remove(c)
        if s.solve(assumptions=list(core)) is True:
            core.add(c)
        else: # UNSAT, use new solver core (clause set refinement)
            core = set(s.get_core())

    return [dmap[avar] for avar in core]

def check_mus(mus, solver="exact"):
    """
        Checks that the given set of constraints is indeed a MUS.

        :param: mus: the set of constraints to check
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
    """

    # assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"mus requires a solver that supports assumption variables, use mus_naive with {solver} instead"
    
    # make assumption (indicator) variables and soft-constrained model
    (m, soft, assump) = make_assump_model(mus, name="mus_sel")
    s = cp.SolverLookup.get(solver, m)

    # create dictionary from assump to soft
    dmap = dict(zip(assump, soft))

    core = set(assump)  # start from all soft constraints

    # deletion-based MUS
    # order so that constraints with many variables are tried and removed first
    for c in sorted(core, key=lambda c : -len(get_variables(dmap[c]))):
        core.remove(c)
        s.solve(assumptions=assump)
        if s.status().exitstatus == ExitStatus.FEASIBLE:
            core.add(c)
        elif s.status().exitstatus == ExitStatus.UNSATISFIABLE:
            return False
        else:
            raise RuntimeError(f"MUS: solver returned unexpected status {s.status().exitstatus}")
        
    return True


def quickxplain(soft, hard=[], solver="ortools"):
    """
        Find a preferred minimal unsatisfiable subset of constraints, based on the ordering of constraints.

        A total order is imposed on the constraints using the ordering of `soft`.
        Constraints with lower index are preferred over ones with higher index

        Assumption-based implementation for solvers that support s.solve(assumptions=...) and s.get_core()
        More naive version available as `quickxplain_naive` to use with other solvers.

        :param: soft: list of soft constraints to find a preferred minimal unsatisfiable subset of
        :param: hard: list of hard constraints, will be added to the model before solving
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")

        CPMpy implementation of the QuickXplain algorithm by Junker:
            Junker, Ulrich. "Preferred explanations and relaxations for over-constrained problems." AAAI-2004. 2004.
            https://cdn.aaai.org/AAAI/2004/AAAI04-027.pdf
    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"quickxplain requires a solver that supports assumption variables, use quickxplain_naive with {solver} instead"

    model, soft, assump = make_assump_model(soft, hard)
    s = cp.SolverLookup.get(solver, model)

    assert s.solve(assumptions=assump) is False, "The model should be UNSAT!"
    dmap = dict(zip(assump, soft))

    # the recursive call
    def do_recursion(soft, hard, delta):

        if len(delta) != 0 and s.solve(assumptions=hard) is False:
            # conflict is in hard constraints, no need to recurse
            return []

        if len(soft) == 1:
            # conflict is not in hard constraints, but only 1 soft constraint
            return list(soft)  # base case of recursion

        split = len(soft) // 2  # determine split point
        more_preferred, less_preferred = soft[:split], soft[split:]  # split constraints into two sets

        # treat more preferred part as hard and find extra constants from less preferred
        delta2 = do_recursion(less_preferred, hard + more_preferred, more_preferred)
        # find which preferred constraints exactly
        delta1 = do_recursion(more_preferred, hard + delta2, delta2)
        return delta1 + delta2

    # optimization: find max index of solver core
    solver_core = frozenset(s.get_core())
    max_idx = max(i for i, a in enumerate(assump) if a in solver_core)

    core = do_recursion(list(assump)[:max_idx + 1], [], [])
    return [dmap[a] for a in core]

def ocus(soft, hard=[], weights=None, meta_constraint=True, solver="ortools", hs_solver="ortools", do_solution_hint=True):
    """
        Find an optimal and constrained MUS according to a linear objective function.
        By not providing a weightvector, this function will return the smallest mus.
        Works by iteratively generating correction subsets and computing optimal hitting sets to those enumerated sets.
        For better performance of the algorithm, use an incemental solver to compute the hitting sets such as Gurobi.

        Assumption-based implementation for solvers that support s.solve(assumptions=...)
        More naive version available as `optimal_mus_naive` to use with other solvers.

        :param: soft: list of soft constraints to find an optimal MUS of
        :param: hard: list of hard constraints, will be added to the model before solving
        :param: weights: list of weights for the soft constraints, will be used to compute the objective function
        :param: meta_constraint: a Boolean CPMpy expression that contains constraints in `soft` as sub-expressions.
            By not providing a meta_constraint, this function will return an optimal mus.
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: hs_solver: the hitting-set solver to use, ideally incremental such as "gurobi"
        :param: do_solution_hint: when true, will favor large satisfiable subsets generated by the SAT-solver

        CPMpy implementation loosely based on the "OCUS" algorithm from:

            Gamba, Emilio, Bart Bogaerts, and Tias Guns. "Efficiently explaining CSPs with unsatisfiable subset optimization."
            Journal of Artificial Intelligence Research 78 (2023): 709-746.
    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"ocus requires a solver that supports assumption variables, use ocus_naive with {solver} instead"
    
    model, soft, assump = make_assump_model(soft, hard)
    dmap = dict(zip(assump, soft)) # map assumption variables to constraints

    s = cp.SolverLookup.get(solver, model)
    if do_solution_hint and hasattr(s, 'solution_hint'): # algo is constructive, so favor large subsets
        s.solution_hint(assump, [1]*len(assump))

    assert s.solve(assumptions=assump) is False

    # initialize hitting set solver
    if weights is None:
        weights = np.ones(len(assump), dtype=int)

    hs_solver = cp.SolverLookup.get(hs_solver)
    hs_solver.minimize(cp.sum(assump * np.array(weights)))

    assump_constraint = replace_cons_with_assump(meta_constraint, dict(zip(soft, assump)))
    assert set(get_variables(assump_constraint)) <= set(assump), f"soft constraints should be replaced with assumption variables by now, but got {assump_constraint}"
    hs_solver += assump_constraint

    while hs_solver.solve() is True:
        hitting_set = [a for a in assump if a.value()]
        if s.solve(assumptions=hitting_set) is False:
            break

        # else, the hitting set is SAT, now try to extend it without extra solve calls.
        # Check which other assumptions/constraints are satisfied (using c.value())
        # complement of grown subset is a correction subset
        # Assumptions encode indicator constraints a -> c, find all false assumptions
        #   that really have to be false given the current solution.
        new_corr_subset = [a for a,c in zip(assump, soft) if a.value() is False and c.value() is False]
        hs_solver += cp.sum(new_corr_subset) >= 1

        # greedily search for other corr subsets disjoint to this one
        sat_subset = list(new_corr_subset)
        while s.solve(assumptions=sat_subset) is True:
            new_corr_subset = [a for a,c in zip(assump, soft) if a.value() is False and c.value() is False]
            sat_subset += new_corr_subset # extend sat subset with new corr subset, guaranteed to be disjoint
            hs_solver += cp.sum(new_corr_subset) >= 1 # add new corr subset to hitting set solver

    if hs_solver.status().exitstatus == ExitStatus.UNSATISFIABLE:
        raise OCUSException(f"No unsatisfiable subset adhereing to constraint {meta_constraint} could be found.")

    return [dmap[a] for a in hitting_set]


def optimal_mus(soft, hard=[], weights=None, solver="ortools", hs_solver="ortools", do_solution_hint=True):
    """
        Find an optimal MUS according to a linear objective function.
    """
    return ocus(soft, hard, weights, meta_constraint=True, solver=solver, hs_solver=hs_solver, do_solution_hint=do_solution_hint)

def smus(soft, hard=[], solver="ortools", hs_solver="ortools"):
    """
        Find a smallest MUS according, equivalent to `optimal_mus` with weights=None
    """
    return optimal_mus(soft, hard=hard, weights=None, solver=solver, hs_solver=hs_solver)


## Naive, non-assumption based versions of MUS-algos above
def mus_naive(soft, hard=[], solver="ortools"):
    """
        A naive pure CP deletion-based MUS algorithm

        Will repeatedly solve the problem from scratch with one less constraint
        For anything but tiny sets of constraints, this will be terribly slow.

        Best to only use for testing on solvers that do not support assumptions.
        For others, use `mus()`

        :param soft: soft constraints, list of expressions
        :param hard: hard constraints, optional, list of expressions
        :param solver: name of a solver, see SolverLookup.solvernames()
    """
    # ensure toplevel list
    soft = toplevel_list(soft, merge_and=False)

    m = cp.Model(hard + soft)
    assert not m.solve(solver=solver), "MUS: model must be UNSAT"

    mus = []
    # order so that constraints with many variables are tried and removed first
    core = sorted(soft, key=lambda c: -len(get_variables(c)))
    for i in range(len(core)):
        subcore = mus + core[i + 1:]  # check if all but 'i' makes core SAT

        if cp.Model(hard + subcore).solve(solver=solver):
            # removing it makes it SAT, must keep for UNSAT
            mus.append(core[i])
        # else: still UNSAT so don't need this candidate

    return mus


def quickxplain_naive(soft, hard=[], solver="ortools"):
    """
        Find a preferred minimal unsatisfiable subset of constraints, based on the ordering of constraints.

        A total order is imposed on the constraints using the ordering of `soft`.
        Constraints with lower index are preferred over ones with higher index

        Naive implementation, re-solving the model from scratch.
        Can be slower depending on the number of global constraints used and solver support for reified constraints.

        CPMpy implementation of the QuickXplain algorithm by Junker:
            Junker, Ulrich. "Preferred explanations and relaxations for over-constrained problems." AAAI-2004. 2004.
            https://cdn.aaai.org/AAAI/2004/AAAI04-027.pdf
    """

    soft = toplevel_list(soft, merge_and=False)
    assert cp.Model(hard + soft).solve(solver) is False, "The model should be UNSAT!"

    # the recursive call
    def do_recursion(soft, hard, delta):

        m = cp.Model(hard)
        if len(delta) != 0 and m.solve(solver) is False:
            # conflict is in hard constraints, no need to recurse
            return []

        if len(soft) == 1:
            # conflict is not in hard constraints, but only 1 soft constraint
            return list(soft)  # base case of recursion

        split = len(soft) // 2  # determine split point
        more_preferred, less_preferred = soft[:split], soft[split:]  # split constraints into two sets

        # treat more preferred part as hard and find extra constants from less preferred
        delta2 = do_recursion(less_preferred, hard + more_preferred, more_preferred)
        # find which preferred constraints exactly
        delta1 = do_recursion(more_preferred, hard + delta2, delta2)
        return delta1 + delta2

    core = do_recursion(soft, hard, [])
    return core

def ocus_naive(soft, hard=[], weights=None, meta_constraint=True, solver="ortools", hs_solver="ortools", do_solution_hint=True):
    """
        Naive implementation of `ocus` without assumption variables and incremental solving
    """
    soft = toplevel_list(soft, merge_and=False)
    bvs = cp.boolvar(shape=(len(soft),))

    if weights is None:
        weights = np.ones(len(bvs), dtype=int)
    hs_solver = cp.SolverLookup.get(hs_solver)
    hs_solver.minimize(cp.sum(bvs * np.array(weights)))

    bv_cons = replace_cons_with_assump(meta_constraint, dict(zip(soft, bvs)))
    assert set(get_variables(bv_cons)) <= set(bvs), f"soft constraints should be replaced with boolean variables by now, but got {bv_cons}"
    hs_solver += bv_cons

    while hs_solver.solve() is True:

        hitting_set = [c for bv, c in zip(bvs, soft) if bv.value()]
        if cp.Model(hard + hitting_set).solve(solver=solver) is False:
            break

        # else, the hitting set is SAT, now try to extend it without extra solve calls.
        # Check which other assumptions/constraints are satisfied using its value() function
        #       sidenote: some vars may not be know to model and are None!
        # complement of grown subset is a correction subset
        false_constraints = [s for s in soft if s.value() is False or s.value() is None]
        corr_subset = [bv for bv,c in zip(bvs, soft) if c in frozenset(false_constraints)]
        hs_solver += cp.sum(corr_subset) >= 1

        # find more corr subsets, disjoint to this one
        sat_subset = hitting_set + false_constraints
        while cp.Model(hard + sat_subset).solve(solver=solver) is True:
            false_constraints = [s for s in soft if s.value() is False or s.value() is None]
            corr_subset = [bv for bv, c in zip(bvs, soft) if c in frozenset(false_constraints)]
            hs_solver += cp.sum(corr_subset) >= 1
            sat_subset += false_constraints

    if hs_solver.status().exitstatus == ExitStatus.UNSATISFIABLE:
        raise OCUSException("No unsatisfiable constrained subset could be found") # TODO: better exception?

    return hitting_set


    

def optimal_mus_naive(soft, hard=[], weights=None, solver="ortools", hs_solver="ortools"):
    """
        Naive implementation of `optimal_mus` without assumption variables and incremental solving
    """
    return ocus_naive(soft, hard, weights, meta_constraint=True, solver=solver, hs_solver=hs_solver)

   


