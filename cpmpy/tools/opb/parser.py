#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## __init__.py
##
"""
OPB parser.

Currently only the restricted OPB PB24 format is supported (without WBO).


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_opb
"""


import os
import re
import sys
import lzma
import argparse
import cpmpy as cp
from io import StringIO
from typing import Union
from functools import reduce
from operator import mul

# Regular expressions
HEADER_RE = re.compile(r'(.*)\s*#variable=\s*(\d+)\s*#constraint=\s*(\d+).*')
TERM_RE = re.compile(r"([+-]?\d+)((?:\s+~?x\d+)+)")
OBJ_TERM_RE = re.compile(r'^min:')
IND_TERM_RE = re.compile(r'([>=|<=|=]+)\s+([+-]?\d+)')
IND_TERM_RE = re.compile(r'(>=|<=|=)\s*([+-]?\d+)')


def _parse_term(line, vars, flipped):
    """
    Parse a line containing OPB terms into a CPMpy expression.

    Supports:
        - Linear terms (e.g., +2 x1)
        - Non-linear terms (e.g., -1 x1 x14)
        - Negated variables using '~' (e.g., ~x5)

    Arguments:
        line (str):                 A string containing one or more terms.
        vars (list[cp.boolvar]):    List or array of CPMpy Boolean variables.

    Returns:
        cp.Expression: A CPMpy expression representing the sum of all parsed terms.

    Example:
        >>> _parse_term("2 x2 x3 +3 x4 ~x5", vars)
        sum([2, 3] * [(IV2*IV3), (IV4*~IV5)])
    """
    neg_sum = 0
    terms = []
    for w, vars_str in TERM_RE.findall(line):
        factors = []

        w = int(w)
        if flipped:
            w = -w
        for v in vars_str.split():
            if v.startswith("~x"):
                idx = int(v[2:])-1 # remove "~x"
                if w < 0:
                    factors.append(vars[idx])
                else:
                    factors.append(~vars[idx])
            else:
                idx = int(v[1:])-1 # remove "x"
                if w < 0:
                    factors.append(~vars[idx])
                else:
                    factors.append(vars[idx])
        if w < 0:
            w = -w
            neg_sum += w
        term = (w, reduce(mul, factors, 1)) # create weighted term

        terms.append(term)

    # return [w*t for w, t in terms], neg_sum
    
    # sort the terms to have a stable order
    terms.sort(key=lambda x: x[0], reverse=True)
    
    # make weighted sum
    weighted_terms = [w * t for w, t in terms]
    
    return cp.sum(weighted_terms), neg_sum

def _parse_constraint(line, vars):
    """
    Parse a single OPB constraint line into a CPMpy comparison expression.

    Arguments:
        line (str):                 A string representing a single OPB constraint.
        vars (list[cp.boolvar]):    List or array of CPMpy Boolean variables. Will be index to get the variables for the constraint.

    Returns:
        cp.expressions.core.Comparison: A CPMpy comparison expression representing
                                        the constraint.

    Example:
        >>> _parse_constraint("-1 x1 x14 -1 x1 ~x17 >= -1", vars)
        sum([-1, -1] * [(IV1*IV14), (IV1*~IV17)]) >= -1
    """
    # print(line)
    op, ind_term = IND_TERM_RE.search(line).groups()
    
    flipped = op == "<="
    lhs, neg_sum = _parse_term(line, vars, flipped)

    rhs = int(ind_term) if ind_term.lstrip("+-").isdigit() else vars[int(ind_term)]
    if flipped:
        rhs = -rhs + neg_sum
    else:
        rhs = rhs + neg_sum
        
    if op == "=":
        c1 = _parse_constraint(line.replace("=", ">="), vars)
        c2 = _parse_constraint(line.replace("=", "<="), vars)
            
        ret = [c1, c2]
        # print(f"returning two constraints for equality: {ret}")
        return ret
        
    else:
        if rhs <= 0:
            return None # no need to add a constraint that is always true
        ret = cp.expressions.core.Comparison(
            name=">=",
            left=lhs,
            right=rhs
        )
        # print(f"returning single constraint for {op}: {ret}")
        return ret

_std_open = open

def get_standardized_pb(line, vars):
    flipped = line[-3] == "<="
    rhs = int(line[-2])  # Right-hand side of the constraint
    if flipped:
        rhs = -rhs
    ws = []
    ls = []
    for i in range((len(line)-2) // 2):
        c = int(line[i*2])  # Coefficient
        if flipped:
            c = -c
        v = int(line[i*2+1][1:])
        if c > 0:
            ws.append(c)
            ls.append(vars[v-1])  # Adjust for 0-based indexing
        else:
            ws.append(-c)
            ls.append(~vars[v-1])
            rhs -= c
            
    if rhs <= 0:
        return [] # no need to add a constraint that is always true
    
    # Create the constraint
    return cp.sum([w * l for w, l in zip(ws, ls)]) >= rhs

def read_opb_file(filename, open=open):
    """
    Parser for OPB (Pseudo-Boolean) format. Reads in an instance and returns its matching CPMpy model.

    Based on PyPBLib's example parser: https://hardlog.udl.cat/static/doc/pypblib/html/library/index.html#example-from-opb-to-cnf-file

    Supports:
        - Linear and non-linear terms (e.g., -1 x1 x14 +2 x2)
        - Negated variables using '~' (e.g., ~x5)
        - Minimisation objective
        - Comparison operators in constraints: '=', '>='

    Arguments:
        opb (str or os.PathLike): 
            - A file path to an OPB file (optionally LZMA-compressed with `.xz`)
            - OR a string containing the OPB content directly
        open: (callable):
            If wcnf is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        cp.Model: The CPMpy model of the OPB instance.

    Example:
        >>> opb_text = '''
        ... * #variable= 5 #constraint= 2 #equal= 1 intsize= 64 #product= 5 sizeproduct= 13
        ... min: 2 x2 x3 +3 x4 ~x5 +2 ~x1 x2 +3 ~x1 x2 x3 ~x4 ~x5 ;
        ... 2 x2 x3 -1 x1 ~x3 = 5 ;
        ... '''
        >>> model = read_opb(opb_text)
        >>> print(model)
        Model(...)
    
    Notes:
        - Comment lines starting with '*' are ignored.
        - Only "min:" objectives are supported; "max:" is not recognized.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Read the first line to get the number of variables and constraints
    first_line = lines[0].strip().split()
    # print(first_line)
    n_vars = int(first_line[2])
    n_constraints = int(first_line[4])
    
    vars = cp.boolvar(shape=n_vars, name="v_i")
    if n_vars == 1:
        vars = cp.cpm_array([vars])
    
    m = cp.Model()

    # Read each constraint line
    for line in lines[1:]:
        if line.startswith("c") or line.startswith("p") or line.startswith("*"):
            continue  # Skip comments and problem line
        line = line.strip().split()
        if line[-3] == "=":
            c1 = line.copy()
            c1[-3] = ">="
            c2 = line.copy()
            c2[-3] = "<="
            cons = [c1, c2]
        else:
            cons = [line]
        for c in cons:
            m += get_standardized_pb(c, vars)

    return m, vars

def read_opb(opb: Union[str, os.PathLike], open=open) -> cp.Model:
    """
    Parser for OPB (Pseudo-Boolean) format. Reads in an instance and returns its matching CPMpy model.

    Based on PyPBLib's example parser: https://hardlog.udl.cat/static/doc/pypblib/html/library/index.html#example-from-opb-to-cnf-file

    Supports:
        - Linear and non-linear terms (e.g., -1 x1 x14 +2 x2)
        - Negated variables using '~' (e.g., ~x5)
        - Minimisation objective
        - Comparison operators in constraints: '=', '>='

    Arguments:
        opb (str or os.PathLike): 
            - A file path to an OPB file (optionally LZMA-compressed with `.xz`)
            - OR a string containing the OPB content directly
        open: (callable):
            If wcnf is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        cp.Model: The CPMpy model of the OPB instance.

    Example:
        >>> opb_text = '''
        ... * #variable= 5 #constraint= 2 #equal= 1 intsize= 64 #product= 5 sizeproduct= 13
        ... min: 2 x2 x3 +3 x4 ~x5 +2 ~x1 x2 +3 ~x1 x2 x3 ~x4 ~x5 ;
        ... 2 x2 x3 -1 x1 ~x3 = 5 ;
        ... '''
        >>> model = read_opb(opb_text)
        >>> print(model)
        Model(...)
    
    Notes:
        - Comment lines starting with '*' are ignored.
        - Only "min:" objectives are supported; "max:" is not recognized.
    """

    
    # If opb is a path to a file -> open file
    if isinstance(opb, (str, os.PathLike)) and os.path.exists(opb):
        if open is not None:
            f = open(opb)
        else:
            f = _std_open(opb, "rt")
    # If opb is a string containing a model -> create a memory-mapped file
    else:
        f = StringIO(opb)

    # Look for header on first line
    line = f.readline()
    header = HEADER_RE.match(line)
    if not header: # If not found on first line, look on second (happens when passing multi line string)
        _line = f.readline()
        header = HEADER_RE.match(_line)
        if not header:
            raise ValueError(f"Missing or incorrect header: \n0: {line}1: {_line}2: ...")
    nr_vars = int(header.group(2)) #+ 1

    # Generator without comment lines
    reader = (l for l in map(str.strip, f) if l and l[0] != '*')

    # CPMpy objects
    vars = cp.boolvar(shape=nr_vars, name="x")
    if nr_vars == 1:
        vars = cp.cpm_array([vars])
    model = cp.Model()
    
    # Special case for first line -> might contain objective function
    first_line = next(reader)
    if OBJ_TERM_RE.match(first_line):
        obj_expr = _parse_term(first_line, vars)
        model.minimize(obj_expr)
    else: # no objective found, parse as a constraint instead
        cons = _parse_constraint(first_line, vars)
        if isinstance(cons, list):
            for c in cons:
                if c is not None:
                    model.add(c)
        else:
            model.add(cons)

    # Start parsing line by line
    for line in reader:
        cons = _parse_constraint(line, vars)
        if isinstance(cons, list):
            for c in cons:
                if c is not None:
                    model.add(c)
        else:
            model.add(cons)

    return model


def main():
    parser = argparse.ArgumentParser(description="Parse and solve an OPB model using CPMpy")
    parser.add_argument("model", help="Path to an OPB file (or raw OPB string if --string is given)")
    parser.add_argument("-s", "--solver", default=None, help="Solver name to use (default: CPMpy's default)")
    parser.add_argument("--string", action="store_true", help="Interpret the first argument (model) as a raw OPB string instead of a file path")
    parser.add_argument("-t", "--time-limit", type=int, default=None, help="Time limit for the solver in seconds (default: no limit)")
    args = parser.parse_args()

    # Build the CPMpy model
    try:
        if args.string:
            model = read_opb(args.model)
        else:
            model = read_opb(os.path.expanduser(args.model))
    except Exception as e:
        sys.stderr.write(f"Error reading model: {e}\n")
        sys.exit(1)

    # Solve the model
    try:
        if args.solver:
            result = model.solve(solver=args.solver, time_limit=args.time_limit)
        else:
            result = model.solve(time_limit=args.time_limit)
    except Exception as e:
        sys.stderr.write(f"Error solving model: {e}\n")
        sys.exit(1)

    # Print results
    print("Status:", model.status())
    if result is not None:
        if model.has_objective():
            print("Objective:", model.objective_value())
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
