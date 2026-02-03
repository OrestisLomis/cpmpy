#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## dimacs.py
##
"""
    This file implements helper functions for exporting CPMpy models from and to DIMACS format.
    DIMACS is a textual format to represent CNF problems.
    The header of the file should be formatted as ``p cnf <n_vars> <n_constraints>``.
    If the number of variables and constraints are not given, it is inferred by the parser.

    Each remaining line of the file is formatted as a list of integers.
    An integer represents a Boolean variable and a negative Boolean variable is represented using a `'-'` sign.
"""


import cpmpy as cp

from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView, Operator

from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.to_cnf import to_cnf, to_gcnf
from cpmpy.transformations.get_variables import get_variables




