#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## __init__.py
##
"""
Set of utilities for working with CNF files.


==================
List of submodules
==================

.. autosummary::
    :nosignatures:

    parser
"""

from .parser import read_dimacs, write_dimacs
