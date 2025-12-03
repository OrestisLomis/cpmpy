#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## __init__.py
##
"""
Dummy CP parser.. the models are read from pickle so just pass


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
from cpmpy.model import Model


def read_cp(cp, open=open) -> cp.Model:
    """
    Dummy parser for CP models stored in pickle files.
    """

    return Model.from_file(cp)
