import pickle
import os # Useful for defining the path
import cpmpy as cp
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.tools.explain.mus import mus, pb_mus
from cpmpy.tools.explain.utils import make_assump_model
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.int2bool import int2bool
from cpmpy.transformations.linearize import canonical_comparison, linearize_constraint, only_ge_comparison, only_positive_bv, only_positive_coefficients, sorted_coefficients
from cpmpy.transformations.normalize import simplify_boolean, toplevel_list
from cpmpy.transformations.reification import only_bv_reifies, only_implies, reify_rewrite

# 1. Define the path to your pickle file
pickle_file_path = '/home/orestis_ubuntu/work/benchmarks/2024/cp-bench-select/isbn_var_12_val_6.pkl'

# 2. Open the file in read binary mode ('rb')
try:
    with open(pickle_file_path, 'rb') as file:
        # 3. Use pickle.load() to deserialize the data
        loaded_data = pickle.load(file)
    
except FileNotFoundError:
    print(f"File not found: {pickle_file_path}")
    loaded_data = None
    
if loaded_data is not None:
    model = loaded_data
    print("Model loaded successfully from pickle file.")
    
    
    # model = cp.Model()
    
    # vars = cp.intvar(0,10,shape=2)
    
    # model += ((vars[0] % 4) == vars[1])
    
    # print(model)
    
    model.solve(solver="pumpkin")
    
    print(model.status())
    
    # print(vars.value())

    # assump_m, _, assumps = make_assump_model(model.constraints, name="sel")
