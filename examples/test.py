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

# # 1. Define the path to your pickle file
# pickle_file_path = '/home/orestis_ubuntu/work/cpmpyfork/cp-mus-bench/Instance1-0-0-1.pkl'

# # 2. Open the file in read binary mode ('rb')
# try:
#     with open(pickle_file_path, 'rb') as file:
#         # 3. Use pickle.load() to deserialize the data
#         loaded_data = pickle.load(file)
    
#     print(f"Successfully loaded data from {pickle_file_path}.")
#     print(f"Type of loaded data: {type(loaded_data)}")
    
#     # print(loaded_data.constraints)
    
#     from cpmpy.tools.explain.breakid import BreakID
#     from cpmpy.tools.explain.breakid import BREAKID_PATH
#     from cpmpy.solvers.pindakaas import CPM_pindakaas
    
#     breakid = BreakID(BREAKID_PATH)
    
#     model = cp.Model(loaded_data.constraints)
    
#     constraints = model.constraints
    
#     # print(constraints)
    
#     # vars = cp.intvar(1,3,shape=4,name='cell')
#     # # var = cp.intvar(1,3)
    
    
#     # constraints = [cp.alldifferent(vars)]
    
#     model, hard, assumps = make_assump_model(constraints, name="sel")
    
#     constraints = model.constraints
#     # print(constraints[0][:10])
#     # constraints = [var >= 2]
    
#     # print(constraints)
    
#     # (sel[329]) -> (sum([count([nv[0,9],nv[1,9],nv[2,9],nv[3,9],nv[4,9],nv[5,9],nv[6,9],nv[7,9]],1), -(IV18), IV19]) == 4),
    
#     slv = CPM_pysat()
    
#     constraints = toplevel_list(constraints)
#     constraints = decompose_in_tree(constraints,supported=frozenset({'alldifferent'}), supported_reified=frozenset({'alldifferent'}), csemap=slv._csemap)  # Alldiff has a specialzed MIP decomp
#     constraints = simplify_boolean(constraints)
#     constraints = flatten_constraint(constraints, csemap=slv._csemap)  # flat normal form
#     constraints = reify_rewrite(constraints, supported=frozenset(['sum', 'wsum', 'alldifferent']), csemap=slv._csemap)  # constraints that support reification
#     constraints = only_numexpr_equality(constraints, supported=frozenset(["sum", "wsum", 'alldifferent']), csemap=slv._csemap)  # supports >, <, !=
#     constraints = only_bv_reifies(constraints, csemap=slv._csemap)
#     constraints = only_implies(constraints, csemap=slv._csemap)  # anything that can create full reif should go above...
#     constraints = linearize_constraint(constraints, supported=frozenset({"sum", "wsum"}), csemap=slv._csemap)  # the core of the MIP-linearization
#     # constraints = int2bool(constraints, slv.ivarmap, encoding="binary")
#     constraints = canonical_comparison(constraints)
#     # constraints = only_ge_comparison(constraints)
#     # constraints = only_positive_coefficients(constraints)
#     # constraints = sorted_coefficients(constraints)
    
#     model = cp.Model(constraints)
    
#     count = [0]*len(assumps)
    
#     for cons in constraints:
#         if cons.name == "->" and cons.args[0] in assumps:
#             g = int(cons.args[0].name.replace("sel[", "").replace("]", ""))
#             count[g] += 1
#             print(cons)
#     print(count)
    
# except Exception as e:
#     raise e


vars = cp.intvar(1,3,shape=3, name="vars")
model = cp.Model()

model += cp.alldifferent(vars)

assump_m, _, assumps = make_assump_model(model.constraints, name="sel")

print("Original model:")
print(assump_m)

slv = CPM_pysat()

constraints = assump_m.constraints

constraints = toplevel_list(constraints)
constraints = decompose_in_tree(constraints,supported=frozenset({'alldifferent'}), supported_reified=frozenset({'alldifferent'}), csemap=slv._csemap)  # Alldiff has a specialzed MIP decomp
constraints = simplify_boolean(constraints)
constraints = flatten_constraint(constraints, csemap=slv._csemap)  # flat normal form
constraints = reify_rewrite(constraints, supported=frozenset(['sum', 'wsum', 'alldifferent']), csemap=slv._csemap)  # constraints that support reification
constraints = only_numexpr_equality(constraints, supported=frozenset(["sum", "wsum", 'alldifferent']), csemap=slv._csemap)  # supports >, <, !=
constraints = only_bv_reifies(constraints, csemap=slv._csemap)
constraints = only_implies(constraints, csemap=slv._csemap)  # anything that can create full reif should go above...
constraints = linearize_constraint(constraints, supported=frozenset({"sum", "wsum"}), csemap=slv._csemap)  # the core of the MIP-linearization
constraints = int2bool(constraints, slv.ivarmap, encoding=slv.encoding)
print("\nTransformed constraints:")
print(cp.cpm_array(constraints))

# constraints = canonical_comparison(constraints)
# constraints = only_ge_comparison(constraints)
# constraints = only_positive_coefficients(constraints)
# constraints = sorted_coefficients(constraints)