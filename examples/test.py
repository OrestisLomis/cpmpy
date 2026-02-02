import cpmpy as cp
from cpmpy.tools.explain.mus import mus

bvs = cp.boolvar(shape=5)

def constraint_func(bvs):
    # some function that generates a list of cons
    constraints = []
    
    constraints.append(bvs[0] | bvs[1])
    constraints.append(~bvs[1] | ~bvs[2])
    constraints.append(bvs[2] | bvs[3])
    constraints.append(~bvs[3] | ~bvs[4])
    
    return cp.all(constraints)

model = cp.Model(
    constraint_func(bvs),
    ~bvs[0] & bvs[4]  
)

print(model.constraints)

model.solve(solver="exact")

assert len(model.constraints) >= len(mus(model.constraints))

