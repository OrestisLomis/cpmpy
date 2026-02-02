import cpmpy as cp 

from cpmpy.tools.explain.marco import marco
from cpmpy.tools.explain.mus import mus

# Define the model

model = cp.Model()

bools = cp.boolvar(shape=1, name="b")

cells = cp.intvar(1, 9, shape=(9, 9), name="cells")

# given clues

model += cells[0, 0] == 1
model += cells[0, 8] == 1
# model += cells[0, 1] == 2
# model += cells[0, 2] == 3

# model += cells[1, 4] == 4
# model += cells[2, 7] == 4

# constraints

def regroup_to_blocks(grid):
    # Create an empty list to store the blocks
    blocks = [[] for _ in range(9)]

    for row_index in range(9):
        for col_index in range(9):
            # Determine which block the current element belongs to
            block_index = (row_index // 3) * 3 + (col_index // 3)
            # Add the element to the appropriate block
            blocks[block_index].append(grid[row_index][col_index])

    return blocks

blocks = regroup_to_blocks(cells)

for i in range(cells.shape[0]):
    model += cp.AllDifferent(cells[i,:])
    model += cp.AllDifferent(cells[:,i])
    model += cp.AllDifferent(blocks[i])
    
    
print(f"single mus: {mus(model.constraints)}")
    
muses = marco(model.constraints, solver="exact", map_solver="exact", return_mus=True, return_mcs=False)

for mus in muses:
    print("MUS:")
    for c in mus[1]:
        print(c) 
    print("-----")