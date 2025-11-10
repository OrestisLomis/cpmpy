import pickle
import os # Useful for defining the path
import cpmpy as cp

# 1. Define the path to your pickle file
pickle_file_path = '/home/orestis_ubuntu/work/cpmpyfork/cp-mus-bench/Instance2-0-0-0.pkl'

# 2. Open the file in read binary mode ('rb')
try:
    with open(pickle_file_path, 'rb') as file:
        # 3. Use pickle.load() to deserialize the data
        loaded_data = pickle.load(file)
    
    print(f"Successfully loaded data from {pickle_file_path}.")
    print(f"Type of loaded data: {type(loaded_data)}")
    
    # print(loaded_data.constraints)
    
    from cpmpy.tools.explain.breakid import BreakID
    from cpmpy.tools.explain.breakid import BREAKID_PATH
    
    breakid = BreakID(BREAKID_PATH)
    
    model = cp.Model(loaded_data.constraints)
    
    # model.solve(solver="pindakaas")
    
    opb_input = breakid.get_input(model.constraints, format="opb")
    
    print(opb_input)
    
    # You can now work with the loaded_data object (e.g., a list, dictionary, or model)
    # print(loaded_data)

except FileNotFoundError:
    print(f"Error: The file '{pickle_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred during loading: {e}")