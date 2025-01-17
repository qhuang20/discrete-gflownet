import os
import pickle



def save_variables(run_dir, variables):
    for var_name, var_value in variables.items():
        with open(os.path.join(run_dir, f'{var_name}.pkl'), 'wb') as f:
            pickle.dump(var_value, f)

def load_variables(run_dir):
    variables = {}
    for filename in os.listdir(run_dir):
        if filename.endswith('.pkl'):
            var_name = filename[:-4]  # Remove .pkl extension
            with open(os.path.join(run_dir, filename), 'rb') as f:
                variables[var_name] = pickle.load(f)
    return variables

