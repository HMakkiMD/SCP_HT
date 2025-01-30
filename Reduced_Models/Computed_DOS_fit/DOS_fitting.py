import random
import numpy as np
import math
import sys
import os
from collections import OrderedDict

def checkfile(filename):

    if not os.path.isfile(filename):
        print(" File %s not found!" % filename)
        sys.exit()

def fill_dict(filename):
    '''Fills a dictionary of options.
       Courtesy of Daniele Padula'''

    opts = OrderedDict()
    try:
        checkfile(filename) # Check if the file exists (subroutine)
        with open(filename) as f:
            for line in f:

                # Ignore comments and empty lines
                if line.startswith('#'): #Comments
                    continue

                if not line.strip(): #Empty lines
                    continue

                # AL: Strippo i caratteri inutili
                line = line.replace('=', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = line.replace(',', ' ') # commas with spaces
                line = line.replace('"', ' ') # quotes with spaces
                line = line.replace("'", " ") # single quotes with spaces

                key_dict = line.split()[0] # Keys of dictionary
                data = line.split()[1:] # Values of dictionary

                # Try to understand whether data should be stored as int, float
                # or string
                try:
                    data = np.array(list(map(int, data)))

                # This can occur with both floats and strings
                except ValueError:
                    try:
                        data = np.array(list(map(float, data)))

                    except ValueError:
                        data = np.array(list(map(str, data)))

                if len(data) == 1: # If 1 value only, then extract from list
                    opts[key_dict] = data[0]
                else:
                    opts[key_dict] = data

    except TypeError:
        pass

    return opts

if len(sys.argv) == 1:
        print('ERROR: You need to indicate the input file name in the command line')
        print()
        print("   Use of the script: python DOS_fitting.py input_filename    ")
        sys.exit(-1)
filename = sys.argv[1]
opts = fill_dict(filename) # Read input file

# Set variables based on those specified in input file
if "num_iterations" in opts:
    num_iterations = opts["num_iterations"]
if "initial_temp" in opts:
    initial_temp = opts["initial_temp"]
if "cooling_rate" in opts:
    cooling_rate = opts["cooling_rate"]
if "HOMO_LUMO_dir" in opts:
    HOMO_LUMO_dir = opts["HOMO_LUMO_dir"]
if "DOS_dir" in opts:
    DOS_dir = opts["DOS_dir"]
if "polymer_types" in opts:
    polymer_types = opts["polymer_types"]
if "diag_disorder" in opts:
    diag_disorder = opts["diag_disorder"]
if "off_diag_disorder" in opts:
    off_diag_disorder = opts["off_diag_disorder"]
if "rigid_alpha_beta_dir" in opts:
    rigid_alpha_beta_dir = opts["rigid_alpha_beta_dir"]

def find_closest_index(lst, target):
    """
    Finds the index of the element in the list `lst` that is closest to the `target` value.
    
    Parameters:
    lst (numpy array): The array of floats to search through.
    target (float): The target float value to find the closest element to.
    
    Returns:
    int: The index of the closest element in the array.
    """
    lst = np.asarray(lst)  # Ensure input is a numpy array
    return np.argmin(np.abs(lst - target))

def broaden(E, sigma, Emin, Emax, dE):
    NP = int((Emax - Emin) / dE)
    X = np.linspace(Emin, Emax, NP)
    C = -1 / (2 * sigma**2)

    # Vectorized Gaussian broadening
    f = np.sum(np.exp(C * (X[:, np.newaxis] - E)**2), axis=1)
    
    f *= 0.39894228 / sigma  # Gaussian normalization
    
    return X, f

def integrate_between_indices(x_vals, y_vals, idx_end, HOMO):
    
    integrals = []
    
    if HOMO == True:
        for i in range(0,300):
            integral = np.trapz(y_vals[idx_end-i:idx_end+1], x_vals[idx_end-i:idx_end+1])
            integrals.append(integral)
            if abs(1.1- integral) < 0.01:
                break
        
    else:
        for i in range(0,300):
            integral = np.trapz(y_vals[idx_end:idx_end+i], x_vals[idx_end:idx_end+i])
            integrals.append(integral)
            if abs(1.1- integral) < 0.01:
                break
        
    tocheck_start = (find_closest_index(integrals, 0.005))
    tocheck_end = (find_closest_index(integrals,0.33))
    
    return tocheck_start, tocheck_end

def homopolymer(N, D, V, sigma1, sigma2, nrep, diag_disorder=None, off_diag_disorder=None):
    # Pre-allocate result array for eigenvalues
    A = np.zeros(N * nrep)
    
    # Pre-generate random disorder if not provided
    if diag_disorder is None:
        diag_disorder = np.random.normal(size=(nrep, N))
    if off_diag_disorder is None:
        off_diag_disorder = np.random.normal(size=(nrep, N - 1))

    # Pre-allocate matrix H once and reuse
    H = np.zeros((N, N))

    idx = 0
    for j in range(nrep):
        # Diagonal and off-diagonal elements (reuse pre-allocated arrays)
        diag_elements = D + sigma1 * diag_disorder[j]
        off_diag_elements = -V + sigma2 * off_diag_disorder[j]

        # Set diagonal and off-diagonal elements in H
        np.fill_diagonal(H, diag_elements)
        H[np.arange(N - 1), np.arange(1, N)] = off_diag_elements
        H[np.arange(1, N), np.arange(N - 1)] = off_diag_elements

        # Compute eigenvalues (eigh is faster for symmetric matrices)
        E, _ = np.linalg.eigh(H)

        # Store the eigenvalues in the result array
        A[idx:idx + N] = E
        idx += N

    return A, H  # Returning the last H matrix


def model(param1,param2,param3,param4,param5,param6,polymer,HOMO):
    # Example model function, replace with your actual model
    
    if HOMO == True:
        starting_point_iso = find_closest_index(DOS_iso[:,0], BG_midpoint)
        tocheck_start_iso, tocheck_end_iso = integrate_between_indices(DOS_iso[:,0],DOS_iso[:,1], starting_point_iso,HOMO)  
    else:
        starting_point_iso = find_closest_index(DOS_iso[:,0], BG_midpoint)
        tocheck_start_iso, tocheck_end_iso = integrate_between_indices(DOS_iso[:,0],DOS_iso[:,1], starting_point_iso,HOMO)  
        
    if HOMO == True:
        starting_point_pc = find_closest_index(DOS_pc[:,0], BG_midpoint)
        tocheck_start_pc, tocheck_end_pc = integrate_between_indices(DOS_pc[:,0],DOS_pc[:,1], starting_point_pc,HOMO)  
    else:
        starting_point_pc = find_closest_index(DOS_pc[:,0], BG_midpoint)
        tocheck_start_pc, tocheck_end_pc = integrate_between_indices(DOS_pc[:,0],DOS_pc[:,1], starting_point_pc,HOMO)  
    
    alpha_iso = param1
    alpha_pc = param2
    beta = param3 * set_beta
    sigma_alpha = param4 
    sigma_alpha_pc = param5 * sigma_alpha
    sigma_beta = param6 * sigma_alpha 
    
    D,H = homopolymer(10,alpha_iso,beta,sigma_alpha,sigma_beta,250,diag_disorder,off_diag_disorder)
    D = np.sort(D)
    X,f = (broaden(D,0.025,-8,0,0.01))
    f_iso = f/2500
    
    D,H = homopolymer(10,alpha_pc,beta,sigma_alpha_pc,sigma_beta,250,diag_disorder,off_diag_disorder)
    D = np.sort(D)
    X,f = (broaden(D,0.025,-8,0,0.01))
    f_pc = f/2500
    
    return f_iso,f_pc,starting_point_iso,starting_point_pc,tocheck_start_iso,tocheck_start_pc,tocheck_end_iso,tocheck_end_pc

def objective_function(param1,param2,param3,param4,param5,param6,polymer,HOMO):
    f_iso,f_pc, starting_point_iso,starting_point_pc,tocheck_start_iso,tocheck_start_pc,tocheck_end_iso,tocheck_end_pc = model(param1,param2,param3,param4,param5,param6,polymer,HOMO)
    
    if HOMO == True:
        abs_diff = []
        for i in range(starting_point_iso-tocheck_end_iso,starting_point_iso-tocheck_start_iso):
            abs_diff.append(np.square(f_iso[i]-DOS_iso[:,1][i]))
        
        integral_iso = np.trapz(abs_diff, DOS_iso[:,0][starting_point_iso-tocheck_end_iso:starting_point_iso-tocheck_start_iso])
    
        abs_diff = []
        for i in range(starting_point_pc-tocheck_end_pc,starting_point_pc-tocheck_start_pc):
            abs_diff.append(np.square(f_pc[i]-DOS_pc[:,1][i]))
        
        integral_pc = np.trapz(abs_diff, DOS_pc[:,0][starting_point_pc-tocheck_end_pc:starting_point_pc-tocheck_start_pc])
    
    else:
        abs_diff = []
        for i in range(starting_point_iso+tocheck_start_iso,starting_point_iso+tocheck_end_iso):
            abs_diff.append(np.square(f_iso[i]-DOS_iso[:,1][i]))
        
        integral_iso = np.trapz(abs_diff, DOS_iso[:,0][starting_point_iso+tocheck_start_iso:starting_point_iso+tocheck_end_iso])
        
        abs_diff = []
        for i in range(starting_point_pc+tocheck_start_pc,starting_point_pc+tocheck_end_pc):
            abs_diff.append(np.square(f_pc[i]-DOS_pc[:,1][i]))
        
        integral_pc = np.trapz(abs_diff, DOS_pc[:,0][starting_point_pc+tocheck_start_pc:starting_point_pc+tocheck_end_pc])
        
    integral = (integral_iso + integral_pc) / 2
            
    return integral,starting_point_iso,starting_point_pc,tocheck_start_iso,tocheck_start_pc,tocheck_end_iso,tocheck_end_pc

# Perturb a single parameter slightly
def perturb_parameter(params, param_idx, step_sizes):
    new_params = list(params)  # Copy the original parameters
    perturbation = random.choice([-step_sizes[param_idx], step_sizes[param_idx]])
    new_params[param_idx] += perturbation  # Perturb only the selected parameter
    return tuple(new_params)

def monte_carlo_with_annealing(initial_params, num_iterations, param_ranges, initial_temp, cooling_rate, step_size, polymer, HOMO):
    current_params = initial_params
    current_cost, *_ = objective_function(*current_params, polymer, HOMO)
    best_params = current_params
    best_cost = current_cost
    temperature = initial_temp

    epsilon = []
    consecutive_worse_moves = 0  # Counter for consecutive rejected moves
    max_consecutive_worse_moves = 200  # Stop after 200 consecutive worse moves

    for iteration in range(num_iterations):
        param_idx = random.randint(0, 5)
        new_params = perturb_parameter(current_params, param_idx, step_size)
        new_params = tuple(max(min(new_params[i], param_ranges[i][1]), param_ranges[i][0]) for i in range(6))

        new_cost, starting_point_iso, starting_point_pc, tocheck_start_iso, tocheck_start_pc, tocheck_end_iso, tocheck_end_pc = objective_function(*new_params, polymer,HOMO)

        delta_epsilon = new_cost - current_cost
        if delta_epsilon < 0 or random.uniform(0, 1) < math.exp(-delta_epsilon / temperature):
            current_params = new_params
            current_cost = new_cost
            epsilon.append(current_cost)

            if delta_epsilon < 0:
                consecutive_worse_moves = 0 # Reset on better moves

            if delta_epsilon > 0:
                consecutive_worse_moves += 1  # Increment on accepted worse moves
        else:
            consecutive_worse_moves += 1  # Increment on rejected worse moves

        # Check for improvement in best cost
        if current_cost < best_cost:
            best_params = current_params
            best_cost = current_cost

        # Update temperature
        temperature *= cooling_rate
       
        # Check if consecutive worse moves have reached the limit
        if consecutive_worse_moves >= max_consecutive_worse_moves:
            print(f"Stopping early at iteration {iteration} due to {max_consecutive_worse_moves} consecutive worse moves.")
            break
    
    return best_params, best_cost, starting_point_iso, starting_point_pc, tocheck_start_iso, tocheck_start_pc, tocheck_end_iso, tocheck_end_pc

diag_disorder = np.loadtxt(diag_disorder)
off_diag_disorder = np.loadtxt(off_diag_disorder)

polymers_type = []
with open(polymer_types, 'r') as f: # get list of polymer types (p- or n-) which determines fitting ranges
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        split_line = line.split()
        polymers_type.append(split_line)

# Example usage
if __name__ == "__main__":
    # Define initial parameters and their ranges
    for i in range(len(polymers_type)):
        polymer = polymers_type[i][0]

        HOMOs_LUMOs = np.loadtxt(HOMO_LUMO_dir + f"HOMO_LUMO_{polymer}.txt") # extract HOMO and LUMO values
        HOMO = HOMOs_LUMOs[0]
        LUMO = HOMOs_LUMOs[1]
        
        BG_midpoint = 0.5*(HOMO+LUMO) # Midpoint of HOMO and LUMO determines integration starting point
        
        alpha_beta = np.loadtxt(rigid_alpha_beta_dir + f"{polymer}.txt")
        
        DOS_iso = np.loadtxt(DOS_dir + f"{polymer}_DOS_0025_iso.txt")
        DOS_pc = np.loadtxt(DOS_dir + f"{polymer}_DOS_0025_pc.txt")
        
        set_alpha = alpha_beta[0]
        set_beta = alpha_beta[1]
        
        print(f'Set alpha for {polymer} is: {set_alpha}')
        print(f'Set beta for {polymer} is: {set_beta}')
        
        initial_params = (set_alpha-0.15, set_alpha-0.15, 1, 0.1, 2.5, 0.5)  # Initial guess for the parameters
        param_ranges = [(set_alpha-0.30, set_alpha+0.30), (set_alpha-0.30, set_alpha+0.30), (0.2, 1.8), (0, 0.30), (1, 5), (0, 1)]
        
        step_sizes = [0.01, 0.01, 0.005, 0.001, 0.025, 0.01]      # Step size for perturbations
    
        # Perform the search
        
        if polymers_type[i][1] == 'p': 
            HOMO = True
            best_params, best_cost, starting_point_iso,starting_point_pc,tocheck_start_iso,tocheck_start_pc,tocheck_end_iso,tocheck_end_pc = monte_carlo_with_annealing(initial_params, num_iterations, param_ranges, initial_temp, cooling_rate, step_sizes,polymer,HOMO)
        else:
            HOMO = False
            best_params, best_cost, starting_point_iso,starting_point_pc,tocheck_start_iso,tocheck_start_pc,tocheck_end_iso,tocheck_end_pc = monte_carlo_with_annealing(initial_params, num_iterations, param_ranges, initial_temp, cooling_rate, step_sizes,polymer,HOMO)        

        alpha = best_params[0]
        alpha_pc = best_params[1]
        beta = set_beta * best_params[2]
        sigma_alpha_iso = best_params[3]
        sigma_alpha_pc = best_params[3] * best_params[4]
        sigma_beta = best_params[3] * best_params[5]
        
        best_params_iso = alpha, beta, sigma_alpha_iso, sigma_beta
        best_params_iso = tuple([float("{}".format(n)) for n in best_params_iso])
        best_params_pc = alpha_pc, beta, sigma_alpha_pc, sigma_beta
        best_params_pc = tuple([float("{}".format(n)) for n in best_params_pc])
        
        print(f"Best parameters found for iso DOS: {best_params_iso}")
        
        print(f"Best parameters found for embedded DOS: {best_params_pc}")
        print(f"Best cost: {best_cost:.4f}")
