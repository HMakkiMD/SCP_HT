import numpy as np
from cclib import ccopen
import sys
from collections import OrderedDict
import os

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
        print("   Use of the script: python Rigid_chain_fit.py input_filename    ")
        sys.exit(-1)
filename = sys.argv[1]
opts = fill_dict(filename) # Read input file

# Set variables based on those specified in input file
if "input_dir" in opts:
    input_dir = opts["input_dir"]
if "polymer_types" in opts:
    polymer_types = opts["polymer_types"]
if "nsamples" in opts:
    nsamples = opts["nsamples"]
if "output_dir" in opts:
    output_dir = opts["output_dir"]
    
def homopolymer(N,D,V,sigma1,sigma2,nrep): # Function to generate model Hamiltonian
    A=[]
    
    for j in range (nrep):
        H=np.zeros((N,N)) #initialise matrix

        for i in range(N):
            H[i,i]=D+sigma1*np.random.normal() # diagonal elements

        for i in range(N-1):
            H[i,i+1]=-V+sigma2*np.random.normal() # off-diagonal elements
            H[i+1,i]=H[i,i+1]

        (E,c)=np.linalg.eig(H)
        A=np.concatenate((A,E))

    return A,H

def fit_to_rigid_chain(polymer,nsamples, HOMO=True):
    
    fullpath = input_dir + f"{polymer}_10.log" # log file is standard output from single point calculation on rigid 10-mer

    data = ccopen(fullpath).parse()
    dhomoidx = data.homos[0] # get index of HOMO
    MO_energies = data.moenergies[0] # get all MO energies

    if HOMO == True: # Set range of indices of MO energies we are interested in (if HOMO or LUMO) and the parameter ranges
        tocheck = MO_energies[dhomoidx-4:dhomoidx+1] 
        param_ranges = [
            (-6.0, -4.0),  # Range for alpha
            (0.01, 0.385) # Range for beta 
        ]
    else:
        tocheck = MO_energies[dhomoidx+1:dhomoidx+6]
        param_ranges = [
            (-4.5, -2.5),  # Range for alpha 
            (0.01, 0.385) # Range for beta 
        ]
        
    curr_diff_min = 1000 
    for i in range(nsamples):
        
        params = [np.random.uniform(low, high) for low, high in param_ranges] # generate random sample of alpha and beta from initial ranges
        
        alpha,beta = params
        
        curr_difference = []
        D,H = homopolymer(10,alpha,beta,0,0,1) # get model eigenvalues and Hamiltonian matrix (last not needed)
        D = np.sort(D) # sort model eigenvalues 
        
        if HOMO == True: # if HOMO only fitting last five orbitals, if LUMO only first five orbitals
            D = D[5:]
        else:
            D = D[:5]
            
        for i in range(len(tocheck)): # objective function = mean absolute difference between model and DFT
            curr_difference.append(np.square(D[i]-tocheck[i]))
        if np.mean(curr_difference) < curr_diff_min:
             curr_diff_min = np.mean(curr_difference)
             opt_alpha = alpha
             opt_beta = beta
        else:
            continue
          
    return opt_alpha, opt_beta,polymer, curr_diff_min, tocheck

polymers_type = []
with open(polymer_types, 'r') as f: # get list of polymer types (p- or n-) which determines fitting ranges
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        split_line = line.split()
        polymers_type.append(split_line)
        
all_cost = []

for i in range(len(polymers_type)):
    opt_params = []
    
    polymer = polymers_type[i][0]
    
    if polymers_type[i][1] == 'p':       
        opt_alpha, opt_beta, polymer, curr_diff_min, tocheck = fit_to_rigid_chain(polymer, nsamples, HOMO=True)    
        D,H = homopolymer(10,opt_alpha,opt_beta,0,0,1)
        D = np.sort(D)
        D= D[5:]
    else:
        opt_alpha, opt_beta, polymer, curr_diff_min, tocheck = fit_to_rigid_chain(polymer, nsamples, HOMO=False)    
        D,H = homopolymer(10,opt_alpha,opt_beta,0,0,1)
        D = np.sort(D)
        D= D[:5]
    
    opt_params.append(opt_alpha)
    opt_params.append(opt_beta)
    opt_params.append(curr_diff_min)
    
    output_file = output_dir + f"{polymer}.txt"
    
    np.savetxt(output_file, opt_params)




