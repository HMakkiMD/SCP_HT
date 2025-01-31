import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy as cp

# Define the exponential decay function for fitting
def exponential_decay(L, P):
    return np.exp(-L / P)

# Function to calculate the persistence length
def calculate_persistence_length(tangents, lengths, num_chains, moleculename):
    n = len(tangents[0])  # Number of monomers (assuming all chains have same length)
    contour_distances = np.cumsum(lengths[0])  # Compute cumulative contour distances for the first chain
    max_L = contour_distances[-1]  # Maximum contour distance

    # Initialize lists to store results for all chains
    all_averaged_cos_theta = []
    all_unique_L = []

    for chain_idx in range(num_chains):  # Loop over all chains
        cos_theta_means = [[] for _ in range(n - 1)]  # n-1 possible distances

        # Compute ⟨cos(θ)⟩ for each L for the current chain
        for i in range(n - 1):  # Iterate over starting monomers
            for j in range(i + 1, n):  # Iterate over subsequent monomers
                L = j - i  # Contour distance between monomers (j-i)
                cos_theta = np.dot(tangents[chain_idx][i], tangents[chain_idx][j])  # Dot product for cos(θ)
                cos_theta_means[L - 1].append(cos_theta)  # Append cos(θ) to the corresponding L

        # Average cos(θ) values for each contour distance
        averaged_cos_theta = [np.mean(cos_theta_means[i]) for i in range(n - 1)]

        # Use actual contour distances for unique_L (i.e., distances based on lengths)
        unique_L = contour_distances[1:]  # Exclude the 0th value

        # Store the results for this chain
        all_averaged_cos_theta.append(averaged_cos_theta)
        all_unique_L.append(unique_L)

    # Compute the mean of averaged_cos_theta and unique_L across all chains
    averaged_cos_theta_all_chains = np.mean(all_averaged_cos_theta, axis=0)
    unique_L_all_chains = np.mean(all_unique_L, axis=0)

    # Fit ⟨cos(θ)⟩ vs. L to the exponential decay model using averaged data
    popt, pcov = curve_fit(exponential_decay, unique_L_all_chains, averaged_cos_theta_all_chains, p0=[10])
    P = popt[0]  # Extract the persistence length
    P_error = np.sqrt(np.diag(pcov))[0]  # Error on P (standard deviation)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(unique_L_all_chains, averaged_cos_theta_all_chains, label="Data (⟨cos(θ)⟩)", color="blue")
    plt.plot(unique_L_all_chains, exponential_decay(unique_L_all_chains, *popt), label=f"Fit (P = {P:.2f})", color="red")
    plt.xlabel("Contour Distance L (Angstrom)")
    plt.ylabel("⟨cos(θ)⟩")
    plt.legend()
    plt.grid()
    plt.savefig(moleculename+'/'+'persistence.eps')

    return P,P_error



from polymers import *

degree = 10  # 10mer polymers
snapshots = 200

moleculenames = list(POLYMER_DIC)
PERSISTENCE_POLYMERS = {}

for moleculename in moleculenames:
    print(moleculename)
    fragmentlist = cp.deepcopy(POLYMER_DIC[moleculename])
    fragmentlist_polymer = fragmentlist * degree
    len_frags = []
    
    # Read the lengths of the fragments
    for frag in fragmentlist_polymer:
        with open(PATH_FRAGMENTS + frag + '.xyz', 'r') as f:
            len_frags.append(int(f.readlines()[0].split()[0]) - 2)
    
    len_frags_cum = np.cumsum(len_frags) - 1
    len_frags_cum_mod = [0]
    for i, value in enumerate(len_frags_cum):
        len_frags_cum_mod.append(value)      # Add I_n
        len_frags_cum_mod.append(value + 1)  # Add I_n + 1
    del len_frags_cum_mod[-1]
    
    
    all_tangents=[]
    all_lengths=[]
    # Loop over each chain snapshot
    for chain in range(snapshots):
        tangents = []
        lengths = []
        # Read the XYZ file for the chain
        with open(PATH_QM + moleculename + '/input_files/' + str(chain + 1) + '_chain_H.xyz', 'r') as f:
            lines = f.readlines()
            for j in range(len(len_frags_cum_mod)):
                if j % 2 == 0:
                    begin = [float(lines[len_frags_cum_mod[j + 1] + 2].split()[1]), 
                             float(lines[len_frags_cum_mod[j + 1] + 2].split()[2]), 
                             float(lines[len_frags_cum_mod[j + 1] + 2].split()[3])]
                    end = [float(lines[len_frags_cum_mod[j] + 2].split()[1]), 
                           float(lines[len_frags_cum_mod[j] + 2].split()[2]), 
                           float(lines[len_frags_cum_mod[j] + 2].split()[3])]
                    tangents.append(np.array(begin) - np.array(end))
                    lengths.append(np.linalg.norm(np.array(begin) - np.array(end)))
        
        all_tangents.append(np.array(tangents) / np.linalg.norm(np.array(tangents), axis=1, keepdims=True))
        all_lengths.append(np.array(lengths))
    persistence_length, error = calculate_persistence_length(all_tangents, all_lengths, snapshots, moleculename)
    PERSISTENCE_POLYMERS[moleculename]=[persistence_length]
    PERSISTENCE_POLYMERS[moleculename].append(error)


with open('PERSISTENCE.txt', 'w') as f:
    f.write('# average   std\n')
    for polymer in PERSISTENCE_POLYMERS.keys():
        f.write(f'{polymer}  ' + ' '.join(f'{v:.2f}' for v in PERSISTENCE_POLYMERS[polymer]) + '\n')
