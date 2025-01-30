import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import sys


from polymers import * 


def integrate_between_indices(x_vals, y_vals, idx_end, HOMO=True):
    
    

    integrals = []
    
    if HOMO == True:
        for i in range(0,3000): # Setting an arbitrary upper limit of 300 (it is irrelevant as the loop breaks once a certain value is reached)
            integral = np.trapz(y_vals[idx_end-i:idx_end+1], x_vals[idx_end-i:idx_end+1]) # Check integral at every value over range
            integrals.append(integral) # Append to list
            if abs(0.6- integral) < 0.01: # Break loop if integral reaches some threshold (we only need up to 0.5)
                break
    
        #plt.plot(x_vals[idx_end-i:idx_end+1], y_vals[idx_end-i:idx_end+1]) # Plots the full range that is integrated over (valence band direction)
        
    else:
        for i in range(0,3000):
            integral = np.trapz(y_vals[idx_end:idx_end+i], x_vals[idx_end:idx_end+i])
            integrals.append(integral)
            if abs(0.6- integral) < 0.01:
                break
    
        
    band_001 = (find_closest_index(integrals, 0.001))*0.001 # Finds the index of the element of the list of integrals with the closest match to an input value
    band_01 = (find_closest_index(integrals, 0.01))*0.001 # 0.01 (DOS spacing) to make it eV
    band_02 = (find_closest_index(integrals, 0.02))*0.001 # 0.01 (DOS spacing) to make it eV 
    band_05 = (find_closest_index(integrals, 0.05))*0.001 # 0.01 (DOS spacing) to make it eV 
    band_20 = (find_closest_index(integrals, 0.20))*0.001 
    band_50 = (find_closest_index(integrals, 0.50))*0.001 
    
    

    return band_001, band_01, band_02,band_05, band_20, band_50

def find_closest_index(lst, target):
    """
    Finds the index of the element in the list `lst` that is closest to the `target` value.
    
    Parameters:
    lst (list of floats): The list of floats to search through.
    target (float): The target float value to find the closest element to.
    
    Returns:
    int: The index of the closest element in the list.
    """
    closest_index = None
    closest_distance = float('inf')
    
    for i, value in enumerate(lst):
        distance = abs(value - target)
        if distance < closest_distance:
            closest_distance = distance
            closest_index = i
            
    return closest_index

tocalc = list(POLYMER_DIC.keys())

valence_iso={}
conduction_iso={}
valence_pc={}
conduction_pc={}
LL_homo_iso={}
LL_lumo_iso={}
LL_homo_pc={}
LL_lumo_pc={}
for polymer in tocalc:
    print(polymer)
    HOMOs_LUMOs = np.loadtxt("/PATH_to_Polymer_HOMO_LUMO"+polymer+"/HOMO_LUMO_"+polymer+".txt") # extract HOMO and LUMO values
    HOMO = HOMOs_LUMOs[0]
    LUMO = HOMOs_LUMOs[1]
    BG_midpoint = 0.5*(HOMO+LUMO) # Define band gap midpoint as starting point
    DOS_iso = np.loadtxt("/PATH_to_DOS"+polymer+"/"+polymer+"_DOS_0025_iso.txt") # Load DOS
    LL_iso = np.loadtxt("/PATH_to_DOS"+polymer+"/"+polymer+"_LL_0025_iso.txt") # Load LL
    
    

    # Create finer energy grid using linear spline interpolation
    finer_energy_spacing = np.arange(DOS_iso[:,0][0], DOS_iso[:,0][-1], 0.001)  # Finer grid with 0.001 eV spacing
    spline = interp1d(DOS_iso[:,0], DOS_iso[:,1], kind='linear')  # Linear spline fit for DOS(E)
    DOS_iso_fine = spline(finer_energy_spacing)  # Interpolated DOS on finer grid
    
    midpoint_index = find_closest_index(finer_energy_spacing, BG_midpoint) # Find the index of the DOS which corresponds to the bandgap midpoint


    band_001_val_iso, band_01_val_iso, band_02_val_iso, band_05_val_iso, band_20_val_iso, band_50_val_iso = (integrate_between_indices(finer_energy_spacing, DOS_iso_fine, midpoint_index,HOMO=True)) # Repeat for HOMO and LUMO 
    band_001_cond_iso, band_01_cond_iso, band_02_cond_iso, band_05_cond_iso, band_20_cond_iso, band_50_cond_iso = (integrate_between_indices(finer_energy_spacing, DOS_iso_fine, midpoint_index,HOMO=False))
    valence_attributes_iso = [band_001_val_iso, band_01_val_iso, band_02_val_iso, band_05_val_iso, band_20_val_iso, band_50_val_iso]
    cond_attributes_iso = [band_001_cond_iso, band_01_cond_iso, band_02_cond_iso, band_05_cond_iso , band_20_cond_iso, band_50_cond_iso]
    valence_iso[polymer]=[-valence_attributes_iso[0]+BG_midpoint,-valence_attributes_iso[1]+BG_midpoint,-valence_attributes_iso[2]+BG_midpoint,
     -valence_attributes_iso[3]+BG_midpoint,-valence_attributes_iso[4]+BG_midpoint,-valence_attributes_iso[5]+BG_midpoint]
    conduction_iso[polymer]=[cond_attributes_iso[0]+BG_midpoint,cond_attributes_iso[1]+BG_midpoint,cond_attributes_iso[2]+BG_midpoint,
    cond_attributes_iso[3]+BG_midpoint,cond_attributes_iso[4]+BG_midpoint,cond_attributes_iso[5]+BG_midpoint]

    # Create finer energy grid using linear spline interpolation
    spline = interp1d(LL_iso[:,0], LL_iso[:,1], kind='linear')  # Linear spline fit for LL
    LL_iso_fine = spline(finer_energy_spacing)  
    LL_homo_iso[polymer] = [LL_iso_fine[find_closest_index(finer_energy_spacing, valence_iso[polymer][0])]] #read LL at homo and lumo point
    LL_lumo_iso[polymer] = [LL_iso_fine[find_closest_index(finer_energy_spacing, conduction_iso[polymer][0])]]

    DOS_pc = np.loadtxt("/PATH_to_LL/"+polymer+"/"+polymer+"_DOS_0025_pc.txt") # Load DOS
    LL_pc = np.loadtxt("/PATH_to_LL/"+polymer+"/"+polymer+"_LL_0025_pc.txt") # Load LL
    
    


    spline = interp1d(DOS_pc[:,0], DOS_pc[:,1], kind='linear')  # Linear spline fit for DOS(E)
    DOS_pc_fine = spline(finer_energy_spacing)  # Interpolated DOS on finer grid

    midpoint_index = find_closest_index(finer_energy_spacing, BG_midpoint) # Find the index of the DOS which corresponds to the bandgap midpoint

    band_001_val_pc, band_01_val_pc, band_02_val_pc, band_05_val_pc, band_20_val_pc, band_50_val_pc = (integrate_between_indices(finer_energy_spacing, DOS_pc_fine, midpoint_index,HOMO=True)) # Repeat for HOMO and LUMO 
    band_001_cond_pc, band_01_cond_pc, band_02_cond_pc, band_05_cond_pc,band_20_cond_pc, band_50_cond_pc = (integrate_between_indices(finer_energy_spacing, DOS_pc_fine, midpoint_index,HOMO=False))
    valence_attributes_pc = [band_001_val_pc, band_01_val_pc,band_02_val_pc, band_05_val_pc, band_20_val_pc, band_50_val_pc]
    cond_attributes_pc = [band_001_cond_pc, band_01_cond_pc, band_02_cond_pc, band_05_cond_pc, band_20_cond_pc, band_50_cond_pc]
    valence_pc[polymer]=[-valence_attributes_pc[0]+BG_midpoint,-valence_attributes_pc[1]+BG_midpoint,-valence_attributes_pc[2]+BG_midpoint,
    -valence_attributes_pc[3]+BG_midpoint,-valence_attributes_pc[4]+BG_midpoint,-valence_attributes_pc[5]+BG_midpoint]
    conduction_pc[polymer]=[cond_attributes_pc[0]+BG_midpoint,cond_attributes_pc[1]+BG_midpoint,cond_attributes_pc[2]+BG_midpoint,
    cond_attributes_pc[3]+BG_midpoint,cond_attributes_pc[4]+BG_midpoint,cond_attributes_pc[5]+BG_midpoint]

    # Create finer energy grid using linear spline interpolation
    spline = interp1d(LL_pc[:,0], LL_pc[:,1], kind='linear')  # Linear spline fit for LL
    LL_pc_fine = spline(finer_energy_spacing)  # Interpolated DOS on finer grid

    LL_homo_pc[polymer] = [LL_pc_fine[find_closest_index(finer_energy_spacing, valence_pc[polymer][0])]] #read LL at homo and lumo point
    LL_lumo_pc[polymer] = [LL_pc_fine[find_closest_index(finer_energy_spacing, conduction_pc[polymer][0])]]
    
with open('band_info.txt', 'w') as file:
    file.write('#values at 0.1%, 1%, 2%, 5%, 20%, and 50% of band are reported here\n\n')
    for polymer in valence_iso.keys():
        # Ensure the polymer exists in all dictionaries
        if polymer in conduction_iso and polymer in valence_pc and polymer in conduction_pc:
            file.write(f"{polymer}_valence_iso " + " ".join(f"{v:.5f}" for v in valence_iso[polymer]) + "\n")
            file.write(f"{polymer}_conduction_iso " + " ".join(f"{v:.5f}" for v in conduction_iso[polymer]) + "\n")
            file.write(f"{polymer}_valence_pc " + " ".join(f"{v:.5f}" for v in valence_pc[polymer]) + "\n")
            file.write(f"{polymer}_conduction_pc " + " ".join(f"{v:.5f}" for v in conduction_pc[polymer]) + "\n")

with open('ll_info.txt', 'w') as file:
    file.write('#localisation length at homo and lumo for iso and pc models\n\n')
    for polymer in LL_homo_iso.keys():
        # Ensure the polymer exists in all dictionaries
        if polymer in LL_lumo_iso and polymer in LL_homo_pc and polymer in LL_lumo_pc:
            file.write(f"{polymer}_homo_iso " + " ".join(f"{v:.5f}" for v in LL_homo_iso[polymer]) + "\n")
            file.write(f"{polymer}_lumo_iso " + " ".join(f"{v:.5f}" for v in LL_lumo_iso[polymer]) + "\n")
            file.write(f"{polymer}_homo_pc " + " ".join(f"{v:.5f}" for v in LL_homo_pc[polymer]) + "\n")
            file.write(f"{polymer}_lumo_pc " + " ".join(f"{v:.5f}" for v in LL_lumo_pc[polymer]) + "\n")
            
