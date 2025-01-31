#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import interp1d
import numpy as np
import math
import copy as cp
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline
import itertools

# List of colors for the edges
edge_colors = ['navy', 'crimson', 'orange', 'green', 'purple']

# Use itertools.cycle to repeat colors if necessary
color_cycle = itertools.cycle(edge_colors)

def inbetweenlines(a,b): #to find relevant parameters from force field files
    copy = False
    inbetween=[]
    for line in lines:
        if line.strip() == a:
            copy = True
            continue
        elif line.strip() == b:
            copy = False
            continue
        elif copy:
            inbetween.append(line.split())
    return inbetween

def dihedral_calculator(point_1, point_2, point_3, point_4): #to calculate dihedral angles from xyz coordinates
    import numpy as np
    a = np.array(point_1)
    b = np.array(point_2)
    c = np.array(point_3)
    d = np.array(point_4)
    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    # normalize b1 so that it does not influence magnitude of vector
    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def sum_of_neighbors(vector, m):
    """
    Calculate the sum of m neighbor values in a vector.

    Parameters:
    vector (list): Input vector of length n.
    m (int): Number of consecutive neighbors to sum.

    Returns:
    list: A vector containing the sum of m neighbor values.
    """
    n = len(vector)
    return [sum(vector[i:i+m]) for i in range(n - m + 1)]

def generate_neighbor_sums(vector):
    """
    Generate n vectors with the sum of m neighbor values for m ranging from 1 to n.

    Parameters:
    vector (list): Input vector of length n.

    Returns:
    list: A list of vectors, each containing the sum of m neighbor values.
    """
    n = len(vector)
    return [sum_of_neighbors(vector, m) for m in range(1, n + 1)]

def concatenate_neighbor_sums(vectors):
    """
    Calculate and concatenate neighbor sums of multiple vectors.

    Parameters:
    vectors (list of lists): List of input vectors.

    Returns:
    list: Resultant list with concatenated neighbor sums.
    """
    if not vectors:
        return []

    # Initialize result with empty lists for each neighbor sum, including the last full vector
    n = len(vectors[0])
    concatenated_sums = [[] for _ in range(n)]

    for vec in vectors:
        current_sums = generate_neighbor_sums(vec)
        
        for i in range(len(current_sums)):
            concatenated_sums[i].extend(current_sums[i])

    return concatenated_sums
def normalize_angle(angle):
    """
    Normalize an angle to the range of [-180, 180] degrees.

    Parameters:
    angle (float): The angle to normalize.

    Returns:
    float: The normalized angle within the range [-180, 180].
    """
    # Normalize the angle to the range [0, 360)
    angle = angle % 360.0
    
    # Shift the angle to the range [-180, 180]
    if angle > 180:
        angle -= 360
    return angle

def normalize_angles(angle_list):
    """
    Normalize a list of angles to the range of [-180, 180] degrees.

    Parameters:
    angle_list (list of floats): The list of angles to normalize.

    Returns:
    list of floats: The list of normalized angles within the range [-180, 180].
    """
    return [normalize_angle(angle) for angle in angle_list]

def calculate_radius_of_gyration(coords):
    # Number of points
    N = coords.shape[0]
    
    # Calculate the center of mass
    center_of_mass = np.mean(coords, axis=0)
    
    # Calculate the squared distances from the center of mass
    squared_distances = np.sum((coords - center_of_mass) ** 2, axis=1)
    
    # Calculate the radius of gyration
    radius_of_gyration = np.sqrt(np.sum(squared_distances) / N)
    
    return radius_of_gyration


def splinefit(angles,n_bins=100): # for fitting spline on actual dihedral values to further use it in integration below
    # Step 1: Create a histogram from the angle data
    hist, bin_edges = np.histogram(np.radians(angles), bins=n_bins, range=(-np.pi, np.pi), density=True)

    # Step 2: Compute the bin centers (these will serve as the x-values for fitting)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Step 3: Fit a periodic spline to the histogram data
    # Ensure it's periodic by using k=3 (cubic spline), ext=3 ensures periodic behavior
    spline_fit = UnivariateSpline(bin_centers, hist, s=0, k=3, ext=3)

    # Step 4: Evaluate the spline fit on a finer grid for smoothness
    theta_fine = np.linspace(-np.pi, np.pi, 1000)  # New finer grid for evaluation
    P_spline = spline_fit(theta_fine)
    return P_spline

def splinefit_nohist(angles,potential):
    
    # Ensure it's periodic by using k=3 (cubic spline), ext=3 ensures periodic behavior
    spline_fit = UnivariateSpline(np.radians(angles), potential, s=0, k=3, ext=3)

    # Step 4: Evaluate the spline fit on a finer grid for smoothness
    theta_fine = np.linspace(-np.pi, np.pi, 1000)  # New finer grid for evaluation
    P_spline = spline_fit(theta_fine)
    return P_spline

def integrate(P1_2, P2_3, angles):
    """Calculate P1_3 by integrating P1_2(theta') * P2_3(theta - theta')."""
    n_points = len(angles)
    P1_3 = np.zeros_like(P1_2)

    # Calculate the integral for each theta in angles
    for i, theta in enumerate(angles):
        # Calculate the shifted angles for theta - theta'
        shifted_angles = theta - angles  # theta - theta'
        
        # Ensure the shifted angles are in the range [-pi, pi]
        shifted_angles = np.mod(shifted_angles + np.pi, 2 * np.pi) - np.pi
        
        # Interpolate P2_3 at the shifted angles
        P2_3_values = np.interp(shifted_angles, angles, P2_3, left=0, right=0)
        
        # Calculate the integrand and integrate over the angle range
        integrand = P1_2 * P2_3_values
        P1_3[i] = simps(integrand, angles)

    # Normalize the result so it sums to 1
    P1_3 /= simps(P1_3, angles)
    return P1_3
def calculate_ratio(P1_n, angles): #calculate the ratio of the area of -30 < theta < +30 or +150 > theta or theta < -150 degree 
    # Convert degrees to radians for defining ranges
    deg_to_rad = np.pi / 180
    angle_30 = 30 * deg_to_rad
    angle_150 = 150 * deg_to_rad

    # Ensure angles wrap properly at the boundaries of -pi and pi
    angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi

    # Find the indices corresponding to the ranges in radians
    idx_30_to_30 = np.where((angles >= -angle_30) & (angles <= angle_30))[0]
    idx_below_150 = np.where(angles <= -angle_150)[0]
    idx_above_150 = np.where(angles >= angle_150)[0]

    # Calculate the area under P1_n for each range
    area_30_to_30 = simps(P1_n[idx_30_to_30], angles[idx_30_to_30])
    area_below_150 = simps(P1_n[idx_below_150], angles[idx_below_150])
    area_above_150 = simps(P1_n[idx_above_150], angles[idx_above_150])
    
    # Total area under the entire P1_n curve
    total_area = simps(P1_n, angles)

    # Check if the total area is valid
    if total_area == 0:
        raise ValueError("Total area under the distribution is zero, normalization issue.")

    # Calculate the ratio of the areas in the specific ranges to the total area
    ratio = (area_30_to_30 + area_below_150 + area_above_150) / total_area

    return ratio

from polymers import *

degree=10 #10mer polymers
snapshots=200

moleculenames = list(POLYMER_DIC)
PLANARITY_POLYMERS={}
PLANARITY_POLYMERS_norm={}
RATIO_POLYMERS={}
RATIO_torpot_POLYMERS={}
RATIO_dimer_POLYMERS={}
END2ENDDIST_POLYMERS={}
RG_POLYMERS={}
for moleculename in moleculenames:
    print(moleculename)
    fragmentlist=cp.deepcopy(POLYMER_DIC[moleculename])
    DIHED=[]
    dihedral_between_monomers=[]
    with open(moleculename+'/'+moleculename+'_RU_'+str(degree)+'.itp', 'r') as f: #the force field (itp files) for 1-mer and 10-mer should be provided here
        lines=f.readlines()
        dihedrals=inbetweenlines('[ dihedrals ]', '')
        for dihedral in dihedrals:
            if dihedral[4]=='8':
                dihedral_between_monomers.append([int(dihedral[0]),int(dihedral[1]),int(dihedral[2]),int(dihedral[3])])
    dihedral_between_monomers.sort()
    rg=[]
    end2enddist=[]
    end2enddist_2=[]
    end2enddist_4=[]
    end2enddist_6=[]
    end2enddist_8=[]
    #to calculate end 2 end distance for shorter segments of the chain (e.g., 2, 4, 6, 8 monomers)
    with open(PATH_FRAGMENTS+POLYMER_DIC[moleculename][-1]+'.xyz', 'r') as f:
        len_lastfrag=int(f.readlines()[0].split()[0])
    
    for i in range(snapshots):
        all_dihedrals=[]
        coordinates=[]
        with open(PATH_QM+moleculename+'/input_files/'+str(i+1)+'_chain_H.xyz', 'r') as f: #the xyz files for iso or embd chains should be provided here
            lines=f.readlines()
            for each in dihedral_between_monomers:
                all_dihedrals.append([[float(lines[each[0]+1].split()[1]),float(lines[each[0]+1].split()[2]),float(lines[each[0]+1].split()[3])],
                                      [float(lines[each[1]+1].split()[1]),float(lines[each[1]+1].split()[2]),float(lines[each[1]+1].split()[3])],
                                      [float(lines[each[2]+1].split()[1]),float(lines[each[2]+1].split()[2]),float(lines[each[2]+1].split()[3])],
                                      [float(lines[each[3]+1].split()[1]),float(lines[each[3]+1].split()[2]),float(lines[each[3]+1].split()[3])]])
            end1=np.array([float(lines[-1].split()[1]),float(lines[-1].split()[2]),float(lines[-1].split()[3])])
            end2=np.array([float(lines[-2].split()[1]),float(lines[-2].split()[2]),float(lines[-2].split()[3])])
            #int((len(lines)-2)/2-len_lastfrag) is the atom number of the midpoint of the chain
            end_2=np.array([float(lines[int(2*(len(lines)-4)/10-len_lastfrag+2)].split()[1]),float(lines[int(2*(len(lines)-4)/10-len_lastfrag+2)].split()[2]),float(lines[int(2*(len(lines)-4)/10-len_lastfrag+2)].split()[3])])
            end_4=np.array([float(lines[int(4*(len(lines)-4)/10-len_lastfrag+2)].split()[1]),float(lines[int(4*(len(lines)-4)/10-len_lastfrag+2)].split()[2]),float(lines[int(4*(len(lines)-4)/10-len_lastfrag+2)].split()[3])])
            end_6=np.array([float(lines[int(6*(len(lines)-4)/10-len_lastfrag+2)].split()[1]),float(lines[int(6*(len(lines)-4)/10-len_lastfrag+2)].split()[2]),float(lines[int(6*(len(lines)-4)/10-len_lastfrag+2)].split()[3])])
            end_8=np.array([float(lines[int(8*(len(lines)-4)/10-len_lastfrag+2)].split()[1]),float(lines[int(8*(len(lines)-4)/10-len_lastfrag+2)].split()[2]),float(lines[int(8*(len(lines)-4)/10-len_lastfrag+2)].split()[3])])

            end2enddist.append(np.linalg.norm(end2-end1))
            end2enddist_2.append(np.linalg.norm(end2-end_2))
            end2enddist_4.append(np.linalg.norm(end2-end_4))
            end2enddist_6.append(np.linalg.norm(end2-end_6))
            end2enddist_8.append(np.linalg.norm(end2-end_8))

            for line in lines[2:]:
                coordinates.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
            rg.append(calculate_radius_of_gyration(np.array(coordinates)))

        all_dihedrals_values=[]
        for index in range(len(all_dihedrals)):
            all_dihedrals_values.append(dihedral_calculator(all_dihedrals[index][0],all_dihedrals[index][1],all_dihedrals[index][2],all_dihedrals[index][3]))
        DIHED.append(all_dihedrals_values)
    concatenated_sums = concatenate_neighbor_sums(DIHED)
    normalized_concatenated_sums=[]
    for i in range(len(concatenated_sums)):
        normalized_concatenated_sums.append(normalize_angles(concatenated_sums[i]))
        
    
    COUNT_planar=[]
    COUNT_not_planar=[]
    PLANARITY=[]
    for i in range(len(normalized_concatenated_sums)):
        count_planar=0
        count_not_planar=0
        for j in range(len(normalized_concatenated_sums[i])):
            if -30 < normalized_concatenated_sums[i][j] < 30 or normalized_concatenated_sums[i][j] < -150 or normalized_concatenated_sums[i][j] > 150:
                count_planar += 1
            else:
                count_not_planar += 1
        COUNT_planar.append(count_planar)
        COUNT_not_planar.append(count_not_planar)
        PLANARITY.append(count_planar/(count_planar+count_not_planar))
    plt.figure(1)
    plt.plot(np.linspace(1,len(PLANARITY),len(PLANARITY)),PLANARITY,linestyle='none',marker='D', color='olive')
    plt.xlabel('number of monomers')
    plt.ylabel('Planarity index')
    plt.savefig(moleculename+'/'+'planarity_monomer.jpeg')
    plt.close()
    dist=[]
    #calculating repeat unit length
    for i in range(degree):
        for j in range(len(fragmentlist)):
            with open(PATH_FRAGMENTS+fragmentlist[j]+'.xyz', 'r') as f:
                lines=f.readlines()
                a=np.array([float(lines[-1].split()[1]), float(lines[-1].split()[2]), float(lines[-1].split()[3])])
                b=np.array([float(lines[-2].split()[1]), float(lines[-2].split()[2]), float(lines[-2].split()[3])])
                dist.append(np.linalg.norm(a-b))
    dist_cumsum=np.cumsum(dist)
    #spline = CubicSpline(dist_cumsum[:-1], PLANARITY)

    dist_cumsum_dense = np.linspace(dist_cumsum[:-1].min(), dist_cumsum[:-1].max(), 1000)
    spline = interp1d(dist_cumsum[:-1], PLANARITY, kind='linear')  # Linear spline fit 

    PLANARITY_dense = spline(dist_cumsum_dense)

    for index, each in enumerate(PLANARITY_dense):
        flag = False  # Track if we find a value < 0.5
        for index, each in enumerate(PLANARITY_dense):
            if each < 0.5:
                PLANARITY_POLYMERS[moleculename] = dist_cumsum_dense[index]
                flag = True  # Set flag when value < 0.5 is found
                break
        
        # If no value < 0.5 was found, assign the last value
        if not flag:
            PLANARITY_POLYMERS[moleculename] = dist_cumsum_dense[-1]
    #PLANARITY normalized by length of polymer
    PLANARITY_POLYMERS_norm[moleculename] = PLANARITY_POLYMERS[moleculename]/dist_cumsum[-1]
        
    with open(moleculename+'/'+moleculename+'_planarity.txt', 'w') as f:
        for item1, item2 in zip(dist_cumsum[:-1], PLANARITY):
            f.write(f'{round(item1,3):<10} {round(item2,3):<10}\n')
    END2ENDDIST_POLYMERS[moleculename]=[round(np.average(end2enddist_2),2)]
    END2ENDDIST_POLYMERS[moleculename].append(round(np.average(end2enddist_4),2))
    END2ENDDIST_POLYMERS[moleculename].append(round(np.average(end2enddist_6),2))
    END2ENDDIST_POLYMERS[moleculename].append(round(np.average(end2enddist_8),2))
    END2ENDDIST_POLYMERS[moleculename].append(round(np.average(end2enddist),2))
    END2ENDDIST_POLYMERS[moleculename].append(round(np.average(end2enddist)/dist_cumsum[-1],2))
    END2ENDDIST_POLYMERS[moleculename].append(round(np.std(end2enddist),2))
    RG_POLYMERS[moleculename]=[round(np.average(rg),2)]
    RG_POLYMERS[moleculename].append(round(np.average(rg)/dist_cumsum[-1],2))
    plt.figure(2)
    plt.plot(dist_cumsum[:-1],PLANARITY,linestyle='none',marker='D', color='olive')
    plt.plot(dist_cumsum_dense,PLANARITY_dense,linestyle='--', color='olive')
    plt.xlabel('length / A')
    plt.ylabel('Planarity index')
    plt.savefig(moleculename+'/'+'planarity_length.jpeg')
    plt.close()           
    

    # Number of points for discretizing the angle
    n_points = 1000

    # Discretize angles between -pi and +pi
    angles = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    dtheta = angles[1] - angles[0]  # Step size

    len_frag= len(fragmentlist) # number of fragments in a repeat unit

    dihed_dict = {f'dihed{i+1}': [] for i in range(len_frag)}  # Initialize lists for dihed1, dihed2, ...

    # Loop to append columns
    for i in range(len(DIHED[0])):
        index = i % len_frag  # Determine the index for dihed lists
        dihed_dict[f'dihed{index + 1}'].append(np.array(DIHED)[:, i])  # Append column to the appropriate list

    # Convert lists to numpy arrays if needed
    for key in dihed_dict:
        dihed_dict[key] = np.array([x for xs in dihed_dict[key] for x in xs])
        plt.hist(dihed_dict[key],label='P'+key[-1]+'_'+str(int(key[-1])+1),density=True, bins=50,facecolor='none',edgecolor=next(color_cycle))
    plt.xlabel('Torsion angle / °')
    plt.legend(frameon=True)
    plt.ylabel('Probability density / $°{^-1}$')
    plt.xlim([-180,180])
    plt.xticks([-180,-120,-60,0,60,120,180])
    plt.grid(True)
    plt.savefig(moleculename+'/'+'hist_Pn_n+1.jpeg')
    plt.close() 

                                   
    fits_dict={f'dihed{i+1}': [] for i in range(len_frag)} #this is for simulation results
    for key in fits_dict:
        fits_dict[key] = splinefit(dihed_dict[key],n_bins=50)
        fits_dict[key] /= simps(fits_dict[key], angles)
    
    fits_dict_dimer={f'dihed{i+1}': [] for i in range(len_frag)} #this is from torsional potential
    for index, key in enumerate(fits_dict_dimer):
        angles_dimer=[]
        potentials_dimer=[]
        with open(PATH_MD+moleculename+'/dimer/dihed_'+str(index+1)+'.xvg', 'r') as f:
            lines=f.readlines()
            for line in lines:
                if line.startswith('@') or line.startswith('#'):
                    continue  # Skip lines starting with '@' or '#'
                angles_dimer.append(float(line.split()[0]))
                potentials_dimer.append(float(line.split()[1]))
        fits_dict_dimer[key] = splinefit_nohist(angles_dimer,potentials_dimer)
        fits_dict_dimer[key] /= simps(fits_dict_dimer[key], angles)
        plt.plot(angles,fits_dict_dimer[key],label='P'+key[-1]+'_'+str(int(key[-1])+1))
    plt.xlabel('Torsion angle / rad')
    plt.legend(frameon=True)
    plt.ylabel('Probability density / $rad{^-1}$')
    plt.grid(True)
    plt.savefig(moleculename+'/'+'hist_Pn_n+1_dimer.jpeg')
    plt.close() 

    fits_dict_torpot={f'dihed{i+1}': [] for i in range(len_frag)} #this is from torsional potential
    T = 300  # e.g., room temperature
    k_B=8.314e-3

    for index, key in enumerate(fits_dict_torpot):
        angles_dft=[]
        potentials_dft=[]
        with open(PATH_OUTPUT+moleculename+'/Torsion/table_d'+str(index+2)+'.xvg', 'r') as f:
            lines=f.readlines()
            for line in lines:
                angles_dft.append(float(line.split()[0]))
                potentials_dft.append(float(line.split()[1]))
        # Calculate Boltzmann distribution
        boltzmann = np.exp(-np.array(potentials_dft) / (k_B * T))
        fits_dict_torpot[key] = splinefit_nohist(angles_dft,boltzmann)
        fits_dict_torpot[key] /= simps(fits_dict_torpot[key], angles)
        plt.plot(angles,fits_dict_torpot[key],label='P'+key[-1]+'_'+str(int(key[-1])+1))
    plt.xlabel('Torsion angle / rad')
    plt.legend(frameon=True)
    plt.ylabel('Probability density / $rad{^-1}$')
    plt.grid(True)
    plt.savefig(moleculename+'/'+'hist_Pn_n+1_torpot.jpeg')
    plt.close() 

    # doing the integrals
    integrals=[fits_dict['dihed1']] #first for simulation results
    RATIO=[]
    for i in range(1,len(DIHED[0])):
        P1_n=integrate(integrals[i-1],fits_dict[f'dihed{i%len_frag+1}'],angles)
        P1_n/= simps(P1_n, angles)
        integrals.append(P1_n)
        RATIO.append(calculate_ratio(integrals[i-1], angles)) # by ratio I mean the ratio of -30<angle<+30 or angle > +150 or angle < -150 
        plt.plot(np.degrees(angles),P1_n,label='P_1-'+str(i+2))
    plt.xlabel('Torsion angle / °')
    plt.legend(frameon=True)
    plt.ylabel('Probability density / $°{^-1}$')
    plt.xlim([-180,180])
    plt.xticks([-180,-120,-60,0,60,120,180])
    plt.grid(True)
    plt.savefig(moleculename+'/'+'p1n.jpeg')
    plt.close() 

    RATIO.append(calculate_ratio(integrals[-1], angles))
    spline = interp1d(dist_cumsum[:-1], RATIO, kind='linear')  # Linear spline fit 
    RATIO_dense = spline(dist_cumsum_dense)
    for index, each in enumerate(RATIO_dense):
        flag = False  # Track if we find a value < 0.5
        for index, each in enumerate(RATIO_dense):
            if each < 0.5:
                RATIO_POLYMERS[moleculename] = dist_cumsum_dense[index]
                flag = True  # Set flag when value < 0.5 is found
                break
        
        # If no value < 0.5 was found, assign the last value
        if not flag:
            RATIO_POLYMERS[moleculename] = dist_cumsum_dense[-1]
    plt.plot(dist_cumsum[:-1],RATIO,linestyle='none',marker='D', color='olive')
    plt.plot(dist_cumsum_dense,RATIO_dense,linestyle='--', color='olive')
    plt.xlabel('length / A')
    plt.ylabel('Planarity index from P1_n')
    plt.savefig(moleculename+'/'+'planarity_length_p1n.jpeg')
    plt.close()      
    
    integrals_dimer=[fits_dict_dimer['dihed1']] #now for dimer simulation 
    RATIO_dimer=[]
    for i in range(1,len(DIHED[0])):
        P1_n=integrate(integrals_dimer[i-1],fits_dict_dimer[f'dihed{i%len_frag+1}'],angles)
        P1_n/= simps(P1_n, angles)
        integrals_dimer.append(P1_n)
        RATIO_dimer.append(calculate_ratio(integrals_dimer[i-1], angles))
        plt.plot(np.degrees(angles),P1_n,label='P_1-'+str(i+2))
    plt.xlabel('Torsion angle / °')
    plt.legend(frameon=True)
    plt.ylabel('Probability density / $°{^-1}$')
    plt.xlim([-180,180])
    plt.xticks([-180,-120,-60,0,60,120,180])
    plt.savefig(moleculename+'/'+'p1n_dimer.jpeg')
    plt.close() 
    RATIO_dimer.append(calculate_ratio(integrals_dimer[-1], angles))
    spline = interp1d(dist_cumsum[:-1], RATIO_dimer, kind='linear')  # Linear spline fit 
    RATIO_dimer_dense = spline(dist_cumsum_dense)
    for index, each in enumerate(RATIO_dimer_dense):
        flag = False  # Track if we find a value < 0.5
        for index, each in enumerate(RATIO_dimer_dense):
            if each < 0.5:
                RATIO_dimer_POLYMERS[moleculename] = dist_cumsum_dense[index]
                flag = True  # Set flag when value < 0.5 is found
                break
        
        # If no value < 0.5 was found, assign the last value
        if not flag:
            RATIO_dimer_POLYMERS[moleculename] = dist_cumsum_dense[-1]
    plt.plot(dist_cumsum[:-1],RATIO_dimer,linestyle='none',marker='D', color='olive')
    plt.plot(dist_cumsum_dense,RATIO_dimer_dense,linestyle='--', color='olive')
    plt.xlabel('length / A')
    plt.ylabel('Planarity index from P1_n')
    plt.savefig(moleculename+'/'+'planarity_length_p1n_dimer.jpeg')
    plt.close()  

    integrals_torpot=[fits_dict_torpot['dihed1']] #now for torsional potential 
    RATIO_torpot=[]
    for i in range(1,len(DIHED[0])):
        P1_n=integrate(integrals_torpot[i-1],fits_dict_torpot[f'dihed{i%len_frag+1}'],angles)
        P1_n/= simps(P1_n, angles)
        integrals_torpot.append(P1_n)
        RATIO_torpot.append(calculate_ratio(integrals_torpot[i-1], angles))
        plt.plot(np.degrees(angles),P1_n,label='P_1-'+str(i+2))
    plt.xlabel('Torsion angle / °')
    plt.legend(frameon=True)
    plt.ylabel('Probability density / $°{^-1}$')
    plt.xlim([-180,180])
    plt.xticks([-180,-120,-60,0,60,120,180])
    plt.savefig(moleculename+'/'+'p1n_torpot.jpeg')
    plt.close() 
    RATIO_torpot.append(calculate_ratio(integrals_torpot[-1], angles))
    spline = interp1d(dist_cumsum[:-1], RATIO_torpot, kind='linear')  # Linear spline fit 
    RATIO_torpot_dense = spline(dist_cumsum_dense)
    for index, each in enumerate(RATIO_torpot_dense):
        flag = False  # Track if we find a value < 0.5
        for index, each in enumerate(RATIO_torpot_dense):
            if each < 0.5:
                RATIO_torpot_POLYMERS[moleculename] = dist_cumsum_dense[index]
                flag = True  # Set flag when value < 0.5 is found
                break
        
        # If no value < 0.5 was found, assign the last value
        if not flag:
            RATIO_torpot_POLYMERS[moleculename] = dist_cumsum_dense[-1]
    plt.plot(dist_cumsum[:-1],RATIO_torpot,linestyle='none',marker='D', color='olive')
    plt.plot(dist_cumsum_dense,RATIO_torpot_dense,linestyle='--', color='olive')
    plt.xlabel('length / A')
    plt.ylabel('Planarity index from P1_n')
    plt.savefig(moleculename+'/'+'planarity_length_p1n_torpot.jpeg')
    plt.close()  
  
with open('PLANARITY.txt', 'w') as f: #gives the Lpp
    for key, value in PLANARITY_POLYMERS.items():
        f.write(f'{key} {value}\n')
with open('PLANARITY_norm.txt', 'w') as f: #gives the normalized Lpp by the contour legth of the SCPs
    for key, value in PLANARITY_POLYMERS_norm.items():
        f.write(f'{key} {value}\n')
with open('PLANARITY_P1_ns.txt', 'w') as f: #gives the Lpp based on the dihedral angles of the repeat units from 200 iso / embd models
    for key, value in RATIO_POLYMERS.items():
        f.write(f'{key} {value}\n')
with open('PLANARITY_P1_dimer.txt', 'w') as f: #gives the Lpp based on the dihedral angles of the repeat units from small MD simulation of dimers
    for key, value in RATIO_dimer_POLYMERS.items():
        f.write(f'{key} {value}\n')
with open('PLANARITY_P1_ns_torpot.txt', 'w') as f: #gives the Lpp based on the torsional angles from Boltzmann distribution from torsional potential
    for key, value in RATIO_torpot_POLYMERS.items():
        f.write(f'{key} {value}\n')
with open('END2END.txt', 'w') as f: #give end to end distance of the SCPs
    f.write('# end_to_end_2   end_to_end_4   end_to_end_6   end_to_end_8   end_to_end   end_to_end_norm   end_to_end_std\n')
    for polymer in END2ENDDIST_POLYMERS.keys():
        f.write(f'{polymer}  ' + ' '.join(f'{v:.2f}' for v in END2ENDDIST_POLYMERS[polymer]) + '\n')
with open('RG.txt', 'w') as f: #give radius of gyration of the SCPs
    for polymer in RG_POLYMERS.keys():
        f.write(f'{polymer}  ' + ' '.join(f'{v:.2f}' for v in RG_POLYMERS[polymer]) + '\n')







