#!/usr/bin/env python
# coding: utf-8

# electrostatic potential exerted by repeat unit on it's shell
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math
import os

from polymers import *

def inbetweenlines(a,b):
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

def calculate_dipole_moment(charges, positions):
    """
    Calculate the dipole moment for a single configuration.
    
    Parameters:
    - charges: 1D array of atomic charges (constant for all configurations).
    - positions: 2D array of atomic positions (shape: num_atoms x 3).
    
    Returns:
    - dipole_moment: 1D array representing the dipole moment vector (e·Å).
    """
    charges = np.array(charges)
    positions = np.array(positions)
    dipole_moment = np.sum(charges[:, np.newaxis] * positions, axis=0)
    dipole_momoent_magnitude = np.linalg.norm(dipole_moment)
    return dipole_moment, dipole_momoent_magnitude

def dipole_average_over_RUs(charges, position_samples):
    # Calculate dipole moments for all configurations
    dipole_moments = np.array([calculate_dipole_moment(charges, positions)[0] for positions in position_samples])
    dipole_momoent_magnitude = np.array([calculate_dipole_moment(charges, positions)[1] for positions in position_samples])
    # Compute the mean dipole moment vector
    mean_dipole = np.mean(dipole_moments, axis=0)

    mean_dipole_momoent_magnitude = np.mean(dipole_momoent_magnitude, axis=0)
    
    # Compute the RMSD of the dipole moments

    #rmsd = np.sqrt(np.mean(np.sum((dipole_moments - mean_dipole)**2, axis=1)))
    
    return mean_dipole, mean_dipole_momoent_magnitude


moleculenames = list(POLYMER_DIC)


DIPOLES_POLYMER={}
DIPOLES_RU={}

for moleculename in moleculenames:

    charges=[]
    with open(moleculename+'/'+moleculename+'_RU_1.itp', 'r') as f: #force field for 1-mer should be provided
        lines=f.readlines()
        atoms=inbetweenlines('[ atoms ]', '[ bonds ]')
        for line in atoms:
            charges.append(float(line[6]))
    charges=np.array(charges)
    coords_RU=[]
    with open(moleculename+'/'+moleculename+'_RU_1.xyz', 'r') as f: #xyz for optimized 1-mer should be provided
            lines=f.readlines()[2:]
            for line in lines:
                coords_RU.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    coords_RU=np.array(coords_RU)
    dipole_moment_RU, dipole_momoent_magnitude = calculate_dipole_moment(charges, coords_RU)
    DIPOLES_RU[moleculename]=dipole_momoent_magnitude

with open('EP_dipole_RU.txt', 'w') as f:
    for key, value in DIPOLES_RU.items():
        f.write(f'{key} {value}\n')




