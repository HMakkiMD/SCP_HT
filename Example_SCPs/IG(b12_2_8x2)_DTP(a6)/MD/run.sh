#!/bin/bash -l

#SBATCH -p node_name
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time 5-00:00:00


module load apps/gromacs_cuda/2022.0

gmx grompp -f minim.mdp -c IG(b12_2_8x2)_DTP(a6)_10_1.gro -p IG(b12_2_8x2)_DTP(a6)_10_1.top -o min.tpr

gmx mdrun -deffnm min -nt 24 -tableb table_corr_d1.xvg table_corr_d2.xvg table_corr_d3.xvg

gmx grompp -f npt.mdp -c min.gro -p IG(b12_2_8x2)_DTP(a6)_10_1.top -o npt.tpr

gmx mdrun -deffnm npt  -nt 12 -tableb table_corr_d1.xvg table_corr_d2.xvg table_corr_d3.xvg

gmx grompp -f eq_soup.mdp -c npt.gro -p IG(b12_2_8x2)_DTP(a6)_10_1.top -o 500/500.tpr

gmx mdrun -deffnm 500/500  -nt 24 -tableb table_corr_d1.xvg table_corr_d2.xvg table_corr_d3.xvg

