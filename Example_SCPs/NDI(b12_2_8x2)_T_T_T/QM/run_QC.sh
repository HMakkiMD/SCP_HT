#!/bin/bash -l

#SBATCH -N  1
#SBATCH -n  40
#SBATCH -t 5-00:00:00
#SBATCH -J IGDPT
#SBATCH -p node_name

module load apps/gaussian/16
module load apps/anaconda3/2022.10
source activate myenv #cc-lib and standard python libraries should be available for this code

python3 QC_calculation_v1-3.py input_variables.inp > output_QC.txt
