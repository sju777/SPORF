#!/bin/bash
#SBATCH --job-name=OpenMLJob
#SBATCH --time=24:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=24
#SBATCH --mail-type=end
#SBATCH --mail-user=sju7@jhu.edu
#### load and unload modules you may need
# module load openml
# module sklearn
# module numpy
# module pandas
# module math
# module warnings
# module rerf
# module datetime
#### execute python code
python SPORF_OpenML_opti-hyper_CC-18.py
echo "Finished with job $SLURM_JOBID"
