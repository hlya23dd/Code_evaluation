#!/bin/bash
##SBATCH -J neuroesp
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=end
##SBATCH --mail-user=hongliu.yang@tu-dresden.de
#SBATCH --time=24:00:00



module purge --force
module load modenv/scs5

module load Keras
module load matplotlib/2.1.2-foss-2018a-Python-3.6.4


python mvar_gen.py

