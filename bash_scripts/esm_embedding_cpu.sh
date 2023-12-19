#!/bin/sh
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o run_make_splits.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu



cd /users/anair27/data/DiffFolder/DiffFolder/
module load cuda
module load gcc/12.3.1
module load openssl/3.0.0
# conda init zsh
# conda activate /users/anair27/data/DiffFolder/DiffFolder/env/difffolder_conda

python3 esm_embedding.py $@ -d 'cpu'
