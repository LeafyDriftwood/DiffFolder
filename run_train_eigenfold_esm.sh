#!/bin/sh
#SBATCH -N 1
#SBATCH -p rsingh47-gcondo 
#SBATCH -n 4 #cores
#SBATCH--gres=gpu:3 --gres-flags=enforce-binding
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o run_train_eigenfold_esm.out
# SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
# SBATCH --mail-user=akira_nair@brown.edu

# to assess availability of gcondo gpus:
# squeue -p rsingh47-gcondo
echo "starting job"
# make sure to activate `conda activate difffolder` before running this script
cd ~/data/DiffFolder/DiffFolder/
module load gcc/12.3.1

python3 train.py --splits splits/sampledlimit256.csv --embeddings_dir /users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_256_all_new --no_edge_embs --lm_node_dim 1280