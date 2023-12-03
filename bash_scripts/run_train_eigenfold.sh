#!/bin/sh
#SBATCH -N 1
    #node
#SBATCH -p rsingh47-gcondo 
#SBATCH--gres=gpu:4 --gres-flags=enforce-binding
#SBATCH -n 4 #cores
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o run_train_eigenfold.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=manav_chakravarthy@brown.edu

python3 train.py --splits splits/sampledlimit256.csv --embeddings_dir embeddings/all256_embeddings --no_edge_embs
