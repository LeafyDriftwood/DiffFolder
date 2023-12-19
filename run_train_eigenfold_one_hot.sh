#!/bin/sh
#SBATCH -N 1
    #node
#SBATCH -p rsingh47-gcondo 
#SBATCH -n 4 #cores
#SBATCH--gres=gpu:4 --gres-flags=enforce-binding
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o out/run_train_eigenfold_one_hot_new.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=wangdrak_dorji@brown.edu

module load gcc/12.3.1

python3 train.py --splits splits/sampledlimit256.csv --embeddings_dir /users/wdorji/data/DiffFolder/DiffFolder/embeddings/one_hot/training_embeddings --no_edge_embs --lm_node_dim 20