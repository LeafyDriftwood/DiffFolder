#!/bin/sh
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o run_train_eigenfold_one_hot.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=wangdrak_dorji@brown.edu

module load gcc/12.3.1

python3 train.py --splits splits/sampledlimit256.csv --embeddings_dir /users/wdorji/data/DiffFolder/DiffFolder/one_hot/training_embeddings --no_edge_embs --lm_node_dim 20
