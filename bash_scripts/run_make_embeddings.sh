#!/bin/sh
#SBATCH -N 1
    #node
#SBATCH -p rsingh47-gcondo 
#SBATCH--gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -n 4 #cores
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o run_make_embeddings_8.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=manav_chakravarthy@brown.edu

python make_embeddings.py --out_dir ./embeddings/all256_embeddings --splits splits/limit256.csv --num_workers 8 --worker_id 8