#!/bin/sh
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o rename_esm_embedding_full.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu

python rename_esm_embeddings.py