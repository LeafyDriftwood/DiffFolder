#!/bin/sh
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o ../out/run_make_prottrans_embeddings.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=wangdrak_dorji@brown.edu

python ../make_prottrans.py --out_dir ../prottrans/all256_embeddings --splits ../splits/limit256.csv --reference_only --num_workers 6 