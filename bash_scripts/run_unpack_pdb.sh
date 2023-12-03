#!/bin/sh
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o run_unpack_pdb.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=wangdrak_dorji@brown.edu

python unpack_pdb.py --num_workers 40