#!/bin/sh
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o run_make_splits.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=wangdrak_dorji@brown.edu

python make_splits.py