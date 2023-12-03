#!/bin/sh
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o downloading_pdb.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu


bash download_pdb.sh ./data
