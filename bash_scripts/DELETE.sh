#!/bin/sh
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o delete_all.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu

# WARNING: DO NOT RUN THIS SCRIPT UNLESS YOU ARE SURE YOU WANT TO DELETE ALL FILES! NOT REVERSIBLE

cd ~/data/DiffFolder/

# remove DiffFolder folder and all its contents
rm -rf DiffFolder

echo "Deleted DiffFolder project"