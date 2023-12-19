#!/bin/sh
#SBATCH --mem=64G
#SBATCH -t 48:00:00
# SBATCH -p gpu --gres=gpu:1
#SBATCH -o out/run_make_one_hot_embeddings_cameo2022.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=wangdrak_dorji@brown.edu

module load cuda
module load gcc/12.3.1
module load openssl/3.0.0

# python make_prottrans.py --out_dir ./prot/apo_embeddings --splits splits/apo.csv
python make_one_hot.py --out_dir ./embeddings/one_hot/cameo2022_embeddings --splits splits/cameo2022.csv