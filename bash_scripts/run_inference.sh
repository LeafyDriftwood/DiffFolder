#!/bin/sh
#SBATCH -p gpu --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o run_inference.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=manav_chakravarthy@brown.edu

python3 inference.py --model_dir ./pretrained_model --ckpt epoch_7.pt --pdb_dir ./structures --embeddings_dir ./embeddings/omega_cameo_embeddings --embeddings_key name --elbo --num_samples 5 --alpha 1 --beta 3 --elbo_step 0.2 --splits splits/cameo2022_orig.csv 
