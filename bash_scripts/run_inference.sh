#!/bin/sh
#SBATCH -n 2
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 36:00:00
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o run_inference_cameo_2.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=manav_chakravarthy@brown.edu

module load gcc/12.3.1
module load cuda
python3 ../inference.py --model_dir ../workdir/omegafold_model --ckpt epoch_25.pt --pdb_dir ../structures --embeddings_dir ../embeddings/omega_cameo_embeddings --embeddings_key name --elbo --num_samples 5 --alpha 1 --beta 3 --elbo_step 0.2 --splits ../splits/cameo_split_2.csv 
