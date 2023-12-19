#!/bin/sh
#SBATCH -N 1
    #node
#SBATCH -p rsingh47-gcondo 
#SBATCH--gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -n 4 #cores
#SBATCH --mem=10G
#SBATCH -t 24:00:00
#SBATCH -o run_inference_single.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=wangdrak_dorji@brown.edu

module load gcc/12.3.1
module load cuda
python3 ../inference.py --model_dir ../workdir/esm_model --ckpt best_model.pt --pdb_dir ../single_structure --embeddings_dir ../esm_embeddings/esm_apo --embeddings_key name --elbo --num_samples 1 --alpha 1 --beta 3 --elbo_step 0.2 --splits ../splits/two_proteins.csv 
