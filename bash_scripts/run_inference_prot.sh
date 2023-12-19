#!/bin/sh
#SBATCH -N 1
    #node
#SBATCH -p rsingh47-gcondo 
#SBATCH--gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -n 4 #cores
#SBATCH --mem=256G
#SBATCH -t 24:00:00
#SBATCH -o run_inference__prot_apo_single.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=wangdrak_dorji@brown.edu

module load gcc/12.3.1
module load cuda
python3 ../inference.py --model_dir ../workdir/prot_model --ckpt epoch_25.pt --pdb_dir ../single_structure --embeddings_dir ../prot/apo_embeddings --embeddings_key name --elbo --num_samples 1 --alpha 1 --beta 3 --elbo_step 0.2 --splits ../splits/two_proteins.csv 
