'''
Hammad I., Akira N.
'''

'''
This script contains functions for embedding sequences using the ESM2 650M model.
You can pip install the required package with: pip install fair-esm
'''

## interact -m 16g ## 
"""
Usage:

ensure environment activated:
conda activate /users/anair27/data/DiffFolder/DiffFolder/env/difffolder_conda

 for two gpus:
sbatch bash_scripts/esm_embedding.sh -i /users/anair27/data/DiffFolder/DiffFolder/splits/limit256.csv -o /users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_256_all -w 2 -id 0
sbatch bash_scripts/esm_embedding.sh -i /users/anair27/data/DiffFolder/DiffFolder/splits/limit256.csv -o /users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_256_all -w 2 -id 1

 for four gpus:
sbatch bash_scripts/esm_embedding.sh -i /users/anair27/data/DiffFolder/DiffFolder/splits/limit256.csv -o /users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_256_all -w 4 -id 0
sbatch bash_scripts/esm_embedding.sh -i /users/anair27/data/DiffFolder/DiffFolder/splits/limit256.csv -o /users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_256_all -w 4 -id 1
sbatch bash_scripts/esm_embedding.sh -i /users/anair27/data/DiffFolder/DiffFolder/splits/limit256.csv -o /users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_256_all -w 4 -id 2
sbatch bash_scripts/esm_embedding.sh -i /users/anair27/data/DiffFolder/DiffFolder/splits/limit256.csv -o /users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_256_all -w 4 -id 3

 for inference
sbatch bash_scripts/esm_embedding_cpu.sh -i /users/anair27/data/DiffFolder/DiffFolder/splits/apo.csv -o /users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_apo -w 1 -id 0

sbatch bash_scripts/esm_embedding_cpu.sh -i /users/anair27/data/DiffFolder/DiffFolder/splits/cameo2022.csv -o /users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_cameo -w 1 -id 0

sbatch bash_scripts/esm_embedding_cpu.sh -i /users/anair27/data/DiffFolder/DiffFolder/splits/codnas.csv -o /users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_codnas -w 1 -id 0
"""
import torch
import esm
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import argparse

def get_model_and_alphabet(device='cuda'):
    '''Get the model and alphabet for embedding sequences'''
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    return model, alphabet

class ESMEmbeddingModel():
    '''Embed sequences using ESM2 650M model'''
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.alphabet = get_model_and_alphabet(device=self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    def embed(self, names, seqs, path_to_save, average_residues=True):
        '''Return a dict of embedded sequences

        Args:
            seqs: list of sequences to embed
            average_residues: whether to average the embeddings for each residue in the sequence (this is typically done to produce ESM embeddings for downstream tasks)
        '''
        embedded_seqs = {}
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        # embed sequences
        for i, (name, seq) in tqdm(enumerate(zip(names, seqs)), desc='Embedding sequences', total=len(seqs)):
            _, _, batch_tokens = self.batch_converter([("sequence", seq)])
            batch_tokens = batch_tokens.to(self.device)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33].cpu()
            del batch_tokens

            sequence_representations = []
            if average_residues:
                for j, tokens_len in enumerate(batch_lens):
                    sequence_representations.append(token_representations[j, 1 : tokens_len - 1].mean(0))
            else:
                for j, tokens_len in enumerate(batch_lens):
                    sequence_representations.append(token_representations[j, 1 : tokens_len - 1])
                # raise NotImplementedError('Not implemented yet')
            embedding_array = np.array(sequence_representations[0].cpu().numpy())
            # embedded_seqs.update({name: embedding_array})
            # np.savez(os.path.join(path_to_save, f"{name}"), embedding_array)
            path_to_file = os.path.join(path_to_save, str(name)[:2])
            if not os.path.exists(path_to_file):
                os.makedirs(path_to_file)
            np.savez(os.path.join(path_to_file, str(name)), **{'node_repr': embedding_array, 'edge_repr': np.zeros((2,2))})
            del embedding_array
            del sequence_representations

        return embedded_seqs

    def cpu(self):
        '''Move model to cpu'''
        self.model = self.model.cpu()
        return self

    def cuda(self):
        '''Move model to cuda'''
        self.model = self.model.cuda()
        return self

    def gpu(self):
        '''Move model to cuda'''
        self.model = self.model.cuda()
        return self
    

def load_data(path_to_csv):
    '''Given a path, read in csv and return a list of names and sequences to embed'''
    df = pd.read_csv(path_to_csv)
    # get names
    names = df['name'].tolist()
    # get sequences
    seqs = df['seqres'].tolist()
    return names, seqs

def save_embedding(embedding:dict, path_to_save: str):
    '''Save embedding to a file'''
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    for name in embedding:
        # get np array for name
        embedding_array = embedding[name]
        # save to file
        np.savez(os.path.join(path_to_save, f"{name}"), embedding_array)

def parse_args(args):
    parser = argparse.ArgumentParser()
    # arg for input csv
    parser.add_argument('-i', '--input', type=str, help='Path to input csv')
    # arg for output folder
    parser.add_argument('-o', '--output', type=str, help='Path to output folder')
    # arg -d for device
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device to use for embedding')
    # arg for number of workers
    parser.add_argument('-w', '--workers', type=int, default=1, help='Number of workers to use for embedding')
    # arg for worker id
    parser.add_argument('-id', '--worker_id', type=int, default=0, help='Worker id')
    return parser.parse_args(args)


def main(args):
    # create a ESM Embedding model
    args = parse_args(args)
    esm_embedding_model = ESMEmbeddingModel(device=args.device)
    print('Loaded model')

    # load data
    path_to_data = args.input
    names, seqs = load_data(path_to_data)
    # split names and seqs into # of batches where # is workers
    # len of data
    data_len = len(names)
    # get batch size
    batch_size = int(data_len / args.workers)
    # get start and end index
    start_idx = args.worker_id * batch_size
    end_idx = start_idx + batch_size
    # get names and seqs
    if args.worker_id == args.workers - 1:
        seqs = seqs[start_idx:]
        names = names[start_idx:]
    else:
        seqs = seqs[start_idx:end_idx]
        names = names[start_idx:end_idx]
    print(f'Loaded data with {len(seqs)} sequences, starting at index {start_idx} and ending at index {end_idx}')


    # embed sequences
    embedding : dict = esm_embedding_model.embed(names=names, seqs=seqs, path_to_save=args.output, average_residues=False)
    
    # save embeddings
    # path_to_save = args.output
    # save_embedding(embedding, path_to_save)
    print('Saved embedding')

if __name__ == '__main__':
    main(sys.argv[1:])




# python esm_embedding.py -i /users/anair27/data/DiffFolder/DiffFolder/splits/sampledlimit256_petite.csv -o /users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_256petite