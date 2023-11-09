'''
Hammad I., Akira N.
'''

'''
This script contains functions for embedding sequences using the ESM2 650M model.
You can pip install the required package with: pip install fair-esm
'''

## interact -m 16g ## 

import torch
import esm
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

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

    def embed(self, names, seqs, average_residues=True):
        '''Return a dict of embedded sequences

        Args:
            seqs: list of sequences to embed
            average_residues: whether to average the embeddings for each residue in the sequence (this is typically done to produce ESM embeddings for downstream tasks)
        '''
        embedded_seqs = {}

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

            embedded_seqs.update({name: sequence_representations[0].cpu().numpy()})

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
    seqs = df['seq'].tolist()
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


def main():
    # create a ESM Embedding model
    esm_embedding_model = ESMEmbeddingModel(device='cpu')
    print('Loaded model')

    # load data
    path_to_data = '/users/anair27/data/DiffFolder/DiffFolder/splits/codnas.csv'
    names, seqs = load_data(path_to_data)
    print(f'Loaded data with {len(seqs)} sequences')


    # embed sequences
    embedding : dict = esm_embedding_model.embed(names=names, seqs=seqs, average_residues=False)
    
    # save embeddings
    path_to_save = '/users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_codnas_embeddings'
    save_embedding(embedding, path_to_save)
    print('Saved embedding')

if __name__ == '__main__':
    main()